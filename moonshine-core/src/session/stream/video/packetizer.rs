use aes_gcm::{
	Aes128Gcm, Key, Nonce,
	aead::{AeadInPlace, KeyInit},
};
use fec_rs::ReedSolomon;
use std::collections::{HashMap, hash_map::Entry};
use std::time::Instant;

use crate::session::SessionKeysReceiver;

use crate::session::stream::video::pyrowave_framing::RecordLayout;
use crate::session::stream::video::shard_batch::{ShardBatch, ShardBuf};

/// Maximum allowed number of shards in the encoder (data + parity).
pub(crate) const MAX_SHARDS: usize = 255;

pub(crate) const NV_VIDEO_PACKET_SIZE: usize = 16;
const RTP_HEADER_SIZE: usize = 12;
const PADDING_SIZE: usize = 4;
/// Byte offset where the NvVideoPacket starts within a shard.
const NV_PACKET_OFFSET: usize = RTP_HEADER_SIZE + PADDING_SIZE;
/// Byte offset where the payload starts within a shard.
const PAYLOAD_OFFSET: usize = NV_PACKET_OFFSET + NV_VIDEO_PACKET_SIZE;

/// Size of the per-shard encryption prefix: iv(12) + frameNumber(4) + tag(16).
const ENC_PREFIX_SIZE: usize = 12 + 4 + 16;

#[repr(u8)]
enum RtpFlag {
	ContainsPicData = 0x1,
	EndOfFrame = 0x2,
	StartOfFrame = 0x4,
}

/// `extraFlags` bit: the payload's frame data starts with a PyroWave record.
const EXTRA_FLAG_PYROWAVE_RECORD_START: u8 = 0x80;

const FRAME_TYPE_KEY: u8 = 2;
const FRAME_TYPE_PREDICTED: u8 = 1;

#[derive(Debug)]
#[repr(C)]
struct VideoFrameHeader {
	header_type: u8,
	frame_processing_latency: u16,
	frame_type: u8,
	last_payload_len: u16,
	/// PyroWave's critical packet count. Other clients ignore these bytes.
	pyrowave_critical_packets: u16,
}

pub(crate) const VIDEO_FRAME_HEADER_SIZE: usize = 8;

impl VideoFrameHeader {
	fn serialize(&self, buffer: &mut [u8]) {
		buffer[0] = self.header_type;
		buffer[1..3].copy_from_slice(&self.frame_processing_latency.to_le_bytes());
		buffer[3] = self.frame_type;
		buffer[4..6].copy_from_slice(&self.last_payload_len.to_le_bytes());
		buffer[6..8].copy_from_slice(&self.pyrowave_critical_packets.to_le_bytes());
	}
}

/// `fecInfo` holds the shard index and count in 10 bits each.
const MAX_DATA_SHARDS_WITHOUT_PARITY: usize = 1023;

/// One FEC block: data shards `[start, end)` plus `parity` parity shards.
/// Clients derive the parity count as `ceil(data * fec_percentage / 100)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BlockPlan {
	start: usize,
	end: usize,
	parity: usize,
	fec_percentage: usize,
}

fn plan_coded_blocks(nr_data_shards: usize, fec_percentage: u8, minimum_fec_packets: u32) -> Vec<BlockPlan> {
	let nr_parity_shards_per_block = MAX_SHARDS * fec_percentage as usize / (100 + fec_percentage as usize);
	let nr_data_shards_per_block = MAX_SHARDS - nr_parity_shards_per_block;

	// Subtract one so that e.g. 100 data shards at 100 per block is one block, not two.
	let nr_blocks = (nr_data_shards - 1) / nr_data_shards_per_block + 1;
	tracing::trace!(
		"Sending a max of {nr_data_shards_per_block} data shards and {nr_parity_shards_per_block} parity shards per block."
	);
	tracing::trace!("Sending {nr_blocks} blocks of video data.");
	if nr_blocks > 4 {
		tracing::debug!(
			"Trying to create {nr_blocks} blocks, but we are limited to 4 blocks so we are sending all remaining packets without FEC."
		);
	}

	(0..nr_blocks.min(4))
		.map(|block_index| {
			let start = block_index * nr_data_shards_per_block;
			let end = if block_index == 3 {
				nr_data_shards
			} else {
				((block_index + 1) * nr_data_shards_per_block).min(nr_data_shards)
			};
			let data = end - start;
			let parity = (data * fec_percentage as usize / 100)
				.max(minimum_fec_packets as usize)
				.min(MAX_SHARDS.saturating_sub(data));
			// Recompute the percentage for rounding and for blocks without parity.
			BlockPlan {
				start,
				end,
				parity,
				fec_percentage: parity * 100 / data,
			}
		})
		.collect()
}

/// Only critical packets carry parity, as block 0. The client minimum is
/// bounded by the Reed-Solomon block size and wire percentage field.
fn plan_pyrowave_blocks(
	nr_data_shards: usize,
	critical_packets: usize,
	critical_fec_percentage: u8,
	minimum_fec_packets: u32,
) -> Vec<BlockPlan> {
	let critical = critical_packets.min(nr_data_shards);
	let mut plan = Vec::with_capacity(4);
	let mut start = 0;
	if critical > 0 && critical_fec_percentage > 0 {
		// The wire stores an eight-bit percentage; the client rounds it up
		// to a parity count. Bound both that representation and the RS block.
		let maximum_percentage = (MAX_SHARDS.saturating_sub(critical) * 100 / critical).min(255);
		let fec_percentage = (critical_fec_percentage as usize)
			.max((minimum_fec_packets as usize * 100).div_ceil(critical))
			.min(maximum_percentage);
		let parity = (critical * fec_percentage).div_ceil(100);
		if parity > 0 {
			plan.push(BlockPlan {
				start: 0,
				end: critical,
				parity,
				fec_percentage,
			});
			start = critical;
		}
	}

	let remaining_blocks = 4 - plan.len();
	let rest = nr_data_shards - start;
	if rest > 0 {
		// Keep blocks at the usual 255 shards when that fits, otherwise grow them.
		let per_block = if rest <= remaining_blocks * MAX_SHARDS {
			MAX_SHARDS
		} else {
			rest.div_ceil(remaining_blocks).min(MAX_DATA_SHARDS_WITHOUT_PARITY)
		};
		while start < nr_data_shards && plan.len() < 4 {
			let end = (start + per_block).min(nr_data_shards);
			plan.push(BlockPlan {
				start,
				end,
				parity: 0,
				fec_percentage: 0,
			});
			start = end;
		}
	}
	plan
}

/// Write an RTP header directly into a byte slice at offset 0.
fn write_rtp_header(buf: &mut [u8], sequence_number: u16, timestamp: u32) {
	buf[0] = 0x90;
	buf[1] = 0; // packet_type
	buf[2..4].copy_from_slice(&sequence_number.to_be_bytes());
	buf[4..8].copy_from_slice(&timestamp.to_be_bytes());
	buf[8..12].copy_from_slice(&0u32.to_be_bytes()); // ssrc
}

/// Write an NvVideoPacket directly into a byte slice.
fn write_nv_video_packet(
	buf: &mut [u8],
	stream_packet_index: u32,
	frame_index: u32,
	flags: u8,
	extra_flags: u8,
	multi_fec_blocks: u8,
	fec_info: u32,
) {
	buf[0..4].copy_from_slice(&stream_packet_index.to_le_bytes());
	buf[4..8].copy_from_slice(&frame_index.to_le_bytes());
	buf[8] = flags;
	buf[9] = extra_flags;
	buf[10] = 0x10; // multi_fec_flags
	buf[11] = multi_fec_blocks;
	buf[12..16].copy_from_slice(&fec_info.to_le_bytes());
}

/// Copy bytes from the logical [header ++ encoded_data] stream into a
/// destination slice, without materializing the concatenation.
fn copy_header_and_data(
	dst: &mut [u8],
	header: &[u8; VIDEO_FRAME_HEADER_SIZE],
	encoded_data: &[u8],
	offset: usize,
	len: usize,
) {
	let total = VIDEO_FRAME_HEADER_SIZE + encoded_data.len();
	let end = (offset + len).min(total);
	let mut written = 0;

	if offset < VIDEO_FRAME_HEADER_SIZE {
		let header_end = VIDEO_FRAME_HEADER_SIZE.min(end);
		let n = header_end - offset;
		dst[written..written + n].copy_from_slice(&header[offset..header_end]);
		written += n;
		if end > VIDEO_FRAME_HEADER_SIZE {
			let n = end - VIDEO_FRAME_HEADER_SIZE;
			dst[written..written + n].copy_from_slice(&encoded_data[..n]);
		}
	} else {
		let data_start = offset - VIDEO_FRAME_HEADER_SIZE;
		let data_end = end - VIDEO_FRAME_HEADER_SIZE;
		let n = data_end - data_start;
		dst[written..written + n].copy_from_slice(&encoded_data[data_start..data_end]);
	}
}

struct FrameLayout<'a> {
	data: &'a [u8],
	payload_size: usize,
	nr_data_shards: usize,
	last_payload_len: u16,
}

impl<'a> FrameLayout<'a> {
	fn new(data: &'a [u8], requested_packet_size: usize) -> Result<Self, ()> {
		let payload_size = requested_packet_size
			.checked_sub(NV_VIDEO_PACKET_SIZE)
			.filter(|size| (1..=u16::MAX as usize).contains(size))
			.ok_or(())?;
		let packet_data_len = VIDEO_FRAME_HEADER_SIZE + data.len();
		let last_payload_len = match packet_data_len % payload_size {
			0 => payload_size,
			len => len,
		};
		Ok(Self {
			data,
			payload_size,
			nr_data_shards: packet_data_len.div_ceil(payload_size),
			last_payload_len: last_payload_len as u16,
		})
	}
}

pub(crate) struct Packetizer {
	fec_encoders: HashMap<(usize, usize), ReedSolomon>,
	/// Watch channel for encryption keys — read eagerly per `packetize()` call.
	keys_rx: SessionKeysReceiver,
	/// Whether video encryption is enabled by the client.
	encrypt: bool,
	/// AES-128-GCM cipher for video encryption, `None` when disabled or uninitialized.
	cipher: Option<Aes128Gcm>,
	/// Last seen `remote_input_key_id` — used to detect key rotation.
	last_key_id: i64,
	/// Monotonically increasing IV counter (one increment per encrypted shard).
	gcm_iv_counter: u64,
}

impl Packetizer {
	pub fn new(encrypt: bool, keys_rx: SessionKeysReceiver) -> Self {
		Self {
			fec_encoders: HashMap::new(),
			keys_rx,
			encrypt,
			cipher: None,
			last_key_id: i64::MIN,
			gcm_iv_counter: 0,
		}
	}

	/// Four FEC blocks fit on the wire; the last may hold up to 1023 data
	/// shards without parity. Reserve the short frame header from that capacity.
	pub fn frame_capacity(requested_packet_size: usize, fec_percentage: u8) -> Option<usize> {
		let payload = requested_packet_size.checked_sub(NV_VIDEO_PACKET_SIZE)?;
		if !(1..=u16::MAX as usize).contains(&payload) {
			return None;
		}
		let parity = MAX_SHARDS * fec_percentage as usize / (100 + fec_percentage as usize);
		Some((3 * (MAX_SHARDS - parity) + 1023) * payload - VIDEO_FRAME_HEADER_SIZE)
	}

	/// Conservative capacity with one block reserved for critical parity.
	pub fn pyrowave_frame_capacity(requested_packet_size: usize, critical_fec: bool) -> Option<usize> {
		Self::frame_capacity(requested_packet_size, 0)?;
		let blocks = if critical_fec { 3 } else { 4 };
		Some(
			blocks * MAX_DATA_SHARDS_WITHOUT_PARITY * (requested_packet_size - NV_VIDEO_PACKET_SIZE)
				- VIDEO_FRAME_HEADER_SIZE,
		)
	}

	/// Update the cipher if the encryption key has rotated.
	/// Called eagerly at the start of each `packetize()` call.
	fn maybe_update_cipher(&mut self) {
		if !self.encrypt {
			return;
		}
		let keys = &*self.keys_rx.borrow();
		if keys.remote_input_key_id == self.last_key_id {
			return;
		}
		self.last_key_id = keys.remote_input_key_id;
		if keys.remote_input_key.len() != 16 {
			tracing::error!(
				"Video encryption key must be exactly 16 bytes, got {}",
				keys.remote_input_key.len()
			);
			self.cipher = None;
			return;
		}
		let key = Key::<Aes128Gcm>::from_slice(&keys.remote_input_key);
		self.cipher = Some(Aes128Gcm::new(key));
		self.gcm_iv_counter = 0;
		tracing::debug!("Video encryption cipher updated for key_id={}", self.last_key_id);
	}

	/// Pre-create FEC encoders for all possible block sizes to avoid
	/// expensive ReedSolomon matrix construction during frame processing.
	pub fn warm_up(&mut self, fec_percentage: u8, minimum_fec_packets: u32) {
		let nr_parity_shards_per_block = MAX_SHARDS * fec_percentage as usize / (100 + fec_percentage as usize);
		let nr_data_shards_per_block = MAX_SHARDS - nr_parity_shards_per_block;

		for nr_data_shards in 1..=nr_data_shards_per_block {
			let nr_parity_shards = (nr_data_shards * fec_percentage as usize / 100)
				.max(minimum_fec_packets as usize)
				.min(MAX_SHARDS.saturating_sub(nr_data_shards));
			if nr_parity_shards > 0 {
				let _ = self.get_fec_encoder(nr_data_shards, nr_parity_shards);
			}
		}

		tracing::debug!("FEC encoder cache warmed with {} entries.", self.fec_encoders.len());
	}

	/// Packetize an H.264/HEVC/AV1 frame into a batch of network-ready shards.
	///
	/// Returns a `ShardBatch` containing all data + parity shards packed
	/// contiguously in a single allocation per block.
	#[allow(clippy::too_many_arguments)]
	pub fn packetize(
		&mut self,
		encoded_data: &[u8],
		is_key_frame: bool,
		requested_packet_size: usize,
		minimum_fec_packets: u32,
		fec_percentage: u8,
		frame_number: u32,
		sequence_number: &mut u32,
		rtp_timestamp: u32,
		frame_processing_latency: u16,
	) -> Result<ShardBatch, ()> {
		let frame = FrameLayout::new(encoded_data, requested_packet_size)?;
		let plan = plan_coded_blocks(frame.nr_data_shards, fec_percentage, minimum_fec_packets);
		let header = VideoFrameHeader {
			header_type: 0x01,
			frame_processing_latency,
			frame_type: if is_key_frame {
				FRAME_TYPE_KEY
			} else {
				FRAME_TYPE_PREDICTED
			},
			last_payload_len: frame.last_payload_len,
			pyrowave_critical_packets: 0,
		};
		self.emit(
			&frame,
			&header,
			&plan,
			&[],
			frame_number,
			sequence_number,
			rtp_timestamp,
		)
	}

	/// Packetize a record-framed PyroWave frame.
	#[allow(clippy::too_many_arguments)]
	pub fn packetize_pyrowave(
		&mut self,
		data: &[u8],
		records: &RecordLayout,
		requested_packet_size: usize,
		critical_fec_percentage: u8,
		minimum_fec_packets: u32,
		frame_number: u32,
		sequence_number: &mut u32,
		rtp_timestamp: u32,
		frame_processing_latency: u16,
	) -> Result<ShardBatch, ()> {
		let layout = FrameLayout::new(data, requested_packet_size)?;
		let plan = plan_pyrowave_blocks(
			layout.nr_data_shards,
			records.critical_packets as usize,
			critical_fec_percentage,
			minimum_fec_packets,
		);
		// The encoder's byte budget should prevent this; never send a cut-off frame.
		if plan.last().map(|block| block.end) != Some(layout.nr_data_shards) {
			tracing::warn!(
				"PyroWave frame {frame_number} needs {} packets, more than four FEC blocks hold.",
				layout.nr_data_shards
			);
			return Err(());
		}
		let header = VideoFrameHeader {
			header_type: 0x01,
			frame_processing_latency,
			frame_type: FRAME_TYPE_KEY,
			last_payload_len: layout.last_payload_len,
			pyrowave_critical_packets: records.critical_packets,
		};
		self.emit(
			&layout,
			&header,
			&plan,
			&records.record_starts,
			frame_number,
			sequence_number,
			rtp_timestamp,
		)
	}

	/// Write the frame's data shards per `plan`, compute each block's parity,
	/// and encrypt. `record_starts[i]` sets the PyroWave record-start flag on data shard `i`.
	#[allow(clippy::too_many_arguments)]
	fn emit(
		&mut self,
		frame: &FrameLayout,
		header: &VideoFrameHeader,
		plan: &[BlockPlan],
		record_starts: &[bool],
		frame_number: u32,
		sequence_number: &mut u32,
		rtp_timestamp: u32,
	) -> Result<ShardBatch, ()> {
		// Eagerly read current encryption key and update cipher if rotated.
		self.maybe_update_cipher();

		tracing::trace!(
			"Packetizing frame {frame_number}, size={}, keyframe={}, blocks={}",
			frame.data.len(),
			header.frame_type == FRAME_TYPE_KEY,
			plan.len()
		);

		let mut header_bytes = [0u8; VIDEO_FRAME_HEADER_SIZE];
		header.serialize(&mut header_bytes);

		let payload_size = frame.payload_size;
		let packet_data_len = VIDEO_FRAME_HEADER_SIZE + frame.data.len();
		let shard_size = PAYLOAD_OFFSET + payload_size;
		// When encryption is enabled, reserve space for the per-shard prefix.
		let prefix_size = if self.cipher.is_some() { ENC_PREFIX_SIZE } else { 0 };
		// The wire stores the last block index in bits 6-7, the current index in 4-5.
		let last_block_index = ((plan.len() - 1) as u8) << 6;

		// Accumulate all blocks into a single batch.
		let mut all_shards = ShardBatch::empty();

		let mut total_alloc_us = 0u128;
		let mut total_data_write_us = 0u128;
		let mut total_fec_encoder_us = 0u128;
		let mut total_fec_compute_us = 0u128;
		let mut total_fec_headers_us = 0u128;
		let mut total_extend_us = 0u128;

		for (block_index, block) in plan.iter().enumerate() {
			let nr_data_shards = block.end - block.start;
			assert!(nr_data_shards != 0);
			let nr_parity_shards = block.parity;
			let fec_percentage = block.fec_percentage;
			let multi_fec_blocks = ((block_index as u8) << 4) | last_block_index;

			let t_fec_encoder = Instant::now();
			let encoder = if nr_parity_shards > 0 {
				Some(self.get_fec_encoder(nr_data_shards, nr_parity_shards)?)
			} else {
				None
			};
			total_fec_encoder_us += t_fec_encoder.elapsed().as_micros();

			tracing::trace!(
				"Sending block {block_index} with {nr_data_shards} data shards and {nr_parity_shards} parity shards."
			);

			// Single allocation for all shards in this block (data + parity), zeroed.
			let total_shards = nr_data_shards + nr_parity_shards;
			let t_alloc = Instant::now();
			let mut shard_buf = ShardBuf::new(total_shards, shard_size, prefix_size);
			total_alloc_us += t_alloc.elapsed().as_micros();

			let t_data_write = Instant::now();

			// Write data shards directly into the flat buffer.
			for (block_shard_index, data_shard_index) in (block.start..block.end).enumerate() {
				let payload_start = data_shard_index * payload_size;
				let payload_len = payload_size.min(packet_data_len - payload_start);

				let shard = shard_buf.shard_mut(block_shard_index);

				// Write RTP header.
				write_rtp_header(shard, *sequence_number as u16, rtp_timestamp);

				// Padding (4 bytes of zeros) is already zeroed.

				// Write NvVideoPacket header.
				let mut flags = RtpFlag::ContainsPicData as u8;
				if block_shard_index == 0 {
					flags |= RtpFlag::StartOfFrame as u8;
				}
				if block_shard_index == nr_data_shards - 1 {
					flags |= RtpFlag::EndOfFrame as u8;
				}
				let extra_flags = if record_starts.get(data_shard_index).copied().unwrap_or(false) {
					EXTRA_FLAG_PYROWAVE_RECORD_START
				} else {
					0
				};
				write_nv_video_packet(
					&mut shard[NV_PACKET_OFFSET..NV_PACKET_OFFSET + NV_VIDEO_PACKET_SIZE],
					*sequence_number << 8,
					frame_number,
					flags,
					extra_flags,
					multi_fec_blocks,
					(block_shard_index << 12 | nr_data_shards << 22 | fec_percentage << 4) as u32,
				);

				// Copy payload from [header ++ encoded_data].
				copy_header_and_data(
					&mut shard[PAYLOAD_OFFSET..],
					&header_bytes,
					frame.data,
					payload_start,
					payload_len,
				);

				// Remaining bytes are already zero (padding for undersized last shard).

				*sequence_number += 1;
			}

			// Parity shards are already zeroed from ShardBuf::new().

			total_data_write_us += t_data_write.elapsed().as_micros();

			if let Some(encoder) = encoder {
				// Create FEC-compatible slice views into the flat buffer.
				let mut fec_slices = shard_buf.as_fec_slices();

				let t_fec_compute = Instant::now();

				encoder
					.encode(&mut fec_slices)
					.map_err(|e| tracing::warn!("Failed to encode packet as FEC shards: {e}"))?;

				total_fec_compute_us += t_fec_compute.elapsed().as_micros();

				let t_fec_headers = Instant::now();

				// Write headers for parity shards. FEC overwrites the entire shard
				// content, so we patch the fields Moonlight needs afterward.
				for block_shard_index in 0..nr_parity_shards {
					let shard = shard_buf.shard_mut(nr_data_shards + block_shard_index);

					// RTP header.
					shard[0] = 0x90;
					shard[1] = 0; // packet_type
					shard[2..4].copy_from_slice(&(*sequence_number as u16).to_be_bytes());

					// NvVideoPacket fields that Moonlight needs.
					let nv = &mut shard[NV_PACKET_OFFSET..NV_PACKET_OFFSET + NV_VIDEO_PACKET_SIZE];
					nv[4..8].copy_from_slice(&frame_number.to_le_bytes()); // frame_index
					nv[11] = multi_fec_blocks; // multi_fec_blocks
					let fec_info = ((nr_data_shards + block_shard_index) << 12
						| nr_data_shards << 22
						| fec_percentage << 4) as u32;
					nv[12..16].copy_from_slice(&fec_info.to_le_bytes()); // fec_info

					*sequence_number += 1;
				}

				total_fec_headers_us += t_fec_headers.elapsed().as_micros();
			}

			// Encrypt each shard if video encryption is enabled.
			if let Some(cipher) = &self.cipher {
				for shard_index in 0..total_shards {
					// Build the 12-byte IV: bytes 0..8 = counter (LE), byte 11 = 'V'.
					let mut iv = [0u8; 12];
					iv[..8].copy_from_slice(&self.gcm_iv_counter.to_le_bytes());
					iv[11] = b'V';
					self.gcm_iv_counter += 1;

					let nonce = Nonce::from_slice(&iv);

					// Encrypt the shard data in-place, returning a detached 16-byte tag.
					let shard_data = shard_buf.shard_mut(shard_index);
					let tag = cipher
						.encrypt_in_place_detached(nonce, b"", shard_data)
						.map_err(|e| tracing::warn!("Failed to encrypt video shard: {e}"))?;

					// Fill the encryption prefix: iv(12) + frameNumber(4) + tag(16).
					let prefix = shard_buf.prefix_mut(shard_index);
					prefix[..12].copy_from_slice(&iv);
					prefix[12..16].copy_from_slice(&frame_number.to_le_bytes());
					prefix[16..32].copy_from_slice(&tag);
				}
			}

			let t_extend = Instant::now();
			all_shards.extend_from(&shard_buf.into_batch());
			total_extend_us += t_extend.elapsed().as_micros();
		}

		tracing::trace!("Finished packetizing frame {frame_number}.");
		tracing::trace!(
			"Packetize breakdown: alloc_us={total_alloc_us} data_write_us={total_data_write_us} fec_encoder_us={total_fec_encoder_us} fec_compute_us={total_fec_compute_us} fec_headers_us={total_fec_headers_us} extend_us={total_extend_us}",
		);
		Ok(all_shards)
	}

	fn get_fec_encoder(&mut self, nr_data_shards: usize, nr_parity_shards: usize) -> Result<&mut ReedSolomon, ()> {
		Ok(match self.fec_encoders.entry((nr_data_shards, nr_parity_shards)) {
			Entry::Occupied(e) => {
				tracing::trace!("Found a FEC encoder for this combination of shards.");
				e.into_mut()
			},
			Entry::Vacant(e) => {
				tracing::trace!("No FEC encoder for this combination of shards, creating a new one.");
				let encoder = e.insert(
					ReedSolomon::new(nr_data_shards, nr_parity_shards)
						.map_err(|e| tracing::warn!("Couldn't create error correction encoder: {e}"))?,
				);
				tracing::trace!("Finished preparing FEC encoder.");

				encoder
			},
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::session::stream::video::pyrowave_framing::FramedPyroWave;

	#[test]
	fn coded_frame_fits_one_block_with_parity() {
		assert_eq!(
			plan_coded_blocks(10, 20, 2),
			vec![BlockPlan {
				start: 0,
				end: 10,
				parity: 2,
				fec_percentage: 20
			}]
		);
	}

	#[test]
	fn coded_frame_past_four_blocks_sends_the_rest_without_parity() {
		// 20% FEC leaves 213 data shards per block.
		let plan = plan_coded_blocks(1000, 20, 2);
		assert_eq!(plan.len(), 4);
		assert_eq!((plan[2].start, plan[2].end, plan[2].parity), (426, 639, 42));
		assert_eq!(
			plan[3],
			BlockPlan {
				start: 639,
				end: 1000,
				parity: 0,
				fec_percentage: 0
			}
		);
	}

	#[test]
	fn pyrowave_shards_carry_the_critical_count_and_record_flags() {
		const PACKET_SIZE: usize = 1392;
		let payload_size = PACKET_SIZE - NV_VIDEO_PACKET_SIZE;
		// Three payloads of frame data: the first two are critical, the third
		// starts in the middle of a record.
		let frame = FramedPyroWave {
			data: vec![0xAB; payload_size * 3 - VIDEO_FRAME_HEADER_SIZE - 100],
			layout: RecordLayout {
				record_starts: vec![true, true, false],
				critical_packets: 2,
			},
		};
		let mut sequence_number = 0;
		let batch = packetizer(false)
			.packetize_pyrowave(
				&frame.data,
				&frame.layout,
				PACKET_SIZE,
				20,
				2,
				7,
				&mut sequence_number,
				0,
				0,
			)
			.unwrap();

		// Two critical shards with two parity shards, then one unprotected shard.
		assert_eq!(batch.shard_count(), 5);
		assert_eq!(sequence_number, 5);
		let shards: Vec<&[u8]> = batch.as_bytes().chunks_exact(batch.shard_size()).collect();
		let extra_flags: Vec<u8> = shards.iter().map(|shard| shard[NV_PACKET_OFFSET + 9]).collect();
		assert_eq!(extra_flags[0], EXTRA_FLAG_PYROWAVE_RECORD_START);
		assert_eq!(extra_flags[1], EXTRA_FLAG_PYROWAVE_RECORD_START);
		assert_eq!(extra_flags[4], 0);

		let fec_info =
			|shard: &[u8]| u32::from_le_bytes(shard[NV_PACKET_OFFSET + 12..NV_PACKET_OFFSET + 16].try_into().unwrap());
		// Block 0: two data shards at 100% parity. Block 1: one data shard, none.
		assert_eq!(fec_info(shards[0]), 2 << 22 | 100 << 4);
		assert_eq!(fec_info(shards[3]), 2 << 22 | 3 << 12 | 100 << 4);
		assert_eq!(fec_info(shards[4]), 1 << 22);
		assert_eq!(shards[0][NV_PACKET_OFFSET + 11], 1 << 6);
		assert_eq!(shards[4][NV_PACKET_OFFSET + 11], 1 << 4 | 1 << 6);

		let header = &shards[0][PAYLOAD_OFFSET..PAYLOAD_OFFSET + VIDEO_FRAME_HEADER_SIZE];
		assert_eq!(header[3], FRAME_TYPE_KEY);
		assert_eq!(u16::from_le_bytes([header[4], header[5]]) as usize, payload_size - 100);
		assert_eq!(u16::from_le_bytes([header[6], header[7]]), 2);
	}

	#[test]
	fn pyrowave_frame_past_four_blocks_is_rejected() {
		const PACKET_SIZE: usize = 1392;
		let payload_size = PACKET_SIZE - NV_VIDEO_PACKET_SIZE;
		let frame = FramedPyroWave {
			data: vec![0; payload_size * (4 * MAX_DATA_SHARDS_WITHOUT_PARITY + 1)],
			layout: RecordLayout {
				record_starts: Vec::new(),
				critical_packets: 0,
			},
		};
		assert!(
			packetizer(false)
				.packetize_pyrowave(&frame.data, &frame.layout, PACKET_SIZE, 20, 2, 1, &mut 0, 0, 0)
				.is_err()
		);
	}

	#[test]
	fn pyrowave_protects_only_the_critical_packets() {
		let plan = plan_pyrowave_blocks(400, 30, 20, 2);
		assert_eq!(
			plan[0],
			BlockPlan {
				start: 0,
				end: 30,
				parity: 6,
				fec_percentage: 20
			}
		);
		assert!(plan[1..].iter().all(|b| b.parity == 0 && b.end - b.start <= MAX_SHARDS));
		assert_eq!(plan.last().unwrap().end, 400);
		assert!(plan.len() <= 4);
	}

	#[test]
	fn pyrowave_critical_parity_honors_the_client_minimum() {
		assert_eq!(plan_pyrowave_blocks(50, 4, 20, 2)[0].parity, 2);
		assert_eq!(plan_pyrowave_blocks(50, 4, 20, 7)[0].parity, 7);
		assert_eq!(plan_pyrowave_blocks(50, 4, 20, 0)[0].parity, 1);
		let block = plan_pyrowave_blocks(50, 1, 20, u32::MAX)[0];
		assert_eq!(block.fec_percentage, 255);
		assert_eq!(block.parity, 3);
		assert!(plan_pyrowave_blocks(50, 4, 0, 7).iter().all(|b| b.parity == 0));
	}

	#[test]
	fn pyrowave_without_a_critical_count_sends_no_parity() {
		let plan = plan_pyrowave_blocks(300, 0, 20, 2);
		assert!(plan.iter().all(|b| b.parity == 0));
		assert_eq!(plan.last().unwrap().end, 300);
	}

	#[test]
	fn large_pyrowave_frames_grow_blocks_past_255_shards() {
		let plan = plan_pyrowave_blocks(2900, 100, 20, 2);
		assert_eq!(plan.len(), 4);
		assert!(
			plan[1..]
				.iter()
				.all(|b| b.end - b.start <= MAX_DATA_SHARDS_WITHOUT_PARITY)
		);
		assert_eq!(plan.last().unwrap().end, 2900);
	}

	#[test]
	fn pyrowave_critical_parity_matches_what_the_client_derives() {
		for nr_data_shards in [1, 2, 7, 13, 99, 150, 230] {
			let block = plan_pyrowave_blocks(nr_data_shards + 10, nr_data_shards, 20, 2)[0];
			if block.parity > 0 {
				assert_eq!((nr_data_shards * block.fec_percentage).div_ceil(100), block.parity);
				assert!(block.parity >= 2);
			}
		}
	}

	fn packetizer(encrypted: bool) -> Packetizer {
		let (_, keys) = tokio::sync::watch::channel(crate::session::SessionKeyData {
			remote_input_key: vec![0x42; 16],
			remote_input_key_id: 1,
		});
		Packetizer::new(encrypted, keys)
	}

	fn reassemble(batch: &ShardBatch, encrypted: bool) -> Vec<u8> {
		let cipher = Aes128Gcm::new_from_slice(&[0x42; 16]).unwrap();
		let mut bytes = Vec::new();
		let mut last_payload_len = 0;
		let mut data_packets = 0;
		for wire in batch.as_bytes().chunks_exact(batch.shard_size()) {
			let packet = if encrypted {
				let mut data = wire[32..].to_vec();
				cipher
					.decrypt_in_place_detached(
						Nonce::from_slice(&wire[..12]),
						b"",
						&mut data,
						aes_gcm::Tag::from_slice(&wire[16..32]),
					)
					.unwrap();
				data
			} else {
				wire.to_vec()
			};
			let fec = u32::from_le_bytes(packet[28..32].try_into().unwrap());
			let index = (fec >> 12) & 0x3ff;
			let data_count = fec >> 22;
			if index >= data_count {
				continue;
			}
			assert_eq!(packet[25], 0);
			if data_packets == 0 {
				assert_eq!(packet[32], 1);
				assert_eq!(packet[35], 2);
				assert_eq!(&packet[38..40], &[0, 0]);
				last_payload_len = u16::from_le_bytes(packet[36..38].try_into().unwrap()) as usize;
			}
			bytes.extend_from_slice(&packet[32..]);
			data_packets += 1;
		}
		let payload = batch.shard_size() - if encrypted { 64 } else { 32 };
		bytes.truncate((data_packets - 1) * payload + last_payload_len);
		bytes.drain(..8);
		bytes
	}

	#[test]
	fn frames_preserve_data_across_blocks() {
		let data: Vec<u8> = (0..20_004).map(|i| (i % 251) as u8).collect();
		for encrypted in [false, true] {
			let batch = packetizer(encrypted)
				.packetize(&data, true, 80, 2, 20, 7, &mut 0, 3000, 10)
				.unwrap();
			assert!(batch.shard_count() > 255, "exercise multiple FEC blocks");
			assert_eq!(reassemble(&batch, encrypted), data);
		}
	}

	#[test]
	fn frame_capacity_matches_four_blocks() {
		// At 20%, three ordinary blocks hold 213 data shards each; the
		// unprotected fourth can represent 1023. Each payload here is 64 bytes.
		let capacity = 106_360;
		assert_eq!(Packetizer::frame_capacity(80, 20), Some(capacity));
		let data = vec![0x5a; capacity];
		let mut packetizer = packetizer(false);
		let batch = packetizer.packetize(&data, true, 80, 0, 20, 1, &mut 0, 0, 0).unwrap();
		assert_eq!(reassemble(&batch, false), data);
		for size in [0, 1, 15, 16, 65552, usize::MAX] {
			assert!(
				packetizer
					.packetize(&[0; 8], true, size, 0, 20, 1, &mut 0, 0, 0)
					.is_err()
			);
		}
	}

	#[test]
	fn invalid_sizes_and_unrepresentable_frames_are_rejected() {
		let (_, keys) = tokio::sync::watch::channel(crate::session::SessionKeyData {
			remote_input_key: Vec::new(),
			remote_input_key_id: 0,
		});
		let mut packetizer = Packetizer::new(false, keys);
		let layout = RecordLayout {
			record_starts: vec![true],
			critical_packets: 1,
		};
		for size in [0, 1, 15, 16, 65552, usize::MAX] {
			assert!(Packetizer::pyrowave_frame_capacity(size, true).is_none());
			assert!(
				packetizer
					.packetize_pyrowave(&[0; 100], &layout, size, 20, 2, 1, &mut 0, 0, 0)
					.is_err()
			);
		}
		assert_eq!(Packetizer::pyrowave_frame_capacity(80, true), Some(196_408));
		assert_eq!(Packetizer::pyrowave_frame_capacity(80, false), Some(261_880));
		assert!(
			packetizer
				.packetize_pyrowave(&vec![0; 262_144], &layout, 80, 20, 2, 1, &mut 0, 0, 0)
				.is_err()
		);
	}
}
