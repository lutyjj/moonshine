//! Independent GameStream reassembly for the loopback benchmark. Missing data
//! shards fail the check even when parity could recover them. Partial frames at
//! connection and shutdown are outside the measurement window.

use std::io::{self, Write};
use std::net::{SocketAddr, UdpSocket};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use pyrowave::{ChromaSampling, Decoder, Device, VideoFormat};

/// RTP header, four bytes of padding, then the 16-byte NV video packet header.
const NV_PACKET_OFFSET: usize = 16;
const PAYLOAD_OFFSET: usize = 32;
/// The short frame header that starts a frame's first payload.
const FRAME_HEADER_SIZE: usize = 8;
const PYROWAVE_SEQUENCE_HEADER_SIZE: usize = 8;
const PYROWAVE_PADDING_MARKER: u32 = 0xFFFF_FFFF;

pub struct PyroWaveStream {
	pub format: VideoFormat,
	pub fps: u32,
	/// Write the last decoded frame here as a single-frame Y4M file.
	pub dump: Option<PathBuf>,
}

#[derive(Debug, Default)]
pub struct Report {
	/// Complete frames, interior incomplete frames, and gaps in frame numbering.
	pub received: u64,
	pub complete: u64,
	/// Decoding samples the stream independently of the completeness check.
	pub decode_attempts: u64,
	pub decoded: u64,
}

impl Report {
	pub fn validate(&self, decoding: bool) -> io::Result<()> {
		if self.complete == 0 {
			return Err(io::Error::other("decode check received no complete frame"));
		}
		if self.received != self.complete {
			return Err(io::Error::other(format!(
				"decode check received {} of {} frames whole",
				self.complete, self.received
			)));
		}
		if decoding && (self.decode_attempts == 0 || self.decoded != self.decode_attempts) {
			return Err(io::Error::other(format!(
				"decode check decoded {} of {} sampled frames",
				self.decoded, self.decode_attempts
			)));
		}
		Ok(())
	}
}

type Worker = JoinHandle<io::Result<(u64, u64)>>;

pub struct DecodeCheck {
	stop: Arc<AtomicBool>,
	receiver: Option<Worker>,
	decoder: Option<Worker>,
}

impl DecodeCheck {
	/// Announce a client to the video stream at `video_addr` and start receiving.
	pub fn start(video_addr: SocketAddr, pyrowave: Option<PyroWaveStream>) -> std::io::Result<Self> {
		let socket = UdpSocket::bind("127.0.0.1:0")?;
		let socket_ref = socket2::SockRef::from(&socket);
		socket_ref.set_recv_buffer_size(32 << 20)?;
		let receive_buffer = socket_ref.recv_buffer_size()?;
		if receive_buffer < 32 << 20 {
			tracing::warn!(receive_buffer, "Decode-check receive buffer was limited by the kernel");
		}
		socket.set_read_timeout(Some(Duration::from_millis(100)))?;
		socket.send_to(b"PING", video_addr)?;

		let stop = Arc::new(AtomicBool::new(false));
		let (frame_tx, frame_rx) = mpsc::sync_channel::<Vec<u8>>(1);
		let decoder = pyrowave.map(|stream| std::thread::spawn(move || decode_frames(stream, frame_rx)));
		let decoding = decoder.is_some();

		let receiver = std::thread::spawn({
			let stop = stop.clone();
			move || {
				let mut assembler = Assembler::default();
				let mut buf = vec![0u8; 65536];
				while !stop.load(Ordering::Relaxed) {
					let len = match socket.recv(&mut buf) {
						Ok(len) => len,
						Err(e)
							if matches!(
								e.kind(),
								io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
							) =>
						{
							continue;
						},
						Err(e) => return Err(e),
					};
					if let Some(frame) = assembler.push(&buf[..len])?
						&& decoding
					{
						let _ = frame_tx.try_send(frame);
					}
				}
				Ok((assembler.received, assembler.complete))
			}
		});

		Ok(Self {
			stop,
			receiver: Some(receiver),
			decoder,
		})
	}

	pub fn finish(mut self) -> io::Result<Report> {
		self.join()
	}

	fn join(&mut self) -> io::Result<Report> {
		self.stop.store(true, Ordering::Relaxed);
		let join = |worker: Option<Worker>| {
			worker.map_or(Ok((0, 0)), |worker| {
				worker
					.join()
					.map_err(|_| io::Error::other("decode-check worker panicked"))?
			})
		};
		// Join both workers even if either fails; closing the receiver releases the decoder.
		let received = join(self.receiver.take());
		let decoded = join(self.decoder.take());
		let (received, complete) = received?;
		let (decode_attempts, decoded) = decoded?;
		Ok(Report {
			received,
			complete,
			decode_attempts,
			decoded,
		})
	}
}

impl Drop for DecodeCheck {
	fn drop(&mut self) {
		let _ = self.join();
	}
}

#[derive(Default)]
struct Block {
	shards: Vec<Option<Vec<u8>>>,
	received: usize,
}

#[derive(Default)]
struct Assembler {
	frame_index: Option<u32>,
	blocks: Vec<Block>,
	complete_blocks: usize,
	first_frame: bool,
	emitted: bool,
	received: u64,
	complete: u64,
}

impl Assembler {
	fn push(&mut self, packet: &[u8]) -> io::Result<Option<Vec<u8>>> {
		let invalid = || io::Error::other("malformed GameStream video packet");
		if packet.len() <= PAYLOAD_OFFSET {
			return Err(invalid());
		}
		let nv = &packet[NV_PACKET_OFFSET..PAYLOAD_OFFSET];
		let frame_index = u32::from_le_bytes(nv[4..8].try_into().unwrap());
		let block = usize::from((nv[11] >> 4) & 0x3);
		let block_count = usize::from(nv[11] >> 6) + 1;
		let fec_info = u32::from_le_bytes(nv[12..16].try_into().unwrap());
		let shard_index = ((fec_info >> 12) & 0x3ff) as usize;
		let data_shards = ((fec_info >> 22) & 0x3ff) as usize;

		if block >= block_count || data_shards == 0 {
			return Err(invalid());
		}
		if self.frame_index != Some(frame_index) {
			if let Some(previous) = self.frame_index {
				let distance = frame_index.wrapping_sub(previous);
				if distance >= 1 << 31 {
					return Ok(None);
				}
				self.received += u64::from(distance - 1);
				if !self.emitted && !self.first_frame {
					self.received += 1;
				}
			}
			self.first_frame = self.frame_index.is_none();
			self.frame_index = Some(frame_index);
			self.blocks = (0..block_count).map(|_| Block::default()).collect();
			self.complete_blocks = 0;
			self.emitted = false;
		}
		if block_count != self.blocks.len() {
			return Err(invalid());
		}
		if self.emitted || shard_index >= data_shards {
			return Ok(None);
		}
		let block = &mut self.blocks[block];
		if block.shards.is_empty() {
			block.shards.resize(data_shards, None);
		} else if block.shards.len() != data_shards {
			return Err(invalid());
		}
		if block.shards[shard_index].is_some() {
			return Ok(None);
		}
		block.shards[shard_index] = Some(packet[PAYLOAD_OFFSET..].to_vec());
		block.received += 1;
		if block.received == data_shards {
			self.complete_blocks += 1;
		}
		if self.complete_blocks != self.blocks.len() {
			return Ok(None);
		}

		let payloads: Vec<&[u8]> = self
			.blocks
			.iter()
			.flat_map(|block| &block.shards)
			.flatten()
			.map(Vec::as_slice)
			.collect();
		let header = payloads[0].get(..FRAME_HEADER_SIZE).ok_or_else(invalid)?;
		if header[0] != 1 {
			return Err(invalid());
		}
		let last_payload_len = usize::from(u16::from_le_bytes([header[4], header[5]]));
		let mut frame = payloads[..payloads.len() - 1].concat();
		frame.extend_from_slice(payloads.last().unwrap().get(..last_payload_len).ok_or_else(invalid)?);
		if frame.len() <= FRAME_HEADER_SIZE {
			return Err(invalid());
		}
		self.emitted = true;
		self.received += 1;
		self.complete += 1;
		Ok(Some(frame.split_off(FRAME_HEADER_SIZE)))
	}
}

/// A record-framed PyroWave frame without its padding records, as the codec's
/// own packets.
fn strip_padding_records(frame: &[u8]) -> Option<Vec<u8>> {
	let mut records = frame.get(..PYROWAVE_SEQUENCE_HEADER_SIZE)?.to_vec();
	let mut offset = PYROWAVE_SEQUENCE_HEADER_SIZE;
	while offset < frame.len() {
		let word = |at: usize| Some(u32::from_le_bytes(frame.get(at..at + 4)?.try_into().ok()?));
		let padding = word(offset)? == PYROWAVE_PADDING_MARKER;
		let len = if padding {
			8usize.checked_add((word(offset + 4)? as usize).checked_mul(4)?)?
		} else {
			((word(offset)? >> 16) & 0xfff) as usize * 4
		};
		if len < 8 {
			return None;
		}
		let end = offset.checked_add(len)?;
		let record = frame.get(offset..end)?;
		if !padding {
			records.extend_from_slice(record);
		}
		offset = end;
	}
	Some(records)
}

fn decode_frames(stream: PyroWaveStream, frames: mpsc::Receiver<Vec<u8>>) -> io::Result<(u64, u64)> {
	let format = stream.format;
	let mut decoder = Device::new()
		.and_then(|device| Decoder::new(device, format))
		.map_err(io::Error::other)?;
	let luma = (format.width * format.height) as usize;
	let chroma = match format.chroma {
		ChromaSampling::Yuv420 => luma / 4,
		ChromaSampling::Yuv444 => luma,
	};
	let mut planes = [vec![0u8; luma], vec![0u8; chroma], vec![0u8; chroma]];
	let (mut attempts, mut decoded) = (0, 0);

	for frame in frames {
		attempts += 1;
		// A repeated frame carries the sequence number of the one before it, and
		// the decoder would discard it as already seen.
		decoder.clear();
		let ok = strip_padding_records(&frame).is_some_and(|records| {
			decoder.push_packet(&records).is_ok()
				&& decoder.is_ready(false)
				&& decoder.decode_cpu(planes.each_mut().map(Vec::as_mut_slice)).is_ok()
		});
		decoded += u64::from(ok);
	}

	if let Some(path) = stream.dump.filter(|_| decoded > 0) {
		let colorspace = match format.chroma {
			ChromaSampling::Yuv420 => "C420jpeg",
			ChromaSampling::Yuv444 => "C444",
		};
		let header = format!(
			"YUV4MPEG2 W{} H{} F{}:1 Ip A1:1 {colorspace} XCOLORRANGE=FULL\nFRAME\n",
			format.width, format.height, stream.fps
		);
		let written = std::fs::File::create(&path).and_then(|mut file| {
			file.write_all(header.as_bytes())?;
			planes.iter().try_for_each(|plane| file.write_all(plane))
		});
		written?;
		tracing::info!("Wrote the last decoded frame to {}", path.display());
	}
	Ok((attempts, decoded))
}

#[cfg(test)]
mod tests {
	use super::*;

	// GameStream wire offsets, independent of the parser's constants.
	fn shard(frame: u32, block: u8, blocks: u8, index: u32, count: u32, payload: &[u8]) -> Vec<u8> {
		let mut packet = vec![0u8; 32];
		packet[0] = 0x90;
		packet[20..24].copy_from_slice(&frame.to_le_bytes());
		packet[27] = (block << 4) | ((blocks - 1) << 6);
		packet[28..32].copy_from_slice(&(index << 12 | count << 22).to_le_bytes());
		packet.extend_from_slice(payload);
		packet
	}

	fn complete(frame: u32) -> Vec<u8> {
		shard(frame, 0, 1, 0, 1, &[1, 0, 0, 2, 12, 0, 0, 0, 9, 8, 7, 6])
	}

	#[test]
	fn reassembles_multiple_blocks_out_of_order_without_counting_duplicates_or_parity() {
		let mut assembler = Assembler::default();
		let last = shard(7, 1, 2, 0, 1, b"ghi\0\0\0\0\0");
		assert!(assembler.push(&last).unwrap().is_none());
		assert!(assembler.push(&last).unwrap().is_none());
		assert!(assembler.push(&shard(7, 0, 2, 1, 2, b"abcdefgh")).unwrap().is_none());
		assert!(assembler.push(&shard(7, 0, 2, 2, 2, b"parity!!")).unwrap().is_none());
		let first = shard(7, 0, 2, 0, 2, &[1, 0, 0, 2, 3, 0, 0, 0]);
		assert_eq!(assembler.push(&first).unwrap().unwrap(), b"abcdefghghi");
		assert_eq!((assembler.received, assembler.complete), (1, 1));
	}

	#[test]
	fn counts_interior_loss_and_frame_gaps_but_excludes_partial_boundaries() {
		let mut assembler = Assembler::default();
		assembler.push(&shard(1, 0, 1, 0, 2, &[0; 8])).unwrap();
		assembler.push(&complete(2)).unwrap();
		assembler.push(&shard(3, 0, 1, 0, 2, &[0; 8])).unwrap();
		assembler.push(&complete(5)).unwrap();
		assembler.push(&complete(2)).unwrap();
		assembler.push(&shard(6, 0, 1, 0, 2, &[0; 8])).unwrap();
		assert_eq!((assembler.received, assembler.complete), (4, 2));
	}

	#[test]
	fn validates_the_frame_before_counting_it() {
		for payload in [
			vec![1; 4],
			vec![0; 12],
			vec![1, 0, 0, 2, 4, 0, 0, 0],
			vec![1, 0, 0, 2, 99, 0, 0, 0],
		] {
			let mut assembler = Assembler::default();
			assert!(assembler.push(&shard(7, 0, 1, 0, 1, &payload)).is_err());
			assert_eq!(assembler.complete, 0);
		}
	}

	#[test]
	fn frame_numbers_wrap() {
		let mut assembler = Assembler::default();
		assembler.push(&complete(u32::MAX)).unwrap();
		assembler.push(&complete(0)).unwrap();
		assert_eq!((assembler.received, assembler.complete), (2, 2));
	}

	#[test]
	fn verdict_requires_complete_frames_and_nonempty_successful_decoding() {
		let report = |received, complete, decode_attempts, decoded| Report {
			received,
			complete,
			decode_attempts,
			decoded,
		};
		assert!(Report::default().validate(false).is_err());
		assert!(report(2, 1, 1, 1).validate(true).is_err());
		assert!(report(2, 2, 0, 0).validate(true).is_err());
		assert!(report(2, 2, 2, 1).validate(true).is_err());
		assert!(report(2, 2, 1, 1).validate(true).is_ok());
		assert!(report(2, 2, 0, 0).validate(false).is_ok());
	}

	#[test]
	fn worker_errors_and_panics_fail_the_check() {
		for panics in [false, true] {
			let check = DecodeCheck {
				stop: Arc::new(AtomicBool::new(false)),
				receiver: Some(std::thread::spawn(|| Ok((1, 1)))),
				decoder: Some(std::thread::spawn(move || {
					assert!(!panics, "decoder panic");
					Err(io::Error::other("decoder initialization failed"))
				})),
			};
			assert!(check.finish().is_err());
		}
	}

	#[test]
	fn drop_stops_and_joins_workers() {
		let stop = Arc::new(AtomicBool::new(false));
		let stopped = Arc::new(AtomicBool::new(false));
		let receiver = std::thread::spawn({
			let stop = stop.clone();
			let stopped = stopped.clone();
			move || {
				while !stop.load(Ordering::Relaxed) {
					std::thread::yield_now();
				}
				stopped.store(true, Ordering::Relaxed);
				Ok((0, 0))
			}
		});
		drop(DecodeCheck {
			stop,
			receiver: Some(receiver),
			decoder: None,
		});
		assert!(stopped.load(Ordering::Relaxed));
	}

	#[test]
	fn padding_records_are_bounded_and_removed() {
		let sequence = [0, 0, 0, 128, 1, 0, 0, 0];
		let block = [0, 0, 2, 0, 0, 0, 0, 0];
		let padding = [255, 255, 255, 255, 1, 0, 0, 0, 0, 0, 0, 0];
		let frame = [sequence.as_slice(), &padding, &block].concat();
		assert_eq!(strip_padding_records(&frame).unwrap(), [sequence, block].concat());
		assert!(strip_padding_records(&frame[..18]).is_none());
		assert!(strip_padding_records(&[sequence, [0, 0, 1, 0, 0, 0, 0, 0]].concat()).is_none());
	}
}
