//! PyroWave record framing for the GameStream video stream.
//!
//! The wire contract is "PyroWave over the GameStream/Sunshine protocol" as
//! published by the PyroWave-capable Moonlight fork (`docs/pyrowave-protocol.md`
//! in Nonary/moonlight-qt). A frame is a sequence of 32-bit aligned records: one
//! PyroWave sequence header, the block records, and padding records. Records
//! are laid out so that payload boundaries fall between records wherever
//! possible, which lets a client decode a frame that lost packets: it skips the
//! damaged records and blurs only the detail they carried.
//!
//! Records are placed in two groups. The first holds PyroWave's coarsest
//! wavelet level, without which nothing decodes; its packets (the "critical"
//! packets) are announced to the client and are the only ones protected by FEC.

use super::packetizer::VIDEO_FRAME_HEADER_SIZE as FRAME_HEADER_SIZE;

const SEQUENCE_HEADER_SIZE: usize = 8;
const BLOCK_HEADER_SIZE: usize = 8;
/// Smallest padding record: the marker word and the word count.
const MIN_PADDING_SIZE: usize = 8;
const PADDING_MARKER: u32 = 0xFFFF_FFFF;

pub(crate) struct FramedPyroWave {
	/// Frame data that follows the short frame header.
	pub data: Vec<u8>,
	pub layout: RecordLayout,
}

pub(crate) struct RecordLayout {
	/// Per payload, whether its frame data starts with a record.
	pub record_starts: Vec<bool>,
	/// Number of leading payloads holding the coarsest wavelet level.
	pub critical_packets: u16,
}

#[derive(Clone, Copy, Debug)]
struct Record {
	offset: usize,
	len: usize,
}

/// `bitstream` is the output of `pyrowave_encoder_packetize` as a single packet:
/// the sequence header followed by the block records. Blocks with an index below
/// `critical_blocks` form the coarsest level. `payload_size` is the number of
/// bytes each video packet carries after its NV video header. Padding uses only
/// the space left within `capacity`; records may span payloads when it runs out.
pub(crate) fn frame(
	bitstream: &[u8],
	critical_blocks: usize,
	payload_size: usize,
	capacity: usize,
) -> Result<FramedPyroWave, String> {
	if payload_size < 64 || !payload_size.is_multiple_of(4) {
		return Err(format!("unsupported PyroWave payload size {payload_size}"));
	}
	if bitstream.len() > capacity {
		return Err("PyroWave bitstream exceeds transport capacity".into());
	}
	let sequence_header = bitstream
		.get(..SEQUENCE_HEADER_SIZE)
		.ok_or("PyroWave bitstream is shorter than its sequence header")?;

	let mut critical = Vec::new();
	let mut detail = Vec::new();
	let mut offset = SEQUENCE_HEADER_SIZE;
	while offset < bitstream.len() {
		let header = bitstream
			.get(offset..offset + BLOCK_HEADER_SIZE)
			.ok_or("truncated PyroWave block header")?;
		let word0 = u32::from_le_bytes(header[0..4].try_into().unwrap());
		let word1 = u32::from_le_bytes(header[4..8].try_into().unwrap());
		let len = ((word0 >> 16) & 0xFFF) as usize * 4;
		let block_index = (word1 >> 8) as usize;
		if len < BLOCK_HEADER_SIZE || offset + len > bitstream.len() {
			return Err(format!("malformed PyroWave block record at offset {offset}"));
		}
		let record = Record { offset, len };
		if block_index < critical_blocks {
			critical.push(record);
		} else {
			detail.push(record);
		}
		offset += len;
	}

	let mut writer = Writer::new(bitstream, payload_size, capacity);
	writer.write(sequence_header);
	writer.place_group(&critical);
	let critical_packets = writer.payloads_touched();
	writer.place_group(&detail);

	Ok(FramedPyroWave {
		layout: RecordLayout {
			critical_packets: u16::try_from(critical_packets).map_err(|_| "too many critical packets")?,
			record_starts: writer.record_starts(),
		},
		data: writer.data,
	})
}

struct Writer<'a> {
	bitstream: &'a [u8],
	payload_size: usize,
	data: Vec<u8>,
	padding_left: usize,
	/// Payload indices whose frame data starts with a record.
	starts: Vec<usize>,
}

impl<'a> Writer<'a> {
	fn new(bitstream: &'a [u8], payload_size: usize, capacity: usize) -> Self {
		Self {
			bitstream,
			payload_size,
			data: Vec::with_capacity((bitstream.len() + payload_size).min(capacity)),
			padding_left: capacity - bitstream.len(),
			// The first payload starts its frame data with the sequence header.
			starts: vec![0],
		}
	}

	/// Position in the payload stream, counting the short frame header.
	fn position(&self) -> usize {
		FRAME_HEADER_SIZE + self.data.len()
	}

	/// Bytes left in the current payload, in `1..=payload_size`.
	fn remaining(&self) -> usize {
		self.payload_size - self.position() % self.payload_size
	}

	fn payloads_touched(&self) -> usize {
		self.position().div_ceil(self.payload_size)
	}

	fn write(&mut self, bytes: &[u8]) {
		let position = self.position();
		if position.is_multiple_of(self.payload_size) {
			self.starts.push(position / self.payload_size);
		}
		self.data.extend_from_slice(bytes);
	}

	fn write_record(&mut self, record: Record) {
		let bytes = &self.bitstream[record.offset..record.offset + record.len];
		self.write(bytes);
	}

	fn pad(&mut self, len: usize) {
		debug_assert!(len >= MIN_PADDING_SIZE && len.is_multiple_of(4) && len <= self.padding_left);
		self.padding_left -= len;
		let words = (len - MIN_PADDING_SIZE) / 4;
		self.write(&PADDING_MARKER.to_le_bytes());
		self.data.extend_from_slice(&(words as u32).to_le_bytes());
		self.data.resize(self.data.len() + words * 4, 0);
	}

	/// Whether `len` bytes can go into the current payload without leaving a
	/// 4-byte tail, which is too small for a padding record.
	fn fits(&self, len: usize) -> bool {
		let remaining = self.remaining();
		len <= remaining && remaining - len != 4
	}

	fn place_group(&mut self, records: &[Record]) {
		// Records that cannot share a payload with a padding record span payloads.
		// Keep their end off a 4-byte tail so the following record still fits.
		let largest_shared = self.payload_size - MIN_PADDING_SIZE;
		for &record in records.iter().filter(|r| r.len > largest_shared) {
			let tail = (self.position() + record.len) % self.payload_size;
			if tail != 0 && self.payload_size - tail == 4 && self.padding_left >= MIN_PADDING_SIZE {
				self.pad(MIN_PADDING_SIZE);
			}
			self.write_record(record);
		}

		// Prefer keeping smaller records within a payload. Fill gaps with the
		// largest later record that fits, then padding if capacity permits.
		let fitting: Vec<Record> = records.iter().copied().filter(|r| r.len <= largest_shared).collect();
		let mut used = vec![false; fitting.len()];
		// Fill candidates by length in words; each bucket pops its earliest record.
		let mut by_words: Vec<Vec<usize>> = vec![Vec::new(); self.payload_size / 4 + 1];
		for (index, record) in fitting.iter().enumerate().rev() {
			by_words[record.len / 4].push(index);
		}

		let mut next = 0;
		loop {
			while next < fitting.len() && used[next] {
				next += 1;
			}
			let Some(&record) = fitting.get(next) else { break };

			if self.fits(record.len) {
				used[next] = true;
				self.write_record(record);
				continue;
			}

			match self.take_fill(&mut by_words, &used, &fitting) {
				Some(index) => {
					used[index] = true;
					self.write_record(fitting[index]);
				},
				None if self.remaining() >= MIN_PADDING_SIZE && self.remaining() <= self.padding_left => {
					self.pad(self.remaining());
				},
				None => {
					// At the transport limit, preserve codec bytes by spanning payloads.
					// The client skips a damaged record until the next flagged boundary.
					used[next] = true;
					self.write_record(record);
				},
			}
		}
	}

	/// The largest unused record that fits the current payload.
	fn take_fill(&self, by_words: &mut [Vec<usize>], used: &[bool], records: &[Record]) -> Option<usize> {
		let remaining = self.remaining();
		for words in (BLOCK_HEADER_SIZE / 4..=remaining / 4).rev() {
			if remaining - words * 4 == 4 {
				continue;
			}
			let bucket = &mut by_words[words];
			while let Some(&index) = bucket.last() {
				bucket.pop();
				if !used[index] {
					debug_assert_eq!(records[index].len, words * 4);
					return Some(index);
				}
			}
		}
		None
	}

	fn record_starts(&self) -> Vec<bool> {
		let mut starts = vec![false; self.payloads_touched()];
		for &payload in &self.starts {
			if let Some(start) = starts.get_mut(payload) {
				*start = true;
			}
		}
		starts
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	const PAYLOAD: usize = 1376;

	fn block(index: u32, len: usize) -> Vec<u8> {
		let words = (len / 4) as u32;
		let mut record = Vec::with_capacity(len);
		record.extend_from_slice(&(words << 16).to_le_bytes());
		record.extend_from_slice(&(index << 8).to_le_bytes());
		record.resize(len, 0xAB);
		record
	}

	fn bitstream(blocks: &[(u32, usize)]) -> Vec<u8> {
		let mut stream = vec![0x11; SEQUENCE_HEADER_SIZE];
		for &(index, len) in blocks {
			stream.extend(block(index, len));
		}
		stream
	}

	/// Parse a laid-out frame the way a receiver does: sequentially. Returns each
	/// record after the sequence header as (block index or `None` for padding,
	/// start, end) in frame-data offsets.
	fn records(frame: &FramedPyroWave) -> Vec<(Option<u32>, usize, usize)> {
		let data = &frame.data;
		let mut offset = SEQUENCE_HEADER_SIZE;
		let mut records = Vec::new();
		while offset < data.len() {
			let word0 = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
			let word1 = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());
			let (block, len) = if word0 == PADDING_MARKER {
				(None, MIN_PADDING_SIZE + word1 as usize * 4)
			} else {
				(Some(word1 >> 8), ((word0 >> 16) & 0xFFF) as usize * 4)
			};
			records.push((block, offset, offset + len));
			offset += len;
		}
		assert_eq!(offset, data.len());
		records
	}

	fn parsed_blocks(frame: &FramedPyroWave) -> Vec<u32> {
		records(frame).into_iter().filter_map(|(block, _, _)| block).collect()
	}

	#[test]
	fn every_block_is_delivered_once() {
		let blocks: Vec<(u32, usize)> = (0..400).map(|i| (i, 8 + (i as usize * 52) % 900)).collect();
		let framed = frame(&bitstream(&blocks), 40, PAYLOAD, usize::MAX).unwrap();
		assert_eq!(&framed.data[..SEQUENCE_HEADER_SIZE], &[0x11; SEQUENCE_HEADER_SIZE]);
		let mut parsed = parsed_blocks(&framed);
		parsed.sort_unstable();
		assert_eq!(parsed, (0..400).collect::<Vec<_>>());
	}

	#[test]
	fn critical_blocks_come_first_and_end_in_the_last_critical_packet() {
		let blocks: Vec<(u32, usize)> = (0..300).map(|i| (i, 8 + (i as usize * 36) % 700)).collect();
		let framed = frame(&bitstream(&blocks), 50, PAYLOAD, usize::MAX).unwrap();
		let parsed = parsed_blocks(&framed);
		assert!(parsed[..50].iter().all(|&b| b < 50));
		assert!(parsed[50..].iter().all(|&b| b >= 50));

		let critical_end = records(&framed)
			.into_iter()
			.filter(|(block, _, _)| block.is_some_and(|b| b < 50))
			.map(|(_, _, end)| end)
			.max()
			.unwrap();
		let packets = (FRAME_HEADER_SIZE + critical_end).div_ceil(PAYLOAD);
		assert_eq!(framed.layout.critical_packets as usize, packets);
	}

	#[test]
	fn small_records_stay_within_payloads_when_padding_fits() {
		let blocks: Vec<(u32, usize)> = (0..500).map(|i| (i, 8 + (i as usize * 44) % 1200)).collect();
		let framed = frame(&bitstream(&blocks), 30, PAYLOAD, usize::MAX).unwrap();
		for (_, start, end) in records(&framed) {
			let (start, end) = (FRAME_HEADER_SIZE + start, FRAME_HEADER_SIZE + end);
			if end - start <= PAYLOAD - MIN_PADDING_SIZE {
				assert_eq!(
					start / PAYLOAD,
					(end - 1) / PAYLOAD,
					"record {start}..{end} crosses a payload"
				);
			}
		}
	}

	#[test]
	fn payloads_after_the_spanning_records_start_with_a_record() {
		let blocks: Vec<(u32, usize)> = (0..500).map(|i| (i, 8 + (i as usize * 44) % 1200)).collect();
		let framed = frame(&bitstream(&blocks), 30, PAYLOAD, usize::MAX).unwrap();
		let offsets: std::collections::HashSet<usize> = records(&framed)
			.into_iter()
			.map(|(_, start, _)| start + FRAME_HEADER_SIZE)
			.collect();
		assert!(framed.layout.record_starts[0]);
		for (payload, &start) in framed.layout.record_starts.iter().enumerate().skip(1) {
			assert_eq!(start, offsets.contains(&(payload * PAYLOAD)), "payload {payload}");
		}
		// Small records dominate this frame, so nearly every payload is flagged.
		let flagged = framed.layout.record_starts.iter().filter(|&&s| s).count();
		assert!(flagged * 10 >= framed.layout.record_starts.len() * 9);
	}

	#[test]
	fn oversized_records_span_payloads() {
		let framed = frame(&bitstream(&[(0, 4000), (1, 3000), (2, 16)]), 1, PAYLOAD, usize::MAX).unwrap();
		let mut parsed = parsed_blocks(&framed);
		parsed.sort_unstable();
		assert_eq!(parsed, vec![0, 1, 2]);
	}

	#[test]
	fn transport_limit_preserves_every_record_and_correct_start_flags() {
		let blocks: Vec<_> = (0..100).map(|i| (i, 700 + (i as usize % 3) * 4)).collect();
		let stream = bitstream(&blocks);
		for extra in [0, 4, 8, 64, PAYLOAD, 10_000] {
			let framed = frame(&stream, 10, PAYLOAD, stream.len() + extra).unwrap();
			assert!(framed.data.len() <= stream.len() + extra);
			let parsed = records(&framed);
			let mut received: Vec<_> = parsed
				.iter()
				.filter_map(|(id, start, end)| id.map(|id| (id, framed.data[*start..*end].to_vec())))
				.collect();
			received.sort_unstable_by_key(|(id, _)| *id);
			for ((id, bytes), &(expected_id, len)) in received.iter().zip(&blocks) {
				assert_eq!(*id, expected_id);
				assert_eq!(*bytes, block(expected_id, len));
			}
			assert_eq!(received.len(), blocks.len());
			let starts: std::collections::HashSet<_> =
				parsed.iter().map(|(_, start, _)| FRAME_HEADER_SIZE + start).collect();
			for (payload, &flag) in framed.layout.record_starts.iter().enumerate().skip(1) {
				assert_eq!(flag, starts.contains(&(payload * PAYLOAD)));
			}
		}
		assert!(frame(&stream, 10, PAYLOAD, stream.len() - 1).is_err());
	}

	#[test]
	fn rejects_a_record_running_past_the_bitstream() {
		let mut stream = bitstream(&[(0, 64)]);
		stream.truncate(stream.len() - 4);
		assert!(frame(&stream, 1, PAYLOAD, usize::MAX).is_err());
	}
}
