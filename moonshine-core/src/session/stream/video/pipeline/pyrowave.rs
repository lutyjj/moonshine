//! PyroWave encoding: Hans-Kristian Arntzen's intra-only wavelet codec, run as
//! Vulkan compute on a private PyroWave device.
//!
//! Compositor frames are imported as DMA-BUFs and encoded through PyroWave's
//! scaled path, which converts RGB to full-range YCbCr on the GPU (BT.709 for
//! SDR, BT.2020 for HDR10, with PQ encoding for scRGB input). Every frame is a
//! key frame, so IDR and reference invalidation requests need no encoder work.

use std::collections::HashMap;
use std::os::fd::{AsRawFd, BorrowedFd, OwnedFd, RawFd};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use ash::vk;
use async_shutdown::ShutdownManager;
use bytes::Bytes;
use pyrowave::{
	ChromaSampling, Device, DmabufDescriptor, Encoder, ImportedImage, InputColorSpace, IntermediatePrecision,
	OutputColorSpace, RgbFormat, ScaleConfig, VideoFormat,
};
use tokio::sync::{broadcast, mpsc, watch};

use super::dmabuf::{DmaBufPlane, ImportParams, same_open_file};
use super::{
	ConsumerMessage, EncodedFrame, FrameContext, MAX_FRAMES_IN_FLIGHT, PacketConsumer, PendingFrame,
	VideoPipelineInner, drm_fourcc_to_input, forward_hdr_state,
};
use crate::session::compositor::frame::{ExportedFrame, FrameColorSpace, HdrModeState};
use crate::session::manager::SessionShutdownReason;
use crate::session::stream::video::packetizer::{NV_VIDEO_PACKET_SIZE, Packetizer};
use crate::session::stream::video::pyrowave_framing;
use crate::session::stream::video::shard_batch::ShardBatch;
use crate::session::stream::video::{FrameStats, VideoChromaSampling, VideoDynamicRange, VideoStreamContext};

struct Import {
	params: ImportParams,
	/// A dup of the exporter's fd, compared with `kcmp(2)` to catch fd recycling.
	identity: OwnedFd,
	image: ImportedImage,
	last_used: Instant,
}

/// Imports idle this long are released, so a reallocated swapchain's old
/// buffers do not stay pinned for the rest of the session.
const IMPORT_IDLE_EVICTION: Duration = Duration::from_secs(5);

pub(super) struct PyroWaveEncoder {
	encoder: Encoder,
	intermediate_precision: IntermediatePrecision,
	critical_blocks: usize,
	imports: HashMap<RawFd, Import>,
}

impl PyroWaveEncoder {
	pub fn new(ctx: &VideoStreamContext) -> Result<Self, pyrowave::Error> {
		let chroma = match ctx.chroma_sampling_type {
			VideoChromaSampling::Yuv420 => ChromaSampling::Yuv420,
			VideoChromaSampling::Yuv444 => ChromaSampling::Yuv444,
		};
		let intermediate_precision = match ctx.dynamic_range {
			VideoDynamicRange::Sdr => IntermediatePrecision::Unorm8,
			VideoDynamicRange::Hdr => IntermediatePrecision::Unorm16,
		};
		let device = Device::new()?;
		let encoder = Encoder::new(
			device,
			VideoFormat {
				width: ctx.width,
				height: ctx.height,
				chroma,
			},
		)?;
		let critical_blocks = encoder.active_blocks(2)?;
		tracing::info!(
			width = ctx.width,
			height = ctx.height,
			chroma = ?ctx.chroma_sampling_type,
			dynamic_range = ?ctx.dynamic_range,
			"Created PyroWave encoder"
		);
		Ok(Self {
			encoder,
			intermediate_precision,
			critical_blocks,
			imports: HashMap::new(),
		})
	}

	pub fn import(&mut self, frame: &ExportedFrame) -> Result<(), pyrowave::Error> {
		if frame.planes.len() != 1 {
			return Err(pyrowave::Error::InvalidInput("PyroWave requires one packed RGB plane"));
		}
		let plane = &frame.planes[0];
		let (_, vk_format) = drm_fourcc_to_input(frame.format);
		let params = ImportParams::new(
			frame.width,
			frame.height,
			vk_format,
			&[DmaBufPlane {
				fd: plane.fd,
				offset: plane.offset,
				stride: plane.stride,
				modifier: frame.modifier,
			}],
		);
		let now = Instant::now();
		self.imports
			.retain(|fd, import| *fd == plane.fd || now.duration_since(import.last_used) < IMPORT_IDLE_EVICTION);
		if let Some(import) = self.imports.get_mut(&plane.fd)
			&& import.params == params
			&& same_open_file(import.identity.as_raw_fd(), plane.fd)
		{
			import.last_used = now;
			return Ok(());
		}

		// A recycled fd or a changed layout.
		self.imports.remove(&plane.fd);

		// SAFETY: the exported buffer is retained until this frame is consumed.
		let borrowed = unsafe { BorrowedFd::borrow_raw(plane.fd) };
		let identity = borrowed.try_clone_to_owned()?;
		let format = match vk_format {
			vk::Format::R8G8B8A8_UNORM => RgbFormat::Rgba8,
			vk::Format::A2B10G10R10_UNORM_PACK32 => RgbFormat::Rgb10A2,
			vk::Format::R16G16B16A16_SFLOAT => RgbFormat::Rgba16Float,
			vk::Format::B8G8R8A8_UNORM => RgbFormat::Bgra8,
			_ => {
				return Err(pyrowave::Error::InvalidInput(
					"PyroWave cannot scale this buffer format",
				));
			},
		};
		let descriptor = DmabufDescriptor {
			width: frame.width,
			height: frame.height,
			format,
			modifier: frame.modifier,
			offset: plane.offset.into(),
			row_stride: plane.stride.into(),
		};
		// SAFETY: the compositor supplied the allocation's exact layout. It keeps
		// the buffer alive until the synchronous encode releases it.
		let image = unsafe { self.encoder.device().import_dmabuf(borrowed, descriptor) }?;
		tracing::debug!(fd = plane.fd, ?params, "Imported compositor buffer into PyroWave");
		self.imports.insert(
			plane.fd,
			Import {
				params,
				identity,
				image,
				last_used: now,
			},
		);
		Ok(())
	}

	/// Encode a frame that [`Self::import`] accepted.
	pub fn encode(&mut self, frame: &ExportedFrame, output_hdr: bool, budget: usize) -> Result<&[u8], pyrowave::Error> {
		let image = &self.imports[&frame.planes[0].fd].image;
		let scale = ScaleConfig {
			input_color_space: match frame.color_space {
				FrameColorSpace::Srgb => InputColorSpace::Srgb,
				FrameColorSpace::Bt2020Pq => InputColorSpace::Hdr10,
				FrameColorSpace::ScrgbLinear => InputColorSpace::ScrgbLinear,
			},
			output_color_space: if output_hdr {
				OutputColorSpace::Hdr10
			} else {
				OutputColorSpace::Srgb
			},
			intermediate_precision: self.intermediate_precision,
		};
		// SAFETY: the frame's storage is a DMA-BUF owned outside this device, and
		// ExportedFrame keeps it alive until `consumed`, set after this synchronous
		// encode.
		unsafe { self.encoder.encode_image(image, scale, budget) }
	}
}

pub(crate) fn probe(
	frame: &ExportedFrame,
	chroma: VideoChromaSampling,
	range: VideoDynamicRange,
) -> Result<(), String> {
	let ctx = VideoStreamContext {
		width: frame.width,
		height: frame.height,
		chroma_sampling_type: chroma,
		dynamic_range: range,
		..Default::default()
	};
	PyroWaveEncoder::new(&ctx)
		.and_then(|mut encoder| encoder.import(frame))
		.map_err(|error| error.to_string())
}

fn frame_budget(bitrate: usize, fps: u32, capacity: usize) -> usize {
	(bitrate / fps.max(1) as usize / 8).min(capacity)
}

impl VideoPipelineInner {
	#[allow(clippy::too_many_arguments)]
	pub(super) fn run_pyrowave_loop(
		&self,
		runtime: tokio::runtime::Handle,
		frame_rx: std::sync::mpsc::Receiver<ExportedFrame>,
		packet_tx: mpsc::Sender<ShardBatch>,
		idr_tx: broadcast::Sender<()>,
		mut idr_frame_request_rx: broadcast::Receiver<()>,
		mut reset_request_rx: broadcast::Receiver<()>,
		stop_session_manager: ShutdownManager<SessionShutdownReason>,
		hdr_metadata_tx: watch::Sender<HdrModeState>,
		stats_tx: broadcast::Sender<FrameStats>,
	) -> Result<(), String> {
		let ctx = &self.context;
		let frame_interval =
			Duration::try_from_secs_f64(1.0 / ctx.fps as f64).map_err(|_| "PyroWave frame rate must be nonzero")?;
		let mut encoder = PyroWaveEncoder::new(ctx).map_err(|error| error.to_string())?;

		let PacketConsumer {
			frame_ctx_tx,
			in_flight,
			task: consumer,
		} = self.spawn_packet_consumer(&runtime, packet_tx, stats_tx, idr_tx);

		let capacity = Packetizer::pyrowave_frame_capacity(ctx.packet_size, self.config.fec_percentage > 0)
			.ok_or_else(|| "Invalid PyroWave packet size".to_string())?;
		let critical_blocks = encoder.critical_blocks;
		let payload_size = ctx.packet_size - NV_VIDEO_PACKET_SIZE;
		let session_hdr = ctx.dynamic_range == VideoDynamicRange::Hdr;

		let budget = frame_budget(ctx.bitrate, ctx.fps, capacity);
		tracing::info!(
			bitrate_mbps = ctx.bitrate / 1_000_000,
			fps = ctx.fps,
			budget_bytes = budget,
			"PyroWave rate control"
		);
		let mut last_frame_time = Instant::now();
		let mut last_hdr_state = HdrModeState::new(session_hdr);
		let mut last_frame: Option<EncodedFrame> = None;
		let mut pts = 0u64;
		let mut last_drop_warn: Option<Instant> = None;
		// Set by a reset or IDR request until the next frame goes out.
		let mut resend = false;

		'encoding: while !stop_session_manager.is_shutdown_triggered() {
			// Every PyroWave frame is a key frame, so a reset or IDR request only
			// needs the current image resent; reference invalidation needs nothing.
			while let Ok(()) | Err(broadcast::error::TryRecvError::Lagged(_)) = reset_request_rx.try_recv() {
				tracing::debug!("Stream reset requested");
				resend = true;
				if frame_ctx_tx.blocking_send(ConsumerMessage::ResetCounters).is_err() {
					break 'encoding;
				}
			}
			while let Ok(()) | Err(broadcast::error::TryRecvError::Lagged(_)) = idr_frame_request_rx.try_recv() {
				resend = true;
			}

			let frame = match frame_rx.recv_timeout(frame_interval) {
				Ok(frame) => {
					last_frame_time = Instant::now();
					frame
				},
				Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
					if last_frame_time.elapsed() > Duration::from_secs(5) {
						tracing::warn!("No frames received for 5 seconds");
						last_frame_time = Instant::now();
					}
					// Answer recovery requests even when the captured image is static.
					if let Some(previous) = &last_frame
						&& resend && in_flight.load(Ordering::Relaxed) < MAX_FRAMES_IN_FLIGHT
					{
						let now = Instant::now();
						let context = FrameContext::immediate(now);
						in_flight.fetch_add(1, Ordering::Relaxed);
						pts += 1;
						let mut packet = previous.clone();
						packet.pts = pts;
						if frame_ctx_tx
							.blocking_send(ConsumerMessage::Frame(context, PendingFrame::Ready(packet)))
							.is_err()
						{
							break;
						}
						resend = false;
					}
					continue;
				},
				Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
					tracing::debug!("Frame channel disconnected - compositor stopped");
					break;
				},
			};

			if in_flight.load(Ordering::Relaxed) >= MAX_FRAMES_IN_FLIGHT {
				if last_drop_warn.is_none_or(|t| t.elapsed() >= Duration::from_secs(1)) {
					tracing::warn!(
						"Video encode backpressure: packet consumer is behind; dropping captured frames to stay realtime."
					);
					last_drop_warn = Some(Instant::now());
				}
				frame.consumed.store(true, Ordering::Release);
				continue;
			}

			let t_received = Instant::now();

			let output_hdr = session_hdr && frame.color_space != FrameColorSpace::Srgb;
			if let Err(error) = encoder.import(&frame) {
				tracing::warn!("Failed to import DMA-BUF: {error}");
				if let Some(failed) = &frame.import_failed {
					failed.store(true, Ordering::Relaxed);
				}
				frame.consumed.store(true, Ordering::Release);
				continue;
			}
			let t_imported = Instant::now();
			let encoded = encoder
				.encode(&frame, output_hdr, budget)
				.map_err(|error| error.to_string());
			// The encode waited for the GPU, so the compositor may reuse the buffer.
			frame.consumed.store(true, Ordering::Release);
			let t_encoded = Instant::now();

			let framed = pyrowave_framing::frame(encoded?, critical_blocks, payload_size, capacity)?;

			let t_ready = Instant::now();

			if session_hdr {
				let state = HdrModeState {
					enabled: output_hdr,
					metadata: frame.hdr_metadata,
				};
				forward_hdr_state(&mut last_hdr_state, state, &hdr_metadata_tx);
			}

			let context = FrameContext {
				created_at: frame.created_at,
				channel_wait: t_received.duration_since(frame.created_at),
				import: t_imported.duration_since(t_received),
				// Color conversion and encode are one synchronous GPU pass.
				convert: t_encoded.duration_since(t_imported),
				submit: t_ready.duration_since(t_encoded),
				submitted_at: t_ready,
				inject_hdr: false,
				hdr_metadata: None,
				buffer_index: frame.buffer_index,
			};
			in_flight.fetch_add(1, Ordering::Relaxed);
			pts += 1;
			let packet = EncodedFrame {
				data: Bytes::from(framed.data),
				pts,
				is_key_frame: true,
				record_layout: Some(Arc::new(framed.layout)),
			};
			last_frame = Some(packet.clone());
			if frame_ctx_tx
				.blocking_send(ConsumerMessage::Frame(context, PendingFrame::Ready(packet)))
				.is_err()
			{
				tracing::debug!("Packet consumer gone; stopping encoding loop.");
				break;
			}
			resend = false;
		}

		drop(frame_ctx_tx);
		if let Err(e) = runtime.block_on(consumer) {
			tracing::warn!("Packet consumer task panicked: {e:?}");
		}
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn budget_follows_bitrate_and_frame_rate() {
		assert_eq!(frame_budget(480_000_000, 120, 2_000_000), 500_000);
	}

	#[test]
	fn budget_is_capped_by_the_packet_limit() {
		assert_eq!(frame_budget(10_000_000_000, 60, 2_000_000), 2_000_000);
	}
}
