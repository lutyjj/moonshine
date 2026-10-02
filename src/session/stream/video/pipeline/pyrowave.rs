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
use smithay::backend::allocator::{Buffer, dmabuf::Dmabuf};
use tokio::sync::{broadcast, mpsc, watch};

use super::dmabuf::{DmaBufPlane, ImportParams, same_open_file};
use super::{
	ConsumerMessage, EncodedFrame, FrameContext, MAX_FRAMES_IN_FLIGHT, PacketConsumer, PendingFrame,
	VideoPipelineInner, drm_fourcc_to_input, fence, forward_hdr_state,
};
use crate::session::compositor::CompositorFrames;
use crate::session::compositor::frame::{FrameColorSpace, HdrModeState, color_space, hdr_metadata};
use crate::session::manager::SessionShutdownReason;
use crate::session::stream::video::PacketBatch;
use crate::session::stream::video::packetizer::{NV_VIDEO_PACKET_SIZE, Packetizer};
use crate::session::stream::video::pyrowave_framing;
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

	pub fn import(&mut self, dmabuf: &Dmabuf) -> Result<(), pyrowave::Error> {
		if dmabuf.num_planes() != 1 {
			return Err(pyrowave::Error::InvalidInput("PyroWave requires one packed RGB plane"));
		}
		let plane = DmaBufPlane {
			fd: dmabuf.handles().next().expect("one plane").as_raw_fd(),
			offset: dmabuf.offsets().next().expect("one plane"),
			stride: dmabuf.strides().next().expect("one plane"),
			modifier: u64::from(dmabuf.format().modifier),
		};
		let (_, vk_format) = drm_fourcc_to_input(dmabuf.format().code as u32);
		let params = ImportParams::new(
			dmabuf.width(),
			dmabuf.height(),
			vk_format,
			&[DmaBufPlane {
				fd: plane.fd,
				offset: plane.offset,
				stride: plane.stride,
				modifier: u64::from(dmabuf.format().modifier),
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
			width: dmabuf.width(),
			height: dmabuf.height(),
			format,
			modifier: u64::from(dmabuf.format().modifier),
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
	pub fn encode(
		&mut self,
		dmabuf: &Dmabuf,
		input: FrameColorSpace,
		output_hdr: bool,
		budget: usize,
	) -> Result<&[u8], pyrowave::Error> {
		let image = &self.imports[&dmabuf.handles().next().expect("imported plane").as_raw_fd()].image;
		let scale = ScaleConfig {
			input_color_space: match input {
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
		// the caller retains it until this synchronous encode completes.
		unsafe { self.encoder.encode_image(image, scale, budget) }
	}
}

pub(crate) fn probe(dmabuf: &Dmabuf, chroma: VideoChromaSampling, range: VideoDynamicRange) -> Result<(), String> {
	let ctx = VideoStreamContext {
		width: dmabuf.width(),
		height: dmabuf.height(),
		chroma_sampling_type: chroma,
		dynamic_range: range,
		..Default::default()
	};
	PyroWaveEncoder::new(&ctx)
		.and_then(|mut encoder| encoder.import(dmabuf))
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
		frames: CompositorFrames,
		packet_tx: mpsc::Sender<PacketBatch>,
		idr_tx: broadcast::Sender<()>,
		mut idr_frame_request_rx: broadcast::Receiver<()>,
		mut reset_request_rx: broadcast::Receiver<u32>,
		stop_session_manager: ShutdownManager<SessionShutdownReason>,
		hdr_metadata_tx: watch::Sender<HdrModeState>,
		stats_tx: broadcast::Sender<FrameStats>,
	) -> Result<(), String> {
		let ctx = &self.context;
		let Some(frames) = frames.get() else {
			return Err("The compositor was not launched".to_string());
		};
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
		let mut last_drop_warn: Option<Instant> = None;
		// Set by a reset or IDR request until the next frame goes out.
		let mut resend = false;

		'encoding: while !stop_session_manager.is_shutdown_triggered() {
			// Every PyroWave frame is a key frame, so a reset or IDR request only
			// needs the current image resent; reference invalidation needs nothing.
			let mut pending_reset = None;
			loop {
				match reset_request_rx.try_recv() {
					Ok(epoch) => pending_reset = Some(epoch),
					Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
					Err(broadcast::error::TryRecvError::Closed | broadcast::error::TryRecvError::Empty) => break,
				}
			}
			if let Some(epoch) = pending_reset {
				tracing::debug!("Stream reset requested");
				resend = true;
				if frame_ctx_tx
					.blocking_send(ConsumerMessage::ResetCounters(epoch))
					.is_err()
				{
					break 'encoding;
				}
			}
			while let Ok(()) | Err(broadcast::error::TryRecvError::Lagged(_)) = idr_frame_request_rx.try_recv() {
				resend = true;
			}

			let frame = match frames.recv_timeout(frame_interval) {
				Ok(frame) => {
					last_frame_time = Instant::now();
					frame
				},
				Err(ferriscope::RecvTimeoutError::Timeout) => {
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
						let packet = previous.clone();
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
				Err(ferriscope::RecvTimeoutError::Disconnected) => {
					tracing::debug!("Frame channel disconnected - compositor stopped");
					let _ = stop_session_manager.trigger_shutdown(SessionShutdownReason::CompositorStopped);
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
				continue;
			}

			let t_received = Instant::now();

			let ferriscope::FrameData::DmaBuf(buffer) = &frame.data;
			if let Some(acquire) = &buffer.acquire_fence
				&& let Err(error) = fence::wait_sync_file(acquire.as_raw_fd(), fence::FENCE_TIMEOUT)
			{
				tracing::warn!("Dropping frame: {error}");
				continue;
			}
			let input = color_space(&frame.info.colour);
			let output_hdr = session_hdr && input != FrameColorSpace::Srgb;
			if let Err(error) = encoder.import(&buffer.dmabuf) {
				tracing::warn!("Failed to import DMA-BUF: {error}");
				continue;
			}
			let t_imported = Instant::now();
			let encoded = encoder
				.encode(&buffer.dmabuf, input, output_hdr, budget)
				.map_err(|error| error.to_string());
			// Encoding is synchronous. Return the pool slot only after the GPU read.
			let captured_at = frame.info.captured_at;
			let sequence = frame.info.sequence;
			let metadata = hdr_metadata(&frame.info.colour);
			drop(frame);
			let t_encoded = Instant::now();

			let framed = pyrowave_framing::frame(encoded?, critical_blocks, payload_size, capacity)?;

			let t_ready = Instant::now();

			if session_hdr {
				let state = HdrModeState {
					enabled: output_hdr,
					metadata,
				};
				forward_hdr_state(&mut last_hdr_state, state, &hdr_metadata_tx);
			}

			let context = FrameContext {
				created_at: captured_at,
				sequence: Some(sequence),
				channel_wait: t_received.duration_since(captured_at),
				import: t_imported.duration_since(t_received),
				// Color conversion and encode are one synchronous GPU pass.
				convert: t_encoded.duration_since(t_imported),
				submit: t_ready.duration_since(t_encoded),
				submitted_at: t_ready,
				inject_hdr: false,
				hdr_metadata: None,
			};
			in_flight.fetch_add(1, Ordering::Relaxed);
			let packet = EncodedFrame {
				data: Bytes::from(framed.data),
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
