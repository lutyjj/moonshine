//! PyroWave encoding: Hans-Kristian Arntzen's intra-only wavelet codec, run as
//! Vulkan compute on a private PyroWave device.
//!
//! Compositor frames are imported as DMA-BUFs and encoded through PyroWave's
//! scaled path, which converts RGB to full-range YCbCr on the GPU (BT.709 for
//! SDR, BT.2020 for HDR10, with PQ encoding for scRGB input). Every frame is a
//! key frame, so IDR and reference invalidation requests need no encoder work.

use std::collections::HashMap;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, IntoRawFd, OwnedFd, RawFd};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use ash::vk;
use async_shutdown::ShutdownManager;
use bytes::Bytes;
use pyrowave::sys as ffi;
use pyrowave::{
	ChromaSampling, Device, Encoder, ImportedImage, InputColorSpace, IntermediatePrecision, OutputColorSpace,
	ScaleConfig, VideoFormat, YcbcrRange,
};
use smithay::backend::allocator::{Buffer, dmabuf::Dmabuf};
use tokio::sync::{broadcast, mpsc, watch};

use super::dmabuf::{DmaBufPlane, ImportParams, same_buffer};
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
	/// `None` after a failed import, so the same buffer is not retried.
	image: Option<ImportedImage>,
	last_used: Instant,
}

/// Imports idle this long are released, so a reallocated swapchain's old
/// buffers do not stay pinned for the rest of the session.
const IMPORT_IDLE_EVICTION: Duration = Duration::from_secs(5);

pub(super) struct PyroWaveEncoder {
	encoder: Encoder,
	intermediate_precision: IntermediatePrecision,
	chroma_midpoint: f32,
	range: YcbcrRange,
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
		// Narrow range uses the profile's bit depth and its midpoint. Full range
		// keeps the midpoints these streams have always used.
		let (range, chroma_midpoint) = match (ctx.dynamic_range, ctx.full_range) {
			(VideoDynamicRange::Sdr, true) => (YcbcrRange::Full, 128.0 / 255.0),
			(VideoDynamicRange::Hdr, true) => (YcbcrRange::Full, 0.5),
			(VideoDynamicRange::Sdr, false) => (YcbcrRange::Narrow { bit_depth: 8 }, 128.0 / 255.0),
			(VideoDynamicRange::Hdr, false) => (YcbcrRange::Narrow { bit_depth: 10 }, 512.0 / 1023.0),
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
			full_range = ctx.full_range,
			"Created PyroWave encoder"
		);
		Ok(Self {
			encoder,
			intermediate_precision,
			chroma_midpoint,
			range,
			critical_blocks,
			imports: HashMap::new(),
		})
	}

	pub fn import(&mut self, dmabuf: &Dmabuf) -> Result<(), String> {
		// Modifiers with compression metadata (AMD DCC) add memory planes to the
		// packed RGB image. They are imported through the first plane's buffer.
		let planes: Vec<_> = dmabuf
			.handles()
			.zip(dmabuf.offsets())
			.zip(dmabuf.strides())
			.map(|((fd, offset), stride)| DmaBufPlane {
				fd: fd.as_raw_fd(),
				offset,
				stride,
				modifier: u64::from(dmabuf.format().modifier),
			})
			.collect();
		let plane = planes[0];
		let handle = dmabuf.handles().next().expect("a plane");
		if planes.len() > 4 || dmabuf.handles().any(|fd| !same_buffer(fd, handle)) {
			return Err("PyroWave requires all planes in one buffer".to_string());
		}
		let (_, vk_format) = drm_fourcc_to_input(dmabuf.format().code as u32);
		let params = ImportParams::new(dmabuf.width(), dmabuf.height(), vk_format, &planes);
		let now = Instant::now();
		self.imports
			.retain(|fd, import| *fd == plane.fd || now.duration_since(import.last_used) < IMPORT_IDLE_EVICTION);
		if let Some(import) = self.imports.get_mut(&plane.fd)
			&& import.params == params
			&& same_buffer(import.identity.as_fd(), handle)
		{
			import.last_used = now;
			return match import.image {
				Some(_) => Ok(()),
				None => Err("PyroWave could not import this buffer".to_string()),
			};
		}

		// A recycled fd or a changed layout.
		self.imports.remove(&plane.fd);

		if !matches!(
			vk_format,
			vk::Format::R8G8B8A8_UNORM
				| vk::Format::A2B10G10R10_UNORM_PACK32
				| vk::Format::R16G16B16A16_SFLOAT
				| vk::Format::B8G8R8A8_UNORM
		) {
			return Err("PyroWave cannot scale this buffer format".to_string());
		}
		// SAFETY: the exported buffer is retained until this frame is consumed.
		let borrowed = unsafe { BorrowedFd::borrow_raw(plane.fd) };
		let identity = borrowed.try_clone_to_owned().map_err(|e| e.to_string())?;
		let handed_over = borrowed.try_clone_to_owned().map_err(|e| e.to_string())?;
		let layouts: Vec<_> = planes
			.iter()
			.map(|p| ffi::VkSubresourceLayout {
				offset: p.offset.into(),
				rowPitch: p.stride.into(),
				..Default::default()
			})
			.collect();
		let modifier = ffi::VkImageDrmFormatModifierExplicitCreateInfoEXT {
			sType: ffi::VkStructureType_VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
			drmFormatModifier: plane.modifier,
			drmFormatModifierPlaneCount: layouts.len() as u32,
			pPlaneLayouts: layouts.as_ptr(),
			..Default::default()
		};
		let create_info = ffi::VkImageCreateInfo {
			sType: ffi::VkStructureType_VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			pNext: (&raw const modifier).cast(),
			imageType: ffi::VkImageType_VK_IMAGE_TYPE_2D,
			format: vk_format.as_raw() as ffi::VkFormat,
			extent: ffi::VkExtent3D {
				width: dmabuf.width(),
				height: dmabuf.height(),
				depth: 1,
			},
			mipLevels: 1,
			arrayLayers: 1,
			samples: ffi::VkSampleCountFlagBits_VK_SAMPLE_COUNT_1_BIT,
			tiling: ffi::VkImageTiling_VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
			usage: ffi::VkImageUsageFlagBits_VK_IMAGE_USAGE_SAMPLED_BIT,
			sharingMode: ffi::VkSharingMode_VK_SHARING_MODE_EXCLUSIVE,
			initialLayout: ffi::VkImageLayout_VK_IMAGE_LAYOUT_UNDEFINED,
			..Default::default()
		};
		let info = ffi::pyrowave_image_create_info {
			device: self.encoder.device().as_raw(),
			// Vulkan consumes the descriptor only if its memory import succeeds, and
			// a failure does not report whether that happened. Closing it afterwards
			// could close a recycled descriptor, so a failed import leaks it once and
			// is remembered below instead of being retried every frame.
			external_handle: handed_over.into_raw_fd() as ffi::pyrowave_os_handle,
			handle_type: ffi::VkExternalMemoryHandleTypeFlagBits_VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
			image_create_info: &create_info,
		};
		let mut handle = std::ptr::null_mut();
		// SAFETY: the compositor supplied the allocation's exact layout, and the
		// create infos outlive the call.
		let created = pyrowave::with_native(|| unsafe { ffi::pyrowave_image_create(&info, &mut handle) });
		let image = if created == ffi::pyrowave_result_PYROWAVE_SUCCESS {
			// SAFETY: the image was created on the encoder's device and has no other owner.
			Some(unsafe { ImportedImage::from_raw(self.encoder.device().clone(), handle) }.map_err(|e| e.to_string())?)
		} else {
			None
		};
		self.imports.insert(
			plane.fd,
			Import {
				params,
				identity,
				image,
				last_used: now,
			},
		);
		if self.imports[&plane.fd].image.is_none() {
			return Err(format!("PyroWave DMA-BUF import failed with {created}"));
		}
		tracing::debug!(fd = plane.fd, ?params, "Imported compositor buffer into PyroWave");
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
		let image = self.imports[&dmabuf.handles().next().expect("imported plane").as_raw_fd()]
			.image
			.as_ref()
			.expect("accepted import");
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
			chroma_midpoint: self.chroma_midpoint,
			range: self.range,
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
		.map_err(|error| error.to_string())
		.and_then(|mut encoder| encoder.import(dmabuf))
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
