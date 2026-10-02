use std::time::Duration;

use tokio::time::Instant;

/// How far behind schedule a send may catch up. Timers wake up to a
/// millisecond late, and without this the average rate would fall short.
const MAX_CATCH_UP: Duration = Duration::from_millis(2);

/// Spreads a frame's packets over time, so a large frame does not leave as one
/// burst that overruns a slower link's queue or the client's receive buffer.
pub(crate) struct Pacer {
	bytes_per_second: f64,
	/// Earliest time the next chunk may leave.
	next: Instant,
}

impl Pacer {
	/// Zero disables pacing. Headers and parity count toward the configured rate.
	pub fn new(configured_mbps: u32) -> Self {
		Self {
			bytes_per_second: configured_mbps as f64 * 1_000_000.0 / 8.0,
			next: Instant::now(),
		}
	}

	/// Start a frame. Idle time before it earns no credit.
	pub fn start_frame(&mut self) {
		self.next = self.next.max(Instant::now());
	}

	/// Wait until `bytes` more may leave.
	pub async fn pace(&mut self, bytes: usize) {
		let now = Instant::now();
		let start = self.schedule(now, bytes);
		if start > now {
			tokio::time::sleep_until(start).await;
		}
	}

	/// Reserve the send slot for `bytes` and return when it starts.
	fn schedule(&mut self, now: Instant, bytes: usize) -> Instant {
		if self.bytes_per_second == 0.0 {
			return now;
		}
		let start = self.next.max(now.checked_sub(MAX_CATCH_UP).unwrap_or(now));
		self.next = start + Duration::from_secs_f64(bytes as f64 / self.bytes_per_second);
		start
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn zero_disables_pacing() {
		let mut pacer = Pacer::new(0);
		let now = pacer.next;
		for _ in 0..10 {
			assert_eq!(pacer.schedule(now, 10_000_000), now);
		}
		assert_eq!(Pacer::new(100).bytes_per_second, 12_500_000.0);
	}

	#[test]
	fn a_frame_is_spread_at_the_pacing_rate() {
		// 800 Mbps is 100 kB per millisecond.
		let mut pacer = Pacer::new(800);
		let t0 = pacer.next;
		let starts: Vec<Duration> = (0..4).map(|_| pacer.schedule(t0, 100_000) - t0).collect();
		assert_eq!(starts, [0, 1, 2, 3].map(Duration::from_millis));
	}

	#[test]
	fn idle_time_before_a_frame_earns_no_credit() {
		let mut pacer = Pacer::new(800);
		pacer.next = Instant::now() - Duration::from_millis(5);
		let before = Instant::now();
		pacer.start_frame();
		assert!((before..=Instant::now()).contains(&pacer.next));
		let reset_at = pacer.next;
		assert_eq!(pacer.schedule(reset_at, 100_000), reset_at);
		assert_eq!(pacer.schedule(reset_at, 100_000), reset_at + Duration::from_millis(1));
	}

	#[test]
	fn a_late_wake_up_catches_up_within_bounds() {
		let mut pacer = Pacer::new(800);
		let t0 = pacer.next;
		assert_eq!(pacer.schedule(t0, 100_000), t0);
		// Woken a millisecond late: the next chunk keeps the original schedule.
		let late = t0 + Duration::from_millis(2);
		assert_eq!(pacer.schedule(late, 100_000), t0 + Duration::from_millis(1));
		// Far behind: only the bounded amount is caught up.
		let stalled = t0 + Duration::from_millis(50);
		assert_eq!(pacer.schedule(stalled, 100_000), stalled - MAX_CATCH_UP);
	}
}
