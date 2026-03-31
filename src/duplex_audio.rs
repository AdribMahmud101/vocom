use crossbeam_channel::{bounded, Receiver, Sender};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub struct RenderReferencePublisher {
    tx: Sender<Vec<f32>>,
    sample_rate: i32,
}

pub struct RenderReferenceConsumer {
    #[cfg_attr(not(feature = "webrtc-aec"), allow(dead_code))]
    rx: Receiver<Vec<f32>>,
    #[cfg_attr(not(feature = "webrtc-aec"), allow(dead_code))]
    sample_rate: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BargeInMetricsSnapshot {
    pub requested: u64,
    pub rejected_low_rms: u64,
    pub rejected_render_ratio: u64,
    pub rejected_persistence: u64,
    pub rejected_confidence: u64,
    pub ducked: u64,
    pub stopped: u64,
}

#[derive(Clone)]
pub struct DuplexPlaybackGate {
    tts_active: Arc<AtomicBool>,
    last_tts_end_ms: Arc<AtomicU64>,
    barge_in_requested: Arc<AtomicBool>,
    last_barge_in_ms: Arc<AtomicU64>,
    last_render_rms_bits: Arc<AtomicU32>,
    last_render_rms_ms: Arc<AtomicU64>,
    barge_in_requested_count: Arc<AtomicU64>,
    barge_in_rejected_low_rms_count: Arc<AtomicU64>,
    barge_in_rejected_render_ratio_count: Arc<AtomicU64>,
    barge_in_rejected_persistence_count: Arc<AtomicU64>,
    barge_in_rejected_confidence_count: Arc<AtomicU64>,
    barge_in_ducked_count: Arc<AtomicU64>,
    barge_in_stopped_count: Arc<AtomicU64>,
}

impl RenderReferencePublisher {
    pub fn publish(&self, samples: Vec<f32>) {
        // Prefer bounded waiting over immediate drop to preserve AEC reference continuity.
        let _ = self.tx.send_timeout(samples, Duration::from_millis(2));
    }

    pub fn sample_rate(&self) -> i32 {
        self.sample_rate
    }
}

impl RenderReferenceConsumer {
    #[cfg_attr(not(feature = "webrtc-aec"), allow(dead_code))]
    pub fn sample_rate(&self) -> i32 {
        self.sample_rate
    }

    #[cfg_attr(not(feature = "webrtc-aec"), allow(dead_code))]
    pub fn try_recv(&self) -> Option<Vec<f32>> {
        self.rx.try_recv().ok()
    }
}

impl DuplexPlaybackGate {
    pub fn new() -> Self {
        Self {
            tts_active: Arc::new(AtomicBool::new(false)),
            last_tts_end_ms: Arc::new(AtomicU64::new(0)),
            barge_in_requested: Arc::new(AtomicBool::new(false)),
            last_barge_in_ms: Arc::new(AtomicU64::new(0)),
            last_render_rms_bits: Arc::new(AtomicU32::new(0.0f32.to_bits())),
            last_render_rms_ms: Arc::new(AtomicU64::new(0)),
            barge_in_requested_count: Arc::new(AtomicU64::new(0)),
            barge_in_rejected_low_rms_count: Arc::new(AtomicU64::new(0)),
            barge_in_rejected_render_ratio_count: Arc::new(AtomicU64::new(0)),
            barge_in_rejected_persistence_count: Arc::new(AtomicU64::new(0)),
            barge_in_rejected_confidence_count: Arc::new(AtomicU64::new(0)),
            barge_in_ducked_count: Arc::new(AtomicU64::new(0)),
            barge_in_stopped_count: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn mark_tts_start(&self) {
        self.tts_active.store(true, Ordering::Release);
    }

    pub fn mark_tts_end(&self) {
        self.tts_active.store(false, Ordering::Release);
        self.last_tts_end_ms.store(now_ms(), Ordering::Release);
    }

    pub fn should_suppress_asr(&self, cooldown_ms: u64) -> bool {
        if self.tts_active.load(Ordering::Acquire) {
            return true;
        }

        let ended_at = self.last_tts_end_ms.load(Ordering::Acquire);
        ended_at != 0 && now_ms().saturating_sub(ended_at) <= cooldown_ms
    }

    pub fn is_tts_active(&self) -> bool {
        self.tts_active.load(Ordering::Acquire)
    }

    pub fn mark_render_frame_rms(&self, rms: f32) {
        self.last_render_rms_bits
            .store(rms.max(0.0).to_bits(), Ordering::Release);
        self.last_render_rms_ms.store(now_ms(), Ordering::Release);
    }

    pub fn render_rms_recent(&self, max_age_ms: u64) -> Option<f32> {
        let updated_at = self.last_render_rms_ms.load(Ordering::Acquire);
        if updated_at == 0 || now_ms().saturating_sub(updated_at) > max_age_ms {
            return None;
        }
        Some(f32::from_bits(
            self.last_render_rms_bits.load(Ordering::Acquire),
        ))
    }

    pub fn request_barge_in_if_active(
        &self,
        rms: f32,
        threshold: f32,
        min_interval_ms: u64,
    ) -> bool {
        if !self.tts_active.load(Ordering::Acquire) || rms < threshold {
            return false;
        }

        let now = now_ms();
        let last = self.last_barge_in_ms.load(Ordering::Acquire);
        if last != 0 && now.saturating_sub(last) < min_interval_ms {
            return false;
        }

        self.last_barge_in_ms.store(now, Ordering::Release);
        self.barge_in_requested.store(true, Ordering::Release);
        self.barge_in_requested_count.fetch_add(1, Ordering::AcqRel);
        true
    }

    pub fn take_barge_in_request_with_timestamp(&self) -> Option<u64> {
        if self.barge_in_requested.swap(false, Ordering::AcqRel) {
            Some(self.last_barge_in_ms.load(Ordering::Acquire))
        } else {
            None
        }
    }

    pub fn note_rejected_low_rms(&self) {
        self.barge_in_rejected_low_rms_count
            .fetch_add(1, Ordering::AcqRel);
    }

    pub fn note_rejected_render_ratio(&self) {
        self.barge_in_rejected_render_ratio_count
            .fetch_add(1, Ordering::AcqRel);
    }

    pub fn note_rejected_persistence(&self) {
        self.barge_in_rejected_persistence_count
            .fetch_add(1, Ordering::AcqRel);
    }

    pub fn note_rejected_confidence(&self) {
        self.barge_in_rejected_confidence_count
            .fetch_add(1, Ordering::AcqRel);
    }

    pub fn note_ducked(&self) {
        self.barge_in_ducked_count.fetch_add(1, Ordering::AcqRel);
    }

    pub fn note_stopped(&self) {
        self.barge_in_stopped_count.fetch_add(1, Ordering::AcqRel);
    }

    pub fn barge_in_metrics_snapshot(&self) -> BargeInMetricsSnapshot {
        BargeInMetricsSnapshot {
            requested: self.barge_in_requested_count.load(Ordering::Acquire),
            rejected_low_rms: self.barge_in_rejected_low_rms_count.load(Ordering::Acquire),
            rejected_render_ratio: self
                .barge_in_rejected_render_ratio_count
                .load(Ordering::Acquire),
            rejected_persistence: self
                .barge_in_rejected_persistence_count
                .load(Ordering::Acquire),
            rejected_confidence: self
                .barge_in_rejected_confidence_count
                .load(Ordering::Acquire),
            ducked: self.barge_in_ducked_count.load(Ordering::Acquire),
            stopped: self.barge_in_stopped_count.load(Ordering::Acquire),
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn render_reference_bus(
    capacity: usize,
    sample_rate: i32,
) -> (RenderReferencePublisher, RenderReferenceConsumer) {
    let (tx, rx) = bounded(capacity);
    (
        RenderReferencePublisher { tx, sample_rate },
        RenderReferenceConsumer { rx, sample_rate },
    )
}
