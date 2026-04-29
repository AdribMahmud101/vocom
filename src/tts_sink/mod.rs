use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64};

use crate::duplex_audio::{DuplexPlaybackGate, RenderReferencePublisher};
use crate::errors::VocomError;

mod cpal_sink;
#[cfg(target_os = "android")]
mod android_sink;

pub use cpal_sink::CpalTtsSink;
#[cfg(target_os = "android")]
pub use android_sink::AndroidTtsSink;

pub struct TtsPlaybackRequest<'a> {
    pub samples: &'a [f32],
    pub sample_rate: i32,
    pub render_reference: Option<&'a RenderReferencePublisher>,
    pub playback_gate: Option<&'a DuplexPlaybackGate>,
}

pub struct TtsPlaybackControl {
    pub stop_requested: Arc<AtomicBool>,
    pub interruptible_after_ms: Arc<AtomicU64>,
    pub playback_target_gain_bits: Arc<AtomicU32>,
    pub playback_current_gain_bits: Arc<AtomicU32>,
    pub interrupt_grace_ms: u64,
}

pub trait TtsSink: Send + Sync {
    fn play(
        &self,
        request: TtsPlaybackRequest<'_>,
        control: TtsPlaybackControl,
    ) -> Result<(), VocomError>;
}

pub fn default_tts_sink() -> Arc<dyn TtsSink> {
    #[cfg(target_os = "android")]
    {
        Arc::new(AndroidTtsSink)
    }

    #[cfg(not(target_os = "android"))]
    {
            Arc::new(CpalTtsSink {})
    }
}
