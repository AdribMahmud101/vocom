use std::collections::VecDeque;

use aec3::voip::VoipAec3;
use sherpa_onnx::LinearResampler;

use crate::config::{AecBackend, AecConfig};
use crate::duplex_audio::RenderReferenceConsumer;
use crate::errors::VocomError;

const FRAME_MS: usize = 10;

pub struct AecProcessor {
    inner: AecInner,
}

enum AecInner {
    Disabled,
    PureRust(PureRustAec),
}

struct PureRustAec {
    voip: VoipAec3,
    capture_to_aec: Option<LinearResampler>,
    aec_to_capture: Option<LinearResampler>,
    render_to_aec: Option<LinearResampler>,
    pending_capture: Vec<f32>,
    pending_render: VecDeque<f32>,
    frame_len: usize,
    consumer: RenderReferenceConsumer,
}

impl AecProcessor {
    pub fn new(
        cfg: &AecConfig,
        capture_sample_rate: i32,
        consumer: RenderReferenceConsumer,
    ) -> Result<Self, VocomError> {
        if !cfg.enabled {
            return Ok(Self {
                inner: AecInner::Disabled,
            });
        }

        match cfg.backend {
            AecBackend::PureRustAec3 => {
                Ok(Self {
                    inner: AecInner::PureRust(build_pure_rust_aec(cfg, capture_sample_rate, consumer)?),
                })
            }
        }
    }

    pub fn process_capture_chunk(&mut self, input: Vec<f32>) -> Result<Vec<f32>, VocomError> {
        match &mut self.inner {
            AecInner::Disabled => Ok(input),
            AecInner::PureRust(aec) => aec.process_chunk(input),
        }
    }
}

fn build_pure_rust_aec(
    cfg: &AecConfig,
    capture_sample_rate: i32,
    consumer: RenderReferenceConsumer,
) -> Result<PureRustAec, VocomError> {
    let aec_sample_rate = cfg.sample_rate;
    let voip = VoipAec3::builder(aec_sample_rate as usize, 1, 1)
        .enable_high_pass(true)
        .enable_noise_suppression(false)
        .initial_delay_ms(cfg.stream_delay_ms.unwrap_or(120))
        .build()
        .map_err(|e| VocomError::AecConfig(format!("failed to create pure-rust AEC3: {e}")))?;

    let capture_to_aec = if capture_sample_rate != aec_sample_rate {
        Some(
            LinearResampler::create(capture_sample_rate, aec_sample_rate).ok_or_else(|| {
                VocomError::AecConfig(format!(
                    "failed to create capture->aec resampler: {capture_sample_rate} -> {aec_sample_rate}"
                ))
            })?,
        )
    } else {
        None
    };

    let aec_to_capture = if capture_sample_rate != aec_sample_rate {
        Some(
            LinearResampler::create(aec_sample_rate, capture_sample_rate).ok_or_else(|| {
                VocomError::AecConfig(format!(
                    "failed to create aec->capture resampler: {aec_sample_rate} -> {capture_sample_rate}"
                ))
            })?,
        )
    } else {
        None
    };

    let render_to_aec = if consumer.sample_rate() != aec_sample_rate {
        Some(
            LinearResampler::create(consumer.sample_rate(), aec_sample_rate).ok_or_else(|| {
                VocomError::AecConfig(format!(
                    "failed to create render->aec resampler: {} -> {aec_sample_rate}",
                    consumer.sample_rate()
                ))
            })?,
        )
    } else {
        None
    };

    let frame_len = (aec_sample_rate as usize * FRAME_MS) / 1000;
    Ok(PureRustAec {
        voip,
        capture_to_aec,
        aec_to_capture,
        render_to_aec,
        pending_capture: Vec::with_capacity(frame_len * 4),
        pending_render: VecDeque::with_capacity(frame_len * 8),
        frame_len,
        consumer,
    })
}


impl PureRustAec {
    fn process_chunk(&mut self, input: Vec<f32>) -> Result<Vec<f32>, VocomError> {
        self.drain_render_queue();

        let capture = if let Some(r) = &self.capture_to_aec {
            r.resample(&input, false)
        } else {
            input
        };

        self.pending_capture.extend(capture);

        let mut processed = Vec::new();
        while self.pending_capture.len() >= self.frame_len {
            let capture_frame: Vec<f32> = self.pending_capture.drain(..self.frame_len).collect();
            let render_frame = self.take_render_frame();
            let mut out = vec![0.0_f32; self.frame_len];

            self.voip
                .process(&capture_frame, Some(&render_frame), false, &mut out)
                .map_err(|e| VocomError::AecProcessing(format!("pure-rust AEC processing failed: {e}")))?;

            processed.extend_from_slice(&out);
        }

        if processed.is_empty() {
            return Ok(Vec::new());
        }

        Ok(if let Some(r) = &self.aec_to_capture {
            r.resample(&processed, false)
        } else {
            processed
        })
    }

    fn drain_render_queue(&mut self) {
        while let Some(mut frame) = self.consumer.try_recv() {
            if let Some(r) = &self.render_to_aec {
                frame = r.resample(&frame, false);
            }
            self.pending_render.extend(frame);
        }
    }

    fn take_render_frame(&mut self) -> Vec<f32> {
        if self.pending_render.len() >= self.frame_len {
            self.pending_render.drain(..self.frame_len).collect()
        } else {
            vec![0.0; self.frame_len]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AecProcessor;
    use crate::config::{AecBackend, AecConfig};
    use crate::duplex_audio::render_reference_bus;

    #[test]
    fn pure_rust_aec_processes_capture_chunk() {
        let (render_pub, render_consumer) = render_reference_bus(16, 16_000);

        // Seed a small render reference so AEC has far-end audio context.
        render_pub.publish(vec![0.1; 320]);

        let cfg = AecConfig {
            enabled: true,
            backend: AecBackend::PureRustAec3,
            sample_rate: 16_000,
            stream_delay_ms: Some(120),
        };

        let mut processor =
            AecProcessor::new(&cfg, 16_000, render_consumer).expect("failed to create AEC");

        let capture_chunk = vec![0.05; 320];
        let output = processor
            .process_capture_chunk(capture_chunk)
            .expect("pure-rust AEC processing failed");

        assert!(!output.is_empty(), "expected processed audio output");
    }
}

