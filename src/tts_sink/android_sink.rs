use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

use jni::objects::{JByteArray, JObject, JValue};
use jni::JavaVM;
use sherpa_onnx::LinearResampler;

use crate::duplex_audio::RenderReferencePublisher;
use crate::errors::VocomError;

use super::{TtsPlaybackControl, TtsPlaybackRequest, TtsSink};

const RENDER_FRAME_MS: usize = 10;
const GAIN_RAMP_MS: usize = 80;
const TTS_DRAIN_PADDING_MS: u64 = 140;

#[derive(Default)]
pub struct AndroidTtsSink;

impl TtsSink for AndroidTtsSink {
    fn play(
        &self,
        request: TtsPlaybackRequest<'_>,
        control: TtsPlaybackControl,
    ) -> Result<(), VocomError> {
        play_with_audio_track(request, control)
    }
}

fn play_with_audio_track(
    request: TtsPlaybackRequest<'_>,
    control: TtsPlaybackControl,
) -> Result<(), VocomError> {
    let playback_started = Instant::now();

    if let Some(gate) = request.playback_gate {
        gate.mark_tts_start();
    }

    struct PlaybackEndGuard<'a> {
        gate: Option<&'a crate::duplex_audio::DuplexPlaybackGate>,
    }

    impl Drop for PlaybackEndGuard<'_> {
        fn drop(&mut self) {
            if let Some(gate) = self.gate {
                gate.mark_tts_end();
            }
        }
    }

    let _playback_guard = PlaybackEndGuard {
        gate: request.playback_gate,
    };

    let target_sample_rate = 48000i32;
    let channels = 1usize;

    let mono = if request.sample_rate != target_sample_rate {
        let resampler = LinearResampler::create(request.sample_rate, target_sample_rate)
            .ok_or_else(|| {
                VocomError::TtsGeneration(format!(
                    "failed to create android tts resampler: {} -> {}",
                    request.sample_rate, target_sample_rate
                ))
            })?;
        resampler.resample(request.samples, false)
    } else {
        request.samples.to_vec()
    };

    if mono.is_empty() {
        return Ok(());
    }

    if let Some(publisher) = request.render_reference {
        publish_render_reference(&mono, target_sample_rate, publisher)?;
    }

    control.interruptible_after_ms.store(
        now_ms().saturating_add(control.interrupt_grace_ms),
        Ordering::Release,
    );

    let peak = mono.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let boost = if peak > 0.0 && peak < 0.90 {
        (0.92 / peak).clamp(1.0, 16.0)
    } else {
        1.0
    };

    let ramp_samples = (((target_sample_rate as usize) * GAIN_RAMP_MS) / 1000)
        .max(1)
        .saturating_mul(channels.max(1));
    let gain_step = 1.0f32 / ramp_samples as f32;

    let ctx = ndk_context::android_context();
    let vm = unsafe { JavaVM::from_raw(ctx.vm().cast()) }
        .map_err(|e| VocomError::Stream(format!("failed to resolve Android JavaVM: {e}")))?;
    let mut env = vm
        .attach_current_thread()
        .map_err(|e| VocomError::Stream(format!("failed to attach JNI thread: {e}")))?;

    let min_buffer_size = env
        .call_static_method(
            "android/media/AudioTrack",
            "getMinBufferSize",
            "(III)I",
            &[
                JValue::Int(target_sample_rate),
                JValue::Int(4), // CHANNEL_OUT_MONO
                JValue::Int(2), // ENCODING_PCM_16BIT
            ],
        )
        .and_then(|v| v.i())
        .map_err(|e| VocomError::Stream(format!("AudioTrack.getMinBufferSize failed: {e}")))?;

    if min_buffer_size <= 0 {
        return Err(VocomError::Stream(format!(
            "invalid AudioTrack buffer size: {min_buffer_size}"
        )));
    }

    let buffer_size = (min_buffer_size as usize).max(target_sample_rate as usize / 10 * 2) as i32;

    let audio_track = env
        .new_object(
            "android/media/AudioTrack",
            "(IIIIII)V",
            &[
                JValue::Int(3), // STREAM_MUSIC
                JValue::Int(target_sample_rate),
                JValue::Int(4), // CHANNEL_OUT_MONO
                JValue::Int(2), // ENCODING_PCM_16BIT
                JValue::Int(buffer_size),
                JValue::Int(1), // MODE_STREAM
            ],
        )
        .map_err(|e| VocomError::Stream(format!("AudioTrack construction failed: {e}")))?;

    env.call_method(&audio_track, "play", "()V", &[])
        .map_err(|e| VocomError::Stream(format!("AudioTrack.play failed: {e}")))?;

    let chunk_frames = (target_sample_rate as usize / 50).max(256); // ~20ms
    let mut total_frames_written: u64 = 0;
    let mut current_gain = f32::from_bits(control.playback_current_gain_bits.load(Ordering::Acquire));

    for chunk in mono.chunks(chunk_frames) {
        if control.stop_requested.load(Ordering::Acquire) {
            break;
        }

        let target_gain = f32::from_bits(control.playback_target_gain_bits.load(Ordering::Acquire));
        let mut sum_sq = 0.0f32;
        let mut pcm = Vec::with_capacity(chunk.len() * 2);

        for &s in chunk {
            let delta = (target_gain - current_gain).clamp(-gain_step, gain_step);
            current_gain = (current_gain + delta).clamp(0.0, 1.0);
            let out = (s * boost * current_gain).clamp(-1.0, 1.0);
            sum_sq += out * out;
            let sample = (out * i16::MAX as f32) as i16;
            pcm.extend_from_slice(&sample.to_le_bytes());
        }

        control
            .playback_current_gain_bits
            .store(current_gain.to_bits(), Ordering::Release);

        if let Some(gate) = request.playback_gate {
            if !chunk.is_empty() {
                let rms = (sum_sq / chunk.len() as f32).sqrt();
                gate.mark_render_frame_rms(rms);
            }
        }

        let arr = JByteArray::from(
            env.byte_array_from_slice(&pcm)
                .map_err(|e| VocomError::Stream(format!("JNI byte array failed: {e}")))?,
        );

        let mut offset = 0usize;
        while offset < pcm.len() {
            let remaining = (pcm.len() - offset) as i32;
            let written = env
                .call_method(
                    &audio_track,
                    "write",
                    "([BII)I",
                    &[
                        JValue::Object(arr.as_ref()),
                        JValue::Int(offset as i32),
                        JValue::Int(remaining),
                    ],
                )
                .and_then(|v| v.i())
                .map_err(|e| VocomError::Stream(format!("AudioTrack.write failed: {e}")))?;

            if written < 0 {
                return Err(VocomError::Stream(format!(
                    "AudioTrack.write returned error code: {written}"
                )));
            }
            if written == 0 {
                return Err(VocomError::Stream(
                    "AudioTrack.write returned 0 bytes; refusing to spin".to_string(),
                ));
            }

            offset = offset.saturating_add(written as usize);
        }

        // PREVENT JNI LEAK: Free the local reference to the chunk array!
        // JNI has a limit of 512 local refs. Without this, long TTS sentences
        // will crash or stutter severely due to Garbage Collection pressure.
        let _ = env.delete_local_ref(JObject::from(arr));

        total_frames_written = total_frames_written.saturating_add(chunk.len() as u64);
    }

    if !control.stop_requested.load(Ordering::Acquire) {
        // Wait for hardware playout to catch up before tearing down the stream.
        let expected_ms = ((total_frames_written * 1000) / (target_sample_rate as u64))
            .saturating_add(TTS_DRAIN_PADDING_MS);
        let max_wait_ms = expected_ms.saturating_add(1000);
        let wait_started = Instant::now();

        loop {
            let played_frames = env
                .call_method(&audio_track, "getPlaybackHeadPosition", "()I", &[])
                .and_then(|v| v.i())
                .map(|v| (v as u32) as u64)
                .unwrap_or(0);

            if played_frames >= total_frames_written {
                break;
            }

            if (wait_started.elapsed().as_millis() as u64) >= max_wait_ms {
                break;
            }

            thread::sleep(Duration::from_millis(10));
        }
    }

    env.call_method(&audio_track, "stop", "()V", &[])
        .map_err(|e| VocomError::Stream(format!("AudioTrack.stop failed: {e}")))?;
    let _ = env.call_method(&audio_track, "flush", "()V", &[]);
    let _ = env.call_method(&audio_track, "release", "()V", &[]);

    Ok(())
}

fn publish_render_reference(
    samples: &[f32],
    sample_rate: i32,
    publisher: &RenderReferencePublisher,
) -> Result<(), VocomError> {
    let render_sr = publisher.sample_rate();
    let mono = if render_sr != sample_rate {
        let resampler = LinearResampler::create(sample_rate, render_sr).ok_or_else(|| {
            VocomError::TtsGeneration(format!(
                "failed to create render resampler: {sample_rate} -> {render_sr}"
            ))
        })?;
        resampler.resample(samples, false)
    } else {
        samples.to_vec()
    };

    let frame_len = ((render_sr as usize) * RENDER_FRAME_MS) / 1000;
    if frame_len == 0 {
        return Ok(());
    }

    for frame in mono.chunks(frame_len) {
        publisher.publish(frame.to_vec());
    }

    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
