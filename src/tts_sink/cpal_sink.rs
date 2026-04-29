use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

use cpal::Device;
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use sherpa_onnx::LinearResampler;

use crate::duplex_audio::{DuplexPlaybackGate, RenderReferencePublisher};
use crate::errors::VocomError;

use super::{TtsPlaybackControl, TtsPlaybackRequest, TtsSink};

const RENDER_FRAME_MS: usize = 10;
const GAIN_RAMP_MS: usize = 80;
const TTS_CHIRP_MS: usize = 0;
const TTS_DRAIN_PADDING_MS: u64 = 60;

#[derive(Default)]
pub struct CpalTtsSink {}

impl TtsSink for CpalTtsSink {
    fn play(
        &self,
        request: TtsPlaybackRequest<'_>,
        control: TtsPlaybackControl,
    ) -> Result<(), VocomError> {
        play_generated_audio(request, control)
    }
}

fn play_generated_audio(
    request: TtsPlaybackRequest<'_>,
    control: TtsPlaybackControl,
) -> Result<(), VocomError> {
    if let Some(gate) = request.playback_gate {
        gate.mark_tts_start();
    }

    struct PlaybackEndGuard<'a> {
        gate: Option<&'a DuplexPlaybackGate>,
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

    if let Some(publisher) = request.render_reference {
        publish_render_reference(request.samples, request.sample_rate, publisher)?;
    }

    let host = cpal::default_host();
    let candidates = output_device_candidates(&host)?;
    let mut last_err: Option<VocomError> = None;

    for (idx, (device, label)) in candidates.into_iter().enumerate() {
        match play_on_device(&device, &label, &request, &control) {
            Ok(()) => {
                if idx > 0 {
                    eprintln!("tts playback recovered on fallback output device: {label}");
                }
                return Ok(());
            }
            Err(err) => {
                eprintln!("tts playback failed on device '{label}': {err}");
                last_err = Some(err);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| {
        VocomError::Stream("tts playback failed: no usable output device".to_string())
    }))
}

fn output_device_candidates(host: &cpal::Host) -> Result<Vec<(Device, String)>, VocomError> {
    let preferred_hint = std::env::var("VOCOM_TTS_OUTPUT_HINT")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();

    let default_name = host
        .default_output_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_default();

    let mut devices: Vec<(i32, Device, String)> = Vec::new();
    let all_devices = host
        .output_devices()
        .map_err(|e| VocomError::Stream(format!("failed to enumerate output devices: {e}")))?;

    for device in all_devices {
        let name = device
            .name()
            .unwrap_or_else(|_| "unknown-output-device".to_string());
        let lower = name.to_ascii_lowercase();
        let mut score = 0i32;
        if !preferred_hint.is_empty() && lower.contains(&preferred_hint) {
            score += 1000;
        }
        if !default_name.is_empty() && name == default_name {
            score += 800;
        }
        if lower.contains("bluetooth")
            || lower.contains("airpod")
            || lower.contains("earbud")
            || lower.contains("headset")
            || lower.contains("buds")
        {
            score += 400;
        }
        devices.push((score, device, name));
    }

    if devices.is_empty() {
        return Err(VocomError::Stream("output audio device not found".to_string()));
    }

    devices.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.2.cmp(&b.2)));
    Ok(devices
        .into_iter()
        .map(|(_, device, name)| (device, name))
        .collect())
}

fn play_on_device(
    device: &Device,
    device_label: &str,
    request: &TtsPlaybackRequest<'_>,
    control: &TtsPlaybackControl,
) -> Result<(), VocomError> {
    let output_config = device
        .default_output_config()
        .map_err(VocomError::DefaultInputConfig)?;
    let output_sample_rate = output_config.sample_rate() as i32;
    let channels = output_config.channels() as usize;
    let ramp_samples = (((output_sample_rate as usize) * GAIN_RAMP_MS) / 1000)
        .max(1)
        .saturating_mul(channels.max(1));
    let gain_step = 1.0f32 / ramp_samples as f32;

    let mono = if output_sample_rate != request.sample_rate {
        let resampler = LinearResampler::create(request.sample_rate, output_sample_rate)
            .ok_or_else(|| {
                VocomError::TtsGeneration(format!(
                    "failed to create tts resampler: {} -> {}",
                    request.sample_rate, output_sample_rate
                ))
            })?;
        resampler.resample(request.samples, false)
    } else {
        request.samples.to_vec()
    };

    if mono.is_empty() {
        return Ok(());
    }

    control.interruptible_after_ms.store(
        now_ms().saturating_add(control.interrupt_grace_ms),
        Ordering::Release,
    );

    let peak = mono.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let gain = if peak > 0.0 && peak < 0.90 {
        (0.92 / peak).clamp(1.0, 16.0)
    } else {
        1.0
    };
    if gain > 1.01 {
        eprintln!("tts playback gain boost applied: peak={peak:.5} gain={gain:.2}");
    }

    let chirp = build_chirp(output_sample_rate, channels);
    let mut interleaved = Vec::with_capacity(chirp.len() + mono.len() * channels);
    interleaved.extend_from_slice(&chirp);
    for sample in mono {
        let boosted = (sample * gain).clamp(-1.0, 1.0);
        for _ in 0..channels {
            interleaved.push(boosted);
        }
    }

    let data = Arc::new(interleaved);
    let position = Arc::new(AtomicUsize::new(0));
    let completed = Arc::new(AtomicBool::new(false));

    let err_fn = {
        let label = device_label.to_string();
        move |err: cpal::StreamError| {
            eprintln!("tts output stream error [{label}]: {err}");
        }
    };

    let stream_config: cpal::StreamConfig = output_config.clone().into();
    let stream = match output_config.sample_format() {
        SampleFormat::F32 => {
            let data = Arc::clone(&data);
            let position = Arc::clone(&position);
            let completed = Arc::clone(&completed);
            let stop_requested = Arc::clone(&control.stop_requested);
            let playback_target_gain_bits = Arc::clone(&control.playback_target_gain_bits);
            let playback_current_gain_bits = Arc::clone(&control.playback_current_gain_bits);
            let gate_f32 = request.playback_gate.cloned();
            device.build_output_stream(
                &stream_config,
                move |output: &mut [f32], _| {
                    fill_output_buffer_f32(
                        output,
                        &data,
                        &position,
                        &completed,
                        stop_requested.as_ref(),
                        playback_target_gain_bits.as_ref(),
                        playback_current_gain_bits.as_ref(),
                        gain_step,
                        gate_f32.as_ref(),
                    )
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let data = Arc::clone(&data);
            let position = Arc::clone(&position);
            let completed = Arc::clone(&completed);
            let stop_requested = Arc::clone(&control.stop_requested);
            let playback_target_gain_bits = Arc::clone(&control.playback_target_gain_bits);
            let playback_current_gain_bits = Arc::clone(&control.playback_current_gain_bits);
            let gate_i16 = request.playback_gate.cloned();
            device.build_output_stream(
                &stream_config,
                move |output: &mut [i16], _| {
                    fill_output_buffer_i16(
                        output,
                        &data,
                        &position,
                        &completed,
                        stop_requested.as_ref(),
                        playback_target_gain_bits.as_ref(),
                        playback_current_gain_bits.as_ref(),
                        gain_step,
                        gate_i16.as_ref(),
                    )
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let data = Arc::clone(&data);
            let position = Arc::clone(&position);
            let completed = Arc::clone(&completed);
            let stop_requested = Arc::clone(&control.stop_requested);
            let playback_target_gain_bits = Arc::clone(&control.playback_target_gain_bits);
            let playback_current_gain_bits = Arc::clone(&control.playback_current_gain_bits);
            let gate_u16 = request.playback_gate.cloned();
            device.build_output_stream(
                &stream_config,
                move |output: &mut [u16], _| {
                    fill_output_buffer_u16(
                        output,
                        &data,
                        &position,
                        &completed,
                        stop_requested.as_ref(),
                        playback_target_gain_bits.as_ref(),
                        playback_current_gain_bits.as_ref(),
                        gain_step,
                        gate_u16.as_ref(),
                    )
                },
                err_fn,
                None,
            )?
        }
        other => {
            return Err(VocomError::Stream(format!(
                "unsupported output sample format on '{device_label}': {other:?}"
            )));
        }
    };

    let expected_playback_ms = ((data.len() as u64) * 1000)
        .saturating_div((output_sample_rate.max(1) as u64).saturating_mul(channels.max(1) as u64));
    let playback_started_at = Instant::now();
    stream.play()?;

    while !completed.load(Ordering::Acquire) {
        if control.stop_requested.load(Ordering::Acquire) {
            completed.store(true, Ordering::Release);
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    if !control.stop_requested.load(Ordering::Acquire) {
        let min_lifetime_ms = expected_playback_ms.saturating_add(TTS_DRAIN_PADDING_MS);
        let elapsed_ms = playback_started_at.elapsed().as_millis() as u64;
        if elapsed_ms < min_lifetime_ms {
            thread::sleep(Duration::from_millis(min_lifetime_ms - elapsed_ms));
        }
    }
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn build_chirp(sample_rate: i32, channels: usize) -> Vec<f32> {
    let frames = ((sample_rate as usize) * TTS_CHIRP_MS) / 1000;
    if frames == 0 || channels == 0 {
        return Vec::new();
    }

    let fade_frames = ((sample_rate as usize) * 20) / 1000;
    let freq_hz = 880.0f32;
    let amp = 0.70f32;
    let mut out = Vec::with_capacity(frames * channels);

    for n in 0..frames {
        let t = n as f32 / sample_rate as f32;
        let mut env = 1.0f32;
        if fade_frames > 0 {
            if n < fade_frames {
                env = n as f32 / fade_frames as f32;
            } else if n >= frames.saturating_sub(fade_frames) {
                env = (frames.saturating_sub(n)) as f32 / fade_frames as f32;
            }
        }

        let s = (2.0f32 * std::f32::consts::PI * freq_hz * t).sin() * amp * env.clamp(0.0, 1.0);
        for _ in 0..channels {
            out.push(s);
        }
    }

    out
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

fn fill_output_buffer_f32(
    output: &mut [f32],
    data: &[f32],
    position: &AtomicUsize,
    completed: &AtomicBool,
    stop_requested: &AtomicBool,
    playback_target_gain_bits: &AtomicU32,
    playback_current_gain_bits: &AtomicU32,
    gain_step: f32,
    gate: Option<&DuplexPlaybackGate>,
) {
    let start = position.fetch_add(output.len(), Ordering::AcqRel);
    write_slice(
        output,
        data,
        start,
        |dst, src| *dst = src,
        completed,
        stop_requested,
        playback_target_gain_bits,
        playback_current_gain_bits,
        gain_step,
        gate,
    );
}

fn fill_output_buffer_i16(
    output: &mut [i16],
    data: &[f32],
    position: &AtomicUsize,
    completed: &AtomicBool,
    stop_requested: &AtomicBool,
    playback_target_gain_bits: &AtomicU32,
    playback_current_gain_bits: &AtomicU32,
    gain_step: f32,
    gate: Option<&DuplexPlaybackGate>,
) {
    let start = position.fetch_add(output.len(), Ordering::AcqRel);
    write_slice(
        output,
        data,
        start,
        |dst, src| *dst = (src.clamp(-1.0, 1.0) * i16::MAX as f32) as i16,
        completed,
        stop_requested,
        playback_target_gain_bits,
        playback_current_gain_bits,
        gain_step,
        gate,
    );
}

fn fill_output_buffer_u16(
    output: &mut [u16],
    data: &[f32],
    position: &AtomicUsize,
    completed: &AtomicBool,
    stop_requested: &AtomicBool,
    playback_target_gain_bits: &AtomicU32,
    playback_current_gain_bits: &AtomicU32,
    gain_step: f32,
    gate: Option<&DuplexPlaybackGate>,
) {
    let start = position.fetch_add(output.len(), Ordering::AcqRel);
    write_slice(
        output,
        data,
        start,
        |dst, src| {
            let normalized = ((src.clamp(-1.0, 1.0) + 1.0) * 0.5 * u16::MAX as f32) as u16;
            *dst = normalized;
        },
        completed,
        stop_requested,
        playback_target_gain_bits,
        playback_current_gain_bits,
        gain_step,
        gate,
    );
}

fn write_slice<T>(
    output: &mut [T],
    data: &[f32],
    start: usize,
    mut convert: impl FnMut(&mut T, f32),
    completed: &AtomicBool,
    stop_requested: &AtomicBool,
    playback_target_gain_bits: &AtomicU32,
    playback_current_gain_bits: &AtomicU32,
    gain_step: f32,
    gate: Option<&DuplexPlaybackGate>,
) where
    T: Default,
{
    if stop_requested.load(Ordering::Acquire) {
        for v in output.iter_mut() {
            *v = T::default();
        }
        completed.store(true, Ordering::Release);
        return;
    }

    if start >= data.len() {
        for v in output.iter_mut() {
            *v = T::default();
        }
        completed.store(true, Ordering::Release);
    } else {
        let available = (data.len() - start).min(output.len());
        let target_gain = f32::from_bits(playback_target_gain_bits.load(Ordering::Acquire));
        let mut current_gain = f32::from_bits(playback_current_gain_bits.load(Ordering::Acquire));
        let mut sum_sq = 0.0f32;
        for i in 0..available {
            let delta = (target_gain - current_gain).clamp(-gain_step, gain_step);
            current_gain = (current_gain + delta).clamp(0.0, 1.0);
            let s = data[start + i] * current_gain;
            convert(&mut output[i], s);
            sum_sq += s * s;
        }
        playback_current_gain_bits.store(current_gain.to_bits(), Ordering::Release);

        if available > 0 {
            if let Some(g) = gate {
                let rms = (sum_sq / available as f32).sqrt();
                g.mark_render_frame_rms(rms);
            }
        }

        if available < output.len() {
            for v in output.iter_mut().skip(available) {
                *v = T::default();
            }
            completed.store(true, Ordering::Release);
        }
    }
}
