use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;

use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Sender, bounded};
use sherpa_onnx::{
    GenerationConfig, LinearResampler, OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig,
    OfflineTtsVitsModelConfig,
};

use crate::config::TtsConfig;
use crate::duplex_audio::{DuplexPlaybackGate, RenderReferencePublisher};
use crate::errors::VocomError;

const RENDER_FRAME_MS: usize = 10;
const GAIN_RAMP_MS: usize = 80;

enum TtsCommand {
    Speak(String),
    Shutdown,
}

pub struct TtsManager {
    command_tx: Sender<TtsCommand>,
    stop_requested: Arc<AtomicBool>,
    playback_target_gain_bits: Arc<AtomicU32>,
    worker: Option<JoinHandle<()>>,
}

impl TtsManager {
    pub fn from_config(
        config: &TtsConfig,
        render_reference: Option<RenderReferencePublisher>,
        playback_gate: Option<DuplexPlaybackGate>,
    ) -> Result<Self, VocomError> {
        assert_path_exists(&config.model_path)?;
        assert_path_exists(&config.tokens_path)?;
        assert_path_exists(&config.data_dir)?;
        if let Some(dict_dir) = &config.dict_dir {
            assert_path_exists(dict_dir)?;
        }

        let tts_config = OfflineTtsConfig {
            model: OfflineTtsModelConfig {
                vits: OfflineTtsVitsModelConfig {
                    model: Some(config.model_path.clone()),
                    tokens: Some(config.tokens_path.clone()),
                    data_dir: Some(config.data_dir.clone()),
                    dict_dir: config.dict_dir.clone(),
                    ..Default::default()
                },
                num_threads: config.num_threads,
                provider: Some(config.provider.clone()),
                ..Default::default()
            },
            ..Default::default()
        };

        let tts = OfflineTts::create(&tts_config).ok_or_else(|| {
            VocomError::TtsConfig("failed to create offline tts from configuration".to_string())
        })?;

        let generation_config = GenerationConfig {
            speed: config.speed,
            sid: config.speaker_id,
            ..Default::default()
        };

        let (command_tx, command_rx) = bounded::<TtsCommand>(64);
        let stop_requested = Arc::new(AtomicBool::new(false));
        let stop_requested_worker = Arc::clone(&stop_requested);
        let playback_target_gain_bits = Arc::new(AtomicU32::new(1.0f32.to_bits()));
        let playback_target_gain_bits_worker = Arc::clone(&playback_target_gain_bits);
        let playback_current_gain_bits = Arc::new(AtomicU32::new(1.0f32.to_bits()));
        let playback_current_gain_bits_worker = Arc::clone(&playback_current_gain_bits);
        let worker = thread::spawn(move || {
            while let Ok(cmd) = command_rx.recv() {
                match cmd {
                    TtsCommand::Speak(text) => {
                        stop_requested_worker.store(false, Ordering::Release);
                        playback_target_gain_bits_worker
                            .store(1.0f32.to_bits(), Ordering::Release);
                        playback_current_gain_bits_worker
                            .store(0.0f32.to_bits(), Ordering::Release);
                        let result = tts.generate_with_config::<fn(&[f32], f32) -> bool>(
                            &text,
                            &generation_config,
                            None,
                        );

                        if let Some(audio) = result {
                            if let Err(err) = play_generated_audio(
                                audio.samples(),
                                audio.sample_rate(),
                                render_reference.as_ref(),
                                playback_gate.as_ref(),
                                Arc::clone(&stop_requested_worker),
                                Arc::clone(&playback_target_gain_bits_worker),
                                Arc::clone(&playback_current_gain_bits_worker),
                            ) {
                                eprintln!("tts playback failed: {err}");
                            }
                        } else {
                            eprintln!("tts generation failed: no audio returned");
                        }
                    }
                    TtsCommand::Shutdown => break,
                }
            }
        });

        Ok(Self {
            command_tx,
            stop_requested,
            playback_target_gain_bits,
            worker: Some(worker),
        })
    }

    pub fn mock_response(input: &str) -> String {
        format!("I heard you say: {input}. This is a local sherpa generated response.")
    }

    pub fn speak_text(&self, text: &str) -> Result<(), VocomError> {
        self.stop_requested.store(false, Ordering::Release);
        self.command_tx
            .send(TtsCommand::Speak(text.to_string()))
            .map_err(|_| VocomError::ChannelDisconnected)
    }

    pub fn stop_current(&self) {
        self.stop_requested.store(true, Ordering::Release);
    }

    pub fn duck_current(&self, level: f32) {
        let gain = level.clamp(0.0, 1.0);
        self.playback_target_gain_bits
            .store(gain.to_bits(), Ordering::Release);
    }

    pub fn restore_volume(&self) {
        self.playback_target_gain_bits
            .store(1.0f32.to_bits(), Ordering::Release);
    }
}

impl Drop for TtsManager {
    fn drop(&mut self) {
        let _ = self.command_tx.send(TtsCommand::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn assert_path_exists(path: &str) -> Result<(), VocomError> {
    if Path::new(path).exists() {
        Ok(())
    } else {
        Err(VocomError::MissingModelPath(path.to_string()))
    }
}

fn play_generated_audio(
    samples: &[f32],
    sample_rate: i32,
    render_reference: Option<&RenderReferencePublisher>,
    playback_gate: Option<&DuplexPlaybackGate>,
    stop_requested: Arc<AtomicBool>,
    playback_target_gain_bits: Arc<AtomicU32>,
    playback_current_gain_bits: Arc<AtomicU32>,
) -> Result<(), VocomError> {
    if let Some(gate) = playback_gate {
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

    let _playback_guard = PlaybackEndGuard { gate: playback_gate };

    // Render reference is published here for echo-cancellation context.
    // Render RMS is NOT marked here — it is marked in the real output callback
    // at the moment samples are actually pushed to hardware (see write_slice).
    if let Some(publisher) = render_reference {
        publish_render_reference(samples, sample_rate, publisher)?;
    }

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| VocomError::Stream("output audio device not found".to_string()))?;

    let output_config = device
        .default_output_config()
        .map_err(VocomError::DefaultInputConfig)?;
    let output_sample_rate = output_config.sample_rate() as i32;
    let channels = output_config.channels() as usize;
    let ramp_samples = (((output_sample_rate as usize) * GAIN_RAMP_MS) / 1000)
        .max(1)
        .saturating_mul(channels.max(1));
    let gain_step = 1.0f32 / ramp_samples as f32;

    let mono = if output_sample_rate != sample_rate {
        let resampler = LinearResampler::create(sample_rate, output_sample_rate).ok_or_else(|| {
            VocomError::TtsGeneration(format!(
                "failed to create tts resampler: {sample_rate} -> {output_sample_rate}"
            ))
        })?;
        resampler.resample(samples, false)
    } else {
        samples.to_vec()
    };

    if mono.is_empty() {
        return Ok(());
    }

    let mut interleaved = Vec::with_capacity(mono.len() * channels);
    for sample in mono {
        for _ in 0..channels {
            interleaved.push(sample);
        }
    }

    let data = Arc::new(interleaved);
    let position = Arc::new(AtomicUsize::new(0));
    let completed = Arc::new(AtomicBool::new(false));

    let err_fn = |err: cpal::StreamError| {
        eprintln!("tts output stream error: {err}");
    };

    let stream_config: cpal::StreamConfig = output_config.clone().into();

    let stream = match output_config.sample_format() {
        SampleFormat::F32 => {
            let data = Arc::clone(&data);
            let position = Arc::clone(&position);
            let completed = Arc::clone(&completed);
            let stop_requested = Arc::clone(&stop_requested);
            let playback_target_gain_bits = Arc::clone(&playback_target_gain_bits);
            let playback_current_gain_bits = Arc::clone(&playback_current_gain_bits);
            let gate_f32 = playback_gate.cloned();

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
            let stop_requested = Arc::clone(&stop_requested);
            let playback_target_gain_bits = Arc::clone(&playback_target_gain_bits);
            let playback_current_gain_bits = Arc::clone(&playback_current_gain_bits);
            let gate_i16 = playback_gate.cloned();

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
            let stop_requested = Arc::clone(&stop_requested);
            let playback_target_gain_bits = Arc::clone(&playback_target_gain_bits);
            let playback_current_gain_bits = Arc::clone(&playback_current_gain_bits);
            let gate_u16 = playback_gate.cloned();

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
                "unsupported output sample format: {other:?}"
            )))
        }
    };

    stream.play()?;

    while !completed.load(Ordering::Acquire) {
        if stop_requested.load(Ordering::Acquire) {
            completed.store(true, Ordering::Release);
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    thread::sleep(Duration::from_millis(40));
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

        // Mark real-time render RMS so the capture path sees an accurate
        // far-end signal age when evaluating barge-in confidence.
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


