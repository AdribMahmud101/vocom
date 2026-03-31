use std::collections::VecDeque;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender, TrySendError};
use sherpa_onnx::{LinearResampler, OfflineRecognizer, OnlineRecognizer, OnlineStream, VoiceActivityDetector};

use crate::aec_manager::AecProcessor;
use crate::asr_manager::ASRModelBuilder;
use crate::config::AecConfig;
use crate::duplex_audio::{DuplexPlaybackGate, RenderReferenceConsumer};
use crate::errors::VocomError;

const HISTORY_SECONDS: usize = 30;



pub enum AsrBackend {
    Offline(OfflineRecognizer),
    Online(OnlineRecognizer),
}

enum AsrRuntime {
    Offline(OfflineRecognizer),
    Online {
        recognizer: OnlineRecognizer,
        stream: OnlineStream,
        utterance_samples: usize,
    },
}

#[derive(Clone, Debug)]
pub struct RealtimeConfig {
    pub target_sample_rate: i32,
    pub chunk_ms: u32,
    pub audio_queue_capacity: usize,
    pub event_queue_capacity: usize,
    pub pre_speech_buffer_ms: u32,
    pub tts_suppression_cooldown_ms: u64,
    pub tts_barge_in_rms_threshold: f32,
    pub tts_barge_in_render_ratio_min: f32,
    pub tts_barge_in_render_rms_max_age_ms: u64,
    pub tts_barge_in_persistence_ms: u64,
    pub tts_barge_in_confidence_threshold: f32,
    pub tts_barge_in_confidence_smoothing: f32,
    pub tts_barge_in_suspect_hold_ms: u64,
    pub tts_barge_in_recover_ms: u64,
    /// Minimum interval between successive barge-in requests emitted to the FSM (ms).
    pub barge_in_min_interval_ms: u64,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 16_000,
            chunk_ms: 20,
            audio_queue_capacity: 64,
            event_queue_capacity: 64,
            pre_speech_buffer_ms: 300,
            tts_suppression_cooldown_ms: 700,
            tts_barge_in_rms_threshold: 0.02,
            tts_barge_in_render_ratio_min: 1.35,
            tts_barge_in_render_rms_max_age_ms: 200,
            tts_barge_in_persistence_ms: 60,
            tts_barge_in_confidence_threshold: 0.68,
            tts_barge_in_confidence_smoothing: 0.3,
            tts_barge_in_suspect_hold_ms: 90,
            tts_barge_in_recover_ms: 350,
            barge_in_min_interval_ms: 150,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TranscriptEvent {
    pub text: String,
    pub sample_rate: i32,
    pub sample_count: usize,
    pub decode_latency_ms: u128,
}

pub struct RealtimeTranscriber {
    stream: Stream,
    audio_rx: Receiver<Vec<f32>>,
    asr: AsrRuntime,
    vad: VoiceActivityDetector,
    aec: AecProcessor,
    duplex_gate: DuplexPlaybackGate,
    resampler: Option<LinearResampler>,
    target_sample_rate: i32,
    pre_speech_buffer_samples: usize,
    history_samples: VecDeque<f32>,
    history_start_index: usize,
    total_samples_seen: usize,
    history_capacity_samples: usize,
    tts_suppression_cooldown_ms: u64,
    tts_barge_in_rms_threshold: f32,
    tts_barge_in_render_ratio_min: f32,
    tts_barge_in_render_rms_max_age_ms: u64,
    tts_barge_in_persistence_ms: u64,
    barge_in_confident_ms: u64,
    barge_in_confidence: f32,
    tts_barge_in_confidence_threshold: f32,
    tts_barge_in_confidence_smoothing: f32,
    barge_in_min_interval_ms: u64,
    online_replay_buffer: VecDeque<f32>,
    online_replay_capacity_samples: usize,
    pending_events: VecDeque<Result<TranscriptEvent, VocomError>>,
}

impl RealtimeTranscriber {
    pub fn recv(&mut self) -> Result<TranscriptEvent, VocomError> {
        loop {
            if let Some(event) = self.pending_events.pop_front() {
                return event;
            }

            let chunk = self
                .audio_rx
                .recv()
                .map_err(|_| VocomError::ChannelDisconnected)?;

            self.process_chunk(chunk);
        }
    }

    pub fn recv_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<TranscriptEvent>, VocomError> {
        if let Some(event) = self.pending_events.pop_front() {
            return event.map(Some);
        }

        match self.audio_rx.recv_timeout(timeout) {
            Ok(chunk) => {
                self.process_chunk(chunk);
                if let Some(event) = self.pending_events.pop_front() {
                    event.map(Some)
                } else {
                    Ok(None)
                }
            }
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(VocomError::ChannelDisconnected),
        }
    }

    pub fn keep_alive(&self) {
        let _ = &self.stream;
    }

    fn process_chunk(&mut self, chunk: Vec<f32>) {
        let audio = if let Some(ref r) = self.resampler {
            r.resample(&chunk, false)
        } else {
            chunk
        };

        if audio.is_empty() {
            return;
        }

        let aec_audio = match self.aec.process_capture_chunk(audio) {
            Ok(v) => v,
            Err(err) => {
                self.pending_events.push_back(Err(err));
                return;
            }
        };

        if aec_audio.is_empty() {
            return;
        }

        self.append_history(&aec_audio);

        let chunk_duration_ms = ((aec_audio.len() as u64) * 1000)
            .saturating_div(self.target_sample_rate.max(1) as u64)
            .max(1);

        if self
            .duplex_gate
            .should_suppress_asr(self.tts_suppression_cooldown_ms)
        {
            let rms = compute_rms(&aec_audio);
            if rms < self.tts_barge_in_rms_threshold {
                self.duplex_gate.note_rejected_low_rms();
                self.barge_in_confident_ms = 0;
                self.barge_in_confidence = 0.0;
                return;
            }

            self.vad.accept_waveform(&aec_audio);
            if !self.vad.detected() {
                // Not speech according to VAD.
                self.barge_in_confident_ms = 0;
                self.barge_in_confidence = 0.0;
                return;
            }

            let render_rms = self
                .duplex_gate
                .render_rms_recent(self.tts_barge_in_render_rms_max_age_ms);
            if let Some(r) = render_rms {
                let min_near_end_rms = r * self.tts_barge_in_render_ratio_min;
                if rms < min_near_end_rms {
                    self.duplex_gate.note_rejected_render_ratio();
                    self.barge_in_confident_ms = 0;
                    self.barge_in_confidence = 0.0;
                    return;
                }
            }

            self.barge_in_confident_ms = self
                .barge_in_confident_ms
                .saturating_add(chunk_duration_ms)
                .min(self.tts_barge_in_persistence_ms);

            if self.barge_in_confident_ms < self.tts_barge_in_persistence_ms {
                self.duplex_gate.note_rejected_persistence();
            }

            let confidence = compute_barge_in_confidence(
                rms,
                render_rms,
                self.tts_barge_in_rms_threshold,
                self.tts_barge_in_render_ratio_min,
                self.barge_in_confident_ms,
                self.tts_barge_in_persistence_ms,
            );

            self.barge_in_confidence = smooth_confidence(
                self.barge_in_confidence,
                confidence,
                self.tts_barge_in_confidence_smoothing,
            );

            if self.barge_in_confidence < self.tts_barge_in_confidence_threshold {
                self.buffer_online_replay_if_enabled(&aec_audio);
                self.duplex_gate.note_rejected_confidence();
                return;
            }

            let _ = self.duplex_gate.request_barge_in_if_active(
                rms,
                self.tts_barge_in_rms_threshold,
                self.barge_in_min_interval_ms,
            );
        } else {
            self.barge_in_confident_ms = 0;
            self.barge_in_confidence = 0.0;
            self.vad.accept_waveform(&aec_audio);
        }

        match self.asr {
            AsrRuntime::Offline(_) => {

                while let Some(segment) = self.vad.front() {
                    let samples = segment.samples().to_vec();
                    let segment_start = segment.start().max(0) as usize;
                    self.vad.pop();

                    if samples.is_empty() {
                        continue;
                    }

                    let pre_start = segment_start.saturating_sub(self.pre_speech_buffer_samples);
                    let mut samples_with_pre_roll = self.history_range(pre_start, segment_start);
                    samples_with_pre_roll.extend_from_slice(&samples);

                    let started = Instant::now();
                    let result = self
                        .decode_segment(&samples_with_pre_roll)
                    .map(|text| TranscriptEvent {
                        text,
                        sample_rate: self.target_sample_rate,
                        sample_count: samples_with_pre_roll.len(),
                        decode_latency_ms: started.elapsed().as_millis(),
                    });

                    self.pending_events.push_back(result);
                }
            }
            AsrRuntime::Online { .. } => {
                let mut combined = self.take_online_replay();
                combined.extend_from_slice(&aec_audio);
                if let Some(event) = self.decode_online_continuous(&combined) {
                    self.pending_events.push_back(event);
                }
            }
        }
    }

    fn buffer_online_replay_if_enabled(&mut self, samples: &[f32]) {
        if !matches!(self.asr, AsrRuntime::Online { .. }) || samples.is_empty() {
            return;
        }

        self.online_replay_buffer.extend(samples.iter().copied());
        while self.online_replay_buffer.len() > self.online_replay_capacity_samples {
            self.online_replay_buffer.pop_front();
        }
    }

    fn take_online_replay(&mut self) -> Vec<f32> {
        if !matches!(self.asr, AsrRuntime::Online { .. }) || self.online_replay_buffer.is_empty() {
            return Vec::new();
        }

        self.online_replay_buffer.drain(..).collect()
    }

    fn decode_segment(&mut self, samples: &[f32]) -> Result<String, VocomError> {
        match &mut self.asr {
            AsrRuntime::Offline(recognizer) => {
                ASRModelBuilder::transcribe_samples(recognizer, self.target_sample_rate, samples)
            }
            AsrRuntime::Online { recognizer, stream, .. } => ASRModelBuilder::transcribe_samples_online(
                recognizer,
                stream,
                self.target_sample_rate,
                samples,
            ),
        }
    }

    fn decode_online_continuous(&mut self, samples: &[f32]) -> Option<Result<TranscriptEvent, VocomError>> {
        let AsrRuntime::Online {
            recognizer,
            stream,
            utterance_samples,
        } = &mut self.asr
        else {
            return None;
        };

        if samples.is_empty() {
            return None;
        }

        stream.accept_waveform(self.target_sample_rate, samples);
        *utterance_samples += samples.len();

        let started = Instant::now();
        while recognizer.is_ready(stream) {
            recognizer.decode(stream);
        }

        let is_endpoint = recognizer.is_endpoint(stream);
        if !is_endpoint {
            return None;
        }

        let result = match recognizer.get_result(stream) {
            Some(r) => r,
            None => {
                recognizer.reset(stream);
                *utterance_samples = 0;
                return Some(Err(VocomError::AsrConfig(
                    "failed to get online transcription result".to_string(),
                )));
            }
        };

        let text = result.text.trim().to_string();
        let sample_count = *utterance_samples;
        recognizer.reset(stream);
        *utterance_samples = 0;

        if text.is_empty() {
            return None;
        }

        Some(Ok(TranscriptEvent {
            text,
            sample_rate: self.target_sample_rate,
            sample_count,
            decode_latency_ms: started.elapsed().as_millis(),
        }))
    }

    fn append_history(&mut self, samples: &[f32]) {
        self.history_samples.extend(samples.iter().copied());
        self.total_samples_seen = self.total_samples_seen.saturating_add(samples.len());

        while self.history_samples.len() > self.history_capacity_samples {
            self.history_samples.pop_front();
            self.history_start_index = self.history_start_index.saturating_add(1);
        }
    }

    fn history_range(&self, start: usize, end: usize) -> Vec<f32> {
        if start >= end {
            return Vec::new();
        }

        let available_start = start.max(self.history_start_index);
        let available_end = end.min(self.total_samples_seen);
        if available_start >= available_end {
            return Vec::new();
        }

        let from = available_start - self.history_start_index;
        let to = available_end - self.history_start_index;
        self.history_samples
            .iter()
            .skip(from)
            .take(to - from)
            .copied()
            .collect()
    }
}

pub fn start_realtime_transcriber(
    asr_backend: AsrBackend,
    vad: VoiceActivityDetector,
    config: RealtimeConfig,
    aec_config: AecConfig,
    render_reference_consumer: RenderReferenceConsumer,
    duplex_gate: DuplexPlaybackGate,
) -> Result<RealtimeTranscriber, VocomError> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or(VocomError::NoInputDevice)?;

    let default_config = device.default_input_config()?;
    let input_sample_rate = default_config.sample_rate() as i32;
    let channels = default_config.channels() as usize;
    let stream_config: cpal::StreamConfig = default_config.clone().into();

    let chunk_frames = ((input_sample_rate as u32 * config.chunk_ms) / 1000).max(1) as usize;

    let (audio_tx, audio_rx) = bounded::<Vec<f32>>(config.audio_queue_capacity);
    let stream = build_input_stream(
        &device,
        &stream_config,
        default_config.sample_format(),
        channels,
        chunk_frames,
        audio_tx,
    )?;

    let target_sample_rate = config.target_sample_rate;

    let resampler = if input_sample_rate != target_sample_rate {
        Some(
            LinearResampler::create(input_sample_rate, target_sample_rate).ok_or_else(|| {
                VocomError::Stream(format!(
                    "failed to create resampler: {input_sample_rate} -> {target_sample_rate}"
                ))
            })?,
        )
    } else {
        None
    };

    let aec = AecProcessor::new(&aec_config, target_sample_rate, render_reference_consumer)?;
    let pre_speech_buffer_samples =
        (target_sample_rate as usize * config.pre_speech_buffer_ms as usize) / 1000;
    let history_capacity_samples = (target_sample_rate as usize) * HISTORY_SECONDS;
    let online_replay_capacity_samples =
        pre_speech_buffer_samples.max(target_sample_rate as usize / 5);

    stream.play()?;

    let asr = match asr_backend {
        AsrBackend::Offline(recognizer) => AsrRuntime::Offline(recognizer),
        AsrBackend::Online(recognizer) => AsrRuntime::Online {
            stream: recognizer.create_stream(),
            recognizer,
            utterance_samples: 0,
        },
    };

    Ok(RealtimeTranscriber {
        stream,
        audio_rx,
        asr,
        vad,
        aec,
        duplex_gate,
        resampler,
        target_sample_rate,
        pre_speech_buffer_samples,
        history_samples: VecDeque::with_capacity(history_capacity_samples),
        history_start_index: 0,
        total_samples_seen: 0,
        history_capacity_samples,
        tts_suppression_cooldown_ms: config.tts_suppression_cooldown_ms,
        tts_barge_in_rms_threshold: config.tts_barge_in_rms_threshold,
        tts_barge_in_render_ratio_min: config.tts_barge_in_render_ratio_min,
        tts_barge_in_render_rms_max_age_ms: config.tts_barge_in_render_rms_max_age_ms,
        tts_barge_in_persistence_ms: config.tts_barge_in_persistence_ms,
        barge_in_confident_ms: 0,
        barge_in_confidence: 0.0,
        tts_barge_in_confidence_threshold: config.tts_barge_in_confidence_threshold,
        tts_barge_in_confidence_smoothing: config.tts_barge_in_confidence_smoothing,
        barge_in_min_interval_ms: config.barge_in_min_interval_ms,
        online_replay_buffer: VecDeque::with_capacity(online_replay_capacity_samples),
        online_replay_capacity_samples,
        pending_events: VecDeque::with_capacity(config.event_queue_capacity),
    })
}

fn smooth_confidence(previous: f32, current: f32, smoothing: f32) -> f32 {
    let s = smoothing.clamp(0.0, 1.0);
    ((1.0 - s) * previous + s * current).clamp(0.0, 1.0)
}

fn compute_barge_in_confidence(
    near_end_rms: f32,
    render_rms: Option<f32>,
    rms_threshold: f32,
    ratio_min: f32,
    confident_ms: u64,
    persistence_ms: u64,
) -> f32 {
    let energy_component = (near_end_rms / rms_threshold.max(1e-6)).clamp(0.0, 1.0);

    let ratio_component = if let Some(render) = render_rms {
        let ratio = near_end_rms / render.max(1e-6);
        (ratio / ratio_min.max(1e-6)).clamp(0.0, 1.0)
    } else {
        0.6
    };

    let persistence_component = (confident_ms as f32 / persistence_ms.max(1) as f32)
        .clamp(0.0, 1.0);

    (0.40 * energy_component + 0.35 * ratio_component + 0.25 * persistence_component)
        .clamp(0.0, 1.0)
}

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    format: SampleFormat,
    channels: usize,
    chunk_frames: usize,
    audio_tx: Sender<Vec<f32>>,
) -> Result<Stream, VocomError> {
    let err_fn = |err: cpal::StreamError| {
        eprintln!("audio stream error: {err}");
    };

    match format {
        SampleFormat::F32 => {
            let tx = audio_tx.clone();
            let mut pending = Vec::<f32>::with_capacity(chunk_frames * 2);
            let stream = device.build_input_stream(
                config,
                move |data: &[f32], _| {
                    push_mono_frames_f32(data, channels, chunk_frames, &tx, &mut pending)
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        SampleFormat::I16 => {
            let tx = audio_tx.clone();
            let mut pending = Vec::<f32>::with_capacity(chunk_frames * 2);
            let stream = device.build_input_stream(
                config,
                move |data: &[i16], _| {
                    push_mono_frames_i16(data, channels, chunk_frames, &tx, &mut pending)
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        SampleFormat::U16 => {
            let tx = audio_tx;
            let mut pending = Vec::<f32>::with_capacity(chunk_frames * 2);
            let stream = device.build_input_stream(
                config,
                move |data: &[u16], _| {
                    push_mono_frames_u16(data, channels, chunk_frames, &tx, &mut pending)
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        other => Err(VocomError::Stream(format!(
            "unsupported input sample format: {other:?}"
        ))),
    }
}

fn push_mono_frames_f32(
    data: &[f32],
    channels: usize,
    chunk_frames: usize,
    tx: &Sender<Vec<f32>>,
    pending: &mut Vec<f32>,
) {
    for frame in data.chunks(channels) {
        let mono = frame.iter().copied().sum::<f32>() / channels as f32;
        pending.push(mono);
        flush_chunks(chunk_frames, tx, pending);
    }
}

fn push_mono_frames_i16(
    data: &[i16],
    channels: usize,
    chunk_frames: usize,
    tx: &Sender<Vec<f32>>,
    pending: &mut Vec<f32>,
) {
    for frame in data.chunks(channels) {
        let mono = frame
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .sum::<f32>()
            / channels as f32;
        pending.push(mono);
        flush_chunks(chunk_frames, tx, pending);
    }
}

fn push_mono_frames_u16(
    data: &[u16],
    channels: usize,
    chunk_frames: usize,
    tx: &Sender<Vec<f32>>,
    pending: &mut Vec<f32>,
) {
    for frame in data.chunks(channels) {
        let mono = frame
            .iter()
            .map(|&s| (s as f32 / u16::MAX as f32) * 2.0 - 1.0)
            .sum::<f32>()
            / channels as f32;
        pending.push(mono);
        flush_chunks(chunk_frames, tx, pending);
    }
}

fn flush_chunks(chunk_frames: usize, tx: &Sender<Vec<f32>>, pending: &mut Vec<f32>) {
    while pending.len() >= chunk_frames {
        let chunk: Vec<f32> = pending.drain(..chunk_frames).collect();
        match tx.try_send(chunk) {
            Ok(_) => {}
            Err(TrySendError::Full(_)) => {
                // Drop overflow chunks rather than blocking the realtime audio callback.
            }
            Err(TrySendError::Disconnected(_)) => return,
        }
    }
}

