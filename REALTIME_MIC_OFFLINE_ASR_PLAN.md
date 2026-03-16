# Real-Time Microphone Input Plan for Offline Recognizer

## Goal
Build low-latency microphone capture that works continuously, then decode completed speech segments using `sherpa_onnx::OfflineRecognizer`.

This keeps inference fully offline while still feeling real-time to the user.

## Why this design
`OfflineRecognizer` is best for whole utterances, not frame-by-frame streaming decoding. So the correct approach is:
1. Capture mic audio continuously.
2. Detect speech boundaries (start/end of utterance).
3. Send each complete utterance to OfflineRecognizer.
4. Print result quickly and continue listening.

## Target behavior
- App starts and warms up recognizer.
- Microphone captures audio in real time.
- User speaks naturally.
- On silence end, one utterance is decoded.
- Transcript appears with timing metrics.
- Loop continues until user exits.

## Dependencies to add
Update [Cargo.toml](Cargo.toml):
- `cpal` for cross-platform microphone capture.
- `crossbeam-channel` (or `std::sync::mpsc`) for audio frame transfer between threads.
- `anyhow` for ergonomic error handling.
- Optional: `rubato` for high-quality resampling if input sample rate is not 16 kHz.

Suggested additions:
```toml
[dependencies]
sherpa-onnx = "0.1.10"
cpal = "0.15"
crossbeam-channel = "0.5"
anyhow = "1"
```

## High-level architecture
Use a 3-stage pipeline:

1. Capture thread
- Reads PCM from default input device using CPAL callback.
- Converts input samples to `f32` mono.
- Resamples to 16000 Hz if needed.
- Sends fixed-size chunks (e.g., 20 ms = 320 samples) to a channel.

2. Segmenter/VAD thread
- Consumes chunks from channel.
- Maintains a rolling buffer.
- Detects speech start/end using VAD (or energy threshold fallback).
- Produces utterance buffers (Vec<f32>) when end-of-speech is detected.

3. Decode thread
- Reuses a single `OfflineRecognizer` instance.
- For each utterance: create stream, accept waveform, decode, collect text.
- Emits result + timing info.

## Implementation phases

### Phase 1: Refactor current code into reusable pieces
In [src/ASRManager.rs](src/ASRManager.rs):
- Keep your builder but add a helper to return a ready recognizer from config.
- Keep recognizer creation centralized (single model load).

In [src/main.rs](src/main.rs):
- Move WAV-only demo path into function `run_file_mode(...)`.
- Add future entry point `run_mic_mode(...)`.

Output of this phase:
- No behavior change yet.
- Cleaner boundaries for mic integration.

### Phase 2: Add microphone capture (CPAL)
Create [src/mic.rs](src/mic.rs) with:
- `MicConfig` (device, sample_rate, channels, chunk_ms).
- `start_capture(tx)` that starts CPAL stream and sends normalized mono chunks.

Key details:
- Handle `f32`, `i16`, and `u16` input formats.
- Downmix stereo to mono (`(L+R)/2`).
- Prefer 16 kHz at device config if supported.

Output of this phase:
- You can print chunk statistics continuously (no decoding yet).

### Phase 3: Add utterance segmentation
Create [src/segmenter.rs](src/segmenter.rs):
- State machine: `Idle -> InSpeech -> Ended`.
- Configurable thresholds:
  - `speech_start_ms`
  - `speech_end_ms`
  - `min_utterance_ms`
  - `max_utterance_ms`

Preferred VAD options:
- Option A (best): Sherpa VAD APIs if available in your Rust crate version.
- Option B (fallback): energy + zero-crossing heuristic.

Output of this phase:
- Program logs speech segments with durations.

### Phase 4: Decode segmented utterances with OfflineRecognizer
Create [src/pipeline.rs](src/pipeline.rs):
- `decode_utterance(recognizer, sample_rate, samples)` helper.
- For each segment:
  - `let stream = recognizer.create_stream();`
  - `stream.accept_waveform(16000, &samples);`
  - `recognizer.decode(&stream);`
  - `stream.get_result()`

Print:
- transcript
- utterance duration
- decode latency
- utterance-level RTF

Output of this phase:
- End-to-end mic -> text loop running.

### Phase 5: Command-line integration
Extend existing args in [src/main.rs](src/main.rs):
- `--mic` (enable microphone mode)
- `--device <name>` optional
- `--sample-rate` optional override
- `--chunk-ms` default 20
- `--vad-threshold`, `--silence-ms`, `--min-utterance-ms`

Keep existing file mode flags for regression checks.

Output of this phase:
- Single binary supports WAV and live mic modes.

### Phase 6: Quality and stability hardening
- Add Ctrl+C graceful shutdown.
- Add bounded channels to prevent RAM growth.
- Drop overly long utterances safely.
- Add startup warmup decode for first-response latency.
- Add clear error messages for no mic device / unsupported format.

## Data and latency targets
- Chunk size: 20 ms (320 samples @16 kHz).
- End-of-speech timeout: 500-900 ms silence.
- Expected first transcript latency: ~700-1500 ms after speech ends (depends on CPU/model).

## Risks and mitigations
1. Device sample rate mismatch
- Mitigation: explicit resampler path and logging of effective rate.

2. Noisy environments produce false segments
- Mitigation: configurable thresholds, optional push-to-talk mode.

3. Callback overruns / dropped frames
- Mitigation: lightweight callback, move heavy work off callback thread.

4. Large utterances increase latency
- Mitigation: cap max utterance length and force decode.

## Minimal acceptance checklist
- App runs in `--mic` mode without crash.
- Detects at least 3 distinct utterances in one session.
- Produces transcripts for each utterance.
- Keeps listening after each decode.
- CPU usage remains stable over 10 minutes.

## Suggested file layout after implementation
- [src/main.rs](src/main.rs): CLI, mode selection
- [src/ASRManager.rs](src/ASRManager.rs): recognizer builder/config
- [src/mic.rs](src/mic.rs): CPAL capture and chunking
- [src/segmenter.rs](src/segmenter.rs): VAD/silence endpointing
- [src/pipeline.rs](src/pipeline.rs): decode orchestration

## Rollout order
1. Land refactor + capture only.
2. Land segmentation.
3. Land decode pipeline.
4. Tune thresholds with real room noise.
5. Add optional enhancements (partial text simulation, punctuation, speaker id).

## Optional future improvement
If you want true token-by-token low-latency partial transcripts while speaking (not only after utterance end), add a second mode based on sherpa-onnx online recognizer APIs. Keep this offline recognizer mode for robust final transcription quality.
