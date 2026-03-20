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
sherpa-onnx = "0.1.12"
cpal = "0.15"
crossbeam-channel = "0.5"
anyhow = "1"
```

## Deep Dive: "On silence end, one utterance is decoded"

This means you do not decode every tiny audio chunk.
You decode only when the current speech segment is considered finished.

### Practical meaning

1. Mic audio arrives continuously in short chunks (for example 20 ms each).
2. While speech is detected, chunks are appended to one utterance buffer.
3. Once enough trailing silence is detected, the utterance is closed.
4. The whole utterance buffer is sent once to OfflineRecognizer.
5. Transcript is emitted, then buffer/state are reset for the next utterance.

### Why this is needed for OfflineRecognizer

OfflineRecognizer is optimized for full segments, not token-by-token decoding while speaking.
So the endpoint detector (silence-based segmentation) is the trigger that tells the decoder:
"This speech unit is done. Decode now."

### Core state machine

- Idle
: No active speech yet.

- InSpeech
: Speech is active; keep collecting chunks.

- MaybeEnd
: A short silence started, but not enough to end yet.

- Ended
: Silence duration passed threshold; finalize and decode utterance.

### Required counters and buffers

- utterance_samples: Vec<f32>
: Collected audio for current utterance.

- speech_ms
: Total detected speech time in current utterance.

- silence_ms
: Continuous trailing silence duration while inside an utterance.

- chunk_ms
: Duration per audio chunk (usually 20 ms).

### Endpointing rules (simple and robust)

Assume each chunk is classified as speech or non-speech by VAD (or energy fallback).

When chunk is speech:
- If state was Idle, start a new utterance.
- Append chunk to utterance_samples.
- speech_ms += chunk_ms.
- silence_ms = 0.
- state = InSpeech.

When chunk is non-speech and state is InSpeech or MaybeEnd:
- Append chunk only if you want trailing context.
- silence_ms += chunk_ms.
- state = MaybeEnd.

If silence_ms >= speech_end_ms:
- state = Ended.
- If speech_ms >= min_utterance_ms: decode utterance_samples.
- Else: drop as too short/noise.
- Reset utterance_samples, speech_ms, silence_ms.
- state = Idle.

Safety cutoff:
- If speech_ms >= max_utterance_ms, force decode to avoid huge buffers and high latency.

### Starter thresholds

- chunk_ms: 20
- speech_end_ms: 700
- min_utterance_ms: 300
- max_utterance_ms: 15000

Tune per environment:
- Noisy room: increase speech_end_ms and min_utterance_ms.
- Fast response needed: reduce speech_end_ms to around 500.

### Pseudocode

```text
for chunk in mic_stream:
  is_speech = vad(chunk)

  if is_speech:
    if state == Idle:
      start_new_utterance()
      state = InSpeech
    append(chunk)
    speech_ms += chunk_ms
    silence_ms = 0
  else if state != Idle:
    silence_ms += chunk_ms
    state = MaybeEnd

  if state != Idle and speech_ms >= max_utterance_ms:
    decode_and_reset()

  if state == MaybeEnd and silence_ms >= speech_end_ms:
    if speech_ms >= min_utterance_ms:
      decode_and_reset()
    else:
      reset_without_decode()
```

### Decode call at endpoint

For each ended utterance:

1. Create stream.
2. Accept waveform at 16000 Hz.
3. Decode.
4. Read result.

Equivalent flow in Rust:

```text
let stream = recognizer.create_stream();
stream.accept_waveform(16000, &utterance_samples);
recognizer.decode(&stream);
let result = stream.get_result();
```

### Common beginner mistakes

1. Decoding every chunk.
: Causes high CPU usage and unstable results.

2. Ending utterance too early.
: Chops words; increase speech_end_ms.

3. No max utterance cap.
: Large memory/latency spikes for long speech.

4. Doing decode in audio callback thread.
: Can cause dropouts and audio overruns.

See also beginner glossary and examples in [ASR_BEGINNER_TERMS.md](ASR_BEGINNER_TERMS.md).

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
