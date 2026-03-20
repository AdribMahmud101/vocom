# ASR Manager Usage Guide

This document explains how to use the ASR module in [src/asr_manager.rs](src/asr_manager.rs).

## What the module provides

The module currently exposes:

- `ASRVariant`: Model type selector (`Moonshinev2`, `Whisper`, `Unknown`)
- `ASRModelBuilder`: Builder for recognizer configuration
- `build()`: Creates `OfflineRecognizer`
- `trascribe(...)`: Reads a wav file and prints transcript

## Current behavior to know first

Before using it, note these current implementation details:

- `ASRVariant::get_variant(...)` returns `&ASRVariant` (a reference), while `ASRModelBuilder::new(...)` expects an owned `ASRVariant`.
- The method name is currently `trascribe(...)` (typo in name). Use this exact name from [src/asr_manager.rs](src/asr_manager.rs).
- `trascribe(...)` prints output to console and returns `()`, so you cannot directly capture transcript as a return value yet.
- `num_threads` and `provider` are currently stored by builder methods but are not applied into `OfflineRecognizerConfig` in [src/asr_manager.rs](src/asr_manager.rs).

So the current usage path is:

1. Build recognizer using `ASRModelBuilder`.
2. Pass the build result into `ASRModelBuilder::trascribe(...)`.

## Basic usage pattern with transcribe function

Use this flow:

1. Choose variant (`Moonshinev2` or `Whisper`).
2. Set encoder, decoder, and tokens file paths.
3. Call `build()`.
4. Call `ASRModelBuilder::trascribe(...)` with recognizer result and wav path.
5. Read transcript from console output (`Transcription: ...`).

## Example in [src/main.rs](src/main.rs)

```rust
mod asr_manager;

use asr_manager::{ASRModelBuilder, ASRVariant};

fn main() {
    let recognizer_result = ASRModelBuilder::new(ASRVariant::Moonshinev2)
        .encoder("models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27/encoder_model.ort")
        .decoder("models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27/decoder_model_merged.ort")
        .tokens("models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27/tokens.txt")
        .build();

    ASRModelBuilder::trascribe(
        recognizer_result,
        "models/sherpa-onnx-whisper-base.en/test_wavs/0.wav".to_string(),
    );
}
```

## Running it

From project root:

```bash
cargo run
```

## Model path examples

Moonshine files in this repo:

- encoder: `models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27/encoder_model.ort`
- decoder: `models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27/decoder_model_merged.ort`
- tokens: `models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27/tokens.txt`

Whisper files in this repo:

- encoder: `models/sherpa-onnx-whisper-base.en/base.en-encoder.onnx`
- decoder: `models/sherpa-onnx-whisper-base.en/base.en-decoder.onnx`
- tokens: `models/sherpa-onnx-whisper-base.en/base.en-tokens.txt`

## Common errors and fixes

- "Encoder path is missing" / "Decoder path is missing" / "Tokens path is missing"
: One of the builder paths was not set.

- "ASR model is not recognized"
: You used `ASRVariant::Unknown` or unsupported variant.

- "Failed to create recognizer from config"
: Usually bad model files, wrong model/variant pairing, or runtime library path issue.

- Panic from `Wave::read(...).expect(...)`
: Happens when wav path is invalid or unreadable.

## Note about naming

If you say transcribe function, that maps to the current method named `trascribe(...)` in [src/asr_manager.rs](src/asr_manager.rs).
