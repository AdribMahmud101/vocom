
mod asr_manager;
mod vad_manager;

use asr_manager::{ASRModelBuilder, ASRVariant};

fn main() {
    let recognizer_result = ASRModelBuilder::new(ASRVariant::Whisper)
        .encoder("models/sherpa-onnx-whisper-base.en/base.en-encoder.onnx")
        .decoder("models/sherpa-onnx-whisper-base.en/base.en-decoder.onnx")
        .tokens("models/sherpa-onnx-whisper-base.en/base.en-tokens.txt")
        .num_threads(4)
        .build();

    // match ASRModelBuilder::transcribe(
    //     recognizer_result,
    //     "models/sherpa-onnx-whisper-base.en/test_wavs/0.wav",
    // ) {
    //     Ok(text) => println!("Transcription: {}", text),
    //     Err(e) => eprintln!("Transcription failed: {}", e),
    // }
}