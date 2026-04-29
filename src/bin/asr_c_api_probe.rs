use std::env;
use std::ffi::{CStr, CString};
use std::mem;

use sherpa_onnx::Wave;
use sherpa_onnx_sys::offline_asr::{
    OfflineRecognizerConfig, SherpaOnnxAcceptWaveformOffline,
    SherpaOnnxCreateOfflineRecognizer, SherpaOnnxCreateOfflineStream,
    SherpaOnnxDecodeOfflineStream, SherpaOnnxDestroyOfflineRecognizer,
    SherpaOnnxDestroyOfflineStream, SherpaOnnxDestroyOfflineStreamResultJson,
    SherpaOnnxGetOfflineStreamResultAsJson,
};

fn main() {
    let encoder = env::var("VOCOM_PROBE_ENCODER")
        .unwrap_or_else(|_| "models/sherpa-onnx-whisper-base.en/base.en-encoder.int8.onnx".to_string());
    let decoder = env::var("VOCOM_PROBE_DECODER")
        .unwrap_or_else(|_| "models/sherpa-onnx-whisper-base.en/base.en-decoder.int8.onnx".to_string());
    let tokens = env::var("VOCOM_PROBE_TOKENS")
        .unwrap_or_else(|_| "models/sherpa-onnx-whisper-base.en/base.en-tokens.txt".to_string());
    let wav = env::var("VOCOM_PROBE_WAV")
        .unwrap_or_else(|_| "models/sherpa-onnx-whisper-base.en/test_wavs/0.wav".to_string());
    let num_threads = env::var("VOCOM_PROBE_THREADS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(1);

    let encoder_c = CString::new(encoder).unwrap();
    let decoder_c = CString::new(decoder).unwrap();
    let tokens_c = CString::new(tokens).unwrap();
    let provider_c = CString::new("cpu").unwrap();
    let model_type_c = CString::new("whisper").unwrap();
    let task_c = CString::new("transcribe").unwrap();
    let decoding_c = CString::new("greedy_search").unwrap();

    eprintln!("asr_c_api_probe: creating recognizer");
    let recognizer = unsafe {
        let mut cfg: OfflineRecognizerConfig = mem::zeroed();
        cfg.feat_config.sample_rate = 16000;
        cfg.feat_config.feature_dim = 80;
        cfg.model_config.whisper.encoder = encoder_c.as_ptr();
        cfg.model_config.whisper.decoder = decoder_c.as_ptr();
        cfg.model_config.whisper.task = task_c.as_ptr();
        cfg.model_config.whisper.tail_paddings = -1;
        cfg.model_config.tokens = tokens_c.as_ptr();
        cfg.model_config.num_threads = num_threads;
        cfg.model_config.provider = provider_c.as_ptr();
        cfg.model_config.model_type = model_type_c.as_ptr();
        cfg.decoding_method = decoding_c.as_ptr();
        cfg.max_active_paths = 4;
        cfg.hotwords_score = 1.5;
        SherpaOnnxCreateOfflineRecognizer(&cfg)
    };
    if recognizer.is_null() {
        eprintln!("asr_c_api_probe: recognizer is null");
        std::process::exit(2);
    }
    eprintln!("asr_c_api_probe: recognizer created");

    let Some(wave) = Wave::read(&wav) else {
        eprintln!("asr_c_api_probe: failed to read wav={wav}");
        unsafe { SherpaOnnxDestroyOfflineRecognizer(recognizer) };
        std::process::exit(3);
    };

    unsafe {
        let stream = SherpaOnnxCreateOfflineStream(recognizer);
        SherpaOnnxAcceptWaveformOffline(
            stream,
            wave.sample_rate(),
            wave.samples().as_ptr(),
            wave.samples().len() as i32,
        );
        SherpaOnnxDecodeOfflineStream(recognizer, stream);
        let result = SherpaOnnxGetOfflineStreamResultAsJson(stream);
        if !result.is_null() {
            eprintln!("{}", CStr::from_ptr(result).to_string_lossy());
            SherpaOnnxDestroyOfflineStreamResultJson(result);
        }
        SherpaOnnxDestroyOfflineStream(stream);
        SherpaOnnxDestroyOfflineRecognizer(recognizer);
    }
}
