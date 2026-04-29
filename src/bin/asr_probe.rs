use std::env;
use std::path::Path;

use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OnlineRecognizer, OnlineRecognizerConfig, Wave,
};

fn main() {
    let mode = env::var("VOCOM_PROBE_MODE").unwrap_or_else(|_| "offline".to_string());
    let variant = env::var("VOCOM_PROBE_VARIANT").unwrap_or_else(|_| "whisper".to_string());
    let encoder = env::var("VOCOM_PROBE_ENCODER")
        .unwrap_or_else(|_| "models/sherpa-onnx-whisper-base.en/base.en-encoder.onnx".to_string());
    let decoder = env::var("VOCOM_PROBE_DECODER")
        .unwrap_or_else(|_| "models/sherpa-onnx-whisper-base.en/base.en-decoder.onnx".to_string());
    let joiner = env::var("VOCOM_PROBE_JOINER").unwrap_or_default();
    let tokens = env::var("VOCOM_PROBE_TOKENS")
        .unwrap_or_else(|_| "models/sherpa-onnx-whisper-base.en/base.en-tokens.txt".to_string());
    let wav = env::var("VOCOM_PROBE_WAV")
        .unwrap_or_else(|_| "models/sherpa-onnx-whisper-base.en/test_wavs/0.wav".to_string());
    let num_threads = env::var("VOCOM_PROBE_THREADS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(2);
    let decoding_method =
        env::var("VOCOM_PROBE_DECODING_METHOD").unwrap_or_else(|_| "modified_beam_search".to_string());
    let decode_ms = env::var("VOCOM_PROBE_DECODE_MS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(1200);
    let tail_paddings = env::var("VOCOM_PROBE_TAIL_PADDINGS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(0);
    let language = env::var("VOCOM_PROBE_LANGUAGE").unwrap_or_default();

    let is_online = mode.eq_ignore_ascii_case("online")
        || variant.eq_ignore_ascii_case("streaming_zipformer")
        || variant.eq_ignore_ascii_case("nemotron_streaming");

    if is_online {
        eprintln!(
            "asr_probe init(online): variant={variant} encoder={encoder} decoder={decoder} joiner={joiner} tokens={tokens} threads={num_threads} decoding={decoding_method}"
        );
        probe_online(
            encoder,
            decoder,
            joiner,
            tokens,
            num_threads,
            decoding_method,
            decode_ms,
        );
        return;
    }

    eprintln!(
        "asr_probe init(offline): variant={variant} encoder={encoder} decoder={decoder} tokens={tokens} threads={num_threads}"
    );
    probe_offline(
        encoder,
        decoder,
        tokens,
        wav,
        num_threads,
        decode_ms,
        tail_paddings,
        language,
    );
}

fn probe_offline(
    encoder: String,
    decoder: String,
    tokens: String,
    wav: String,
    num_threads: i32,
    decode_ms: u32,
    tail_paddings: i32,
    language: String,
) {
    let mut cfg = OfflineRecognizerConfig::default();
    cfg.model_config.tokens = Some(tokens);
    cfg.model_config.num_threads = num_threads;
    cfg.model_config.provider = Some("cpu".to_string());
    cfg.model_config.model_type = Some("whisper".to_string());
    cfg.decoding_method = Some("greedy_search".to_string());
    cfg.hotwords_score = 1.5;
    cfg.model_config.whisper.encoder = Some(encoder);
    cfg.model_config.whisper.decoder = Some(decoder);
    cfg.model_config.whisper.language = Some(language);
    cfg.model_config.whisper.task = Some("transcribe".to_string());
    cfg.model_config.whisper.tail_paddings = tail_paddings;
    cfg.model_config.whisper.enable_token_timestamps = false;
    cfg.model_config.whisper.enable_segment_timestamps = false;

    let Some(recognizer) = OfflineRecognizer::create(&cfg) else {
        eprintln!("asr_probe error: failed to create OfflineRecognizer");
        std::process::exit(2);
    };
    eprintln!("asr_probe: recognizer initialized");

    if !Path::new(&wav).exists() {
        eprintln!("asr_probe: wav not found, done");
        return;
    }

    let Some(wave) = Wave::read(&wav) else {
        eprintln!("asr_probe error: failed to read wav={wav}");
        std::process::exit(3);
    };
    let stream = recognizer.create_stream();
    stream.accept_waveform(wave.sample_rate(), wave.samples());
    recognizer.decode(&stream);
    if let Some(result) = stream.get_result() {
        eprintln!("asr_probe transcript: {}", result.text);
    } else {
        eprintln!("asr_probe warning: no transcript result");
    }

    // Additional synthetic decode stress to catch runtime incompatibilities early.
    let synthetic_samples = ((16_000_u64 * decode_ms as u64) / 1000) as usize;
    if synthetic_samples > 0 {
        let stream = recognizer.create_stream();
        let zeros = vec![0.0_f32; synthetic_samples];
        stream.accept_waveform(16_000, &zeros);
        recognizer.decode(&stream);
        let _ = stream.get_result();
    }
}

fn probe_online(
    encoder: String,
    decoder: String,
    joiner: String,
    tokens: String,
    num_threads: i32,
    decoding_method: String,
    decode_ms: u32,
) {
    if joiner.trim().is_empty() {
        eprintln!("asr_probe error: missing joiner path for online probe");
        std::process::exit(4);
    }

    let mut cfg = OnlineRecognizerConfig::default();
    cfg.model_config.tokens = Some(tokens);
    cfg.model_config.num_threads = num_threads;
    cfg.model_config.provider = Some("cpu".to_string());
    cfg.decoding_method = Some(decoding_method);
    cfg.enable_endpoint = true;
    cfg.rule1_min_trailing_silence = 0.35;
    cfg.rule2_min_trailing_silence = 0.8;
    cfg.rule3_min_utterance_length = 8.0;
    cfg.model_config.transducer.encoder = Some(encoder);
    cfg.model_config.transducer.decoder = Some(decoder);
    cfg.model_config.transducer.joiner = Some(joiner);

    let Some(recognizer) = OnlineRecognizer::create(&cfg) else {
        eprintln!("asr_probe error: failed to create OnlineRecognizer");
        std::process::exit(5);
    };
    eprintln!("asr_probe: online recognizer initialized");

    let stream = recognizer.create_stream();
    let synthetic_samples = ((16_000_u64 * decode_ms as u64) / 1000) as usize;
    let sample_count = synthetic_samples.max(16_000);
    let zeros = vec![0.0_f32; sample_count];
    stream.accept_waveform(16_000, &zeros);
    stream.input_finished();

    while recognizer.is_ready(&stream) {
        recognizer.decode(&stream);
    }

    let _ = recognizer.get_result(&stream);
    recognizer.reset(&stream);
    eprintln!("asr_probe: online decode pass finished");
}
