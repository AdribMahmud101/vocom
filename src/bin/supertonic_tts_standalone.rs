use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;

use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig,
    OfflineTtsSupertonicModelConfig,
};

fn main() {
    if let Err(err) = run() {
        eprintln!("fatal error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_dir = "models/sherpa-onnx-supertonic-tts-int8-2026-03-06".to_string();
    let mut all_voices_demo = false;
    let mut demo_text = "Today as always, men fall into two groups: slaves and free men.".to_string();

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--all-voices-demo" => {
                all_voices_demo = true;
            }
            "--text" => {
                let next = i + 1;
                if next >= args.len() {
                    return Err("--text requires a value".to_string());
                }
                demo_text = args[next].clone();
                i = next;
            }
            val if val.starts_with("--") => {
                return Err(format!("unknown option: {val}"));
            }
            val => {
                model_dir = val.to_string();
            }
        }
        i += 1;
    }

    let duration_predictor = req_file(&model_dir, "duration_predictor.int8.onnx")?;
    let text_encoder = req_file(&model_dir, "text_encoder.int8.onnx")?;
    let vector_estimator = req_file(&model_dir, "vector_estimator.int8.onnx")?;
    let vocoder = req_file(&model_dir, "vocoder.int8.onnx")?;
    let tts_json = req_file(&model_dir, "tts.json")?;
    let unicode_indexer = req_file(&model_dir, "unicode_indexer.bin")?;
    let voice_style = req_file(&model_dir, "voice.bin")?;

    let config = OfflineTtsConfig {
        model: OfflineTtsModelConfig {
            supertonic: OfflineTtsSupertonicModelConfig {
                duration_predictor: Some(duration_predictor),
                text_encoder: Some(text_encoder),
                vector_estimator: Some(vector_estimator),
                vocoder: Some(vocoder),
                tts_json: Some(tts_json),
                unicode_indexer: Some(unicode_indexer),
                voice_style: Some(voice_style),
            },
            num_threads: 2,
            debug: false,
            provider: Some("cpu".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };

    let tts = OfflineTts::create(&config).ok_or_else(|| "failed to create OfflineTts".to_string())?;

    println!("Supertonic ready. sample_rate={} speakers={}", tts.sample_rate(), tts.num_speakers());

    fs::create_dir_all("target/tts").map_err(|e| format!("failed to create target/tts: {e}"))?;

    if all_voices_demo {
        run_all_voices_demo(&tts, &demo_text)?;
        return Ok(());
    }

    println!("Type text and press Enter. Type 'exit' to quit.");

    loop {
        print!("> ");
        io::stdout().flush().map_err(|e| format!("stdout flush failed: {e}"))?;

        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .map_err(|e| format!("stdin read failed: {e}"))?;

        let text = line.trim();
        if text.eq_ignore_ascii_case("exit") {
            break;
        }
        if text.is_empty() {
            continue;
        }

        let mut extra = HashMap::new();
        extra.insert("lang".to_string(), serde_json::json!("en"));

        let gen_cfg = GenerationConfig {
            sid: 6,
            num_steps: 5,
            speed: 1.1,
            extra: Some(extra),
            ..Default::default()
        };

        let audio = tts
            .generate_with_config(text, &gen_cfg, None::<fn(&[f32], f32) -> bool>)
            .ok_or_else(|| "generation failed".to_string())?;

        let out = "target/tts/generated-supertonic-standalone.wav";
        if !audio.save(out) {
            return Err(format!("failed to save wav: {out}"));
        }

        if !play_wav(out) {
            println!("Saved {out}, but no audio player was found (tried aplay, paplay, ffplay).");
        }
    }

    Ok(())
}

fn run_all_voices_demo(tts: &OfflineTts, text: &str) -> Result<(), String> {
    let speakers = tts.num_speakers();
    if speakers <= 0 {
        return Err("model reports no available speakers".to_string());
    }

    println!("Running all-voices demo for {speakers} speakers...");

    for sid in 0..speakers {
        let mut extra = HashMap::new();
        extra.insert("lang".to_string(), serde_json::json!("en"));

        let gen_cfg = GenerationConfig {
            sid,
            num_steps: 5,
            speed: 1.1,
            extra: Some(extra),
            ..Default::default()
        };

        let audio = tts
            .generate_with_config(text, &gen_cfg, None::<fn(&[f32], f32) -> bool>)
            .ok_or_else(|| format!("generation failed for sid={sid}"))?;

        let out = format!("target/tts/generated-supertonic-voice-{sid}.wav");
        if !audio.save(&out) {
            return Err(format!("failed to save wav: {out}"));
        }

        println!("Generated {out}");
        if !play_wav(&out) {
            println!("No audio player found; kept {out}");
        }
    }

    Ok(())
}

fn req_file(model_dir: &str, name: &str) -> Result<String, String> {
    let p = Path::new(model_dir).join(name);
    if p.exists() {
        Ok(p.to_string_lossy().to_string())
    } else {
        Err(format!("missing required supertonic asset: {}", p.to_string_lossy()))
    }
}

fn play_wav(path: &str) -> bool {
    let candidates: [(&str, &[&str]); 3] = [
        ("aplay", &["-q", path]),
        ("paplay", &[path]),
        ("ffplay", &["-nodisp", "-autoexit", "-loglevel", "error", path]),
    ];

    for (program, args) in candidates {
        let status = Command::new(program).args(args).status();
        if let Ok(ok) = status {
            if ok.success() {
                return true;
            }
        }
    }

    false
}
