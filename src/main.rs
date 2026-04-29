
mod asr_manager;
mod aec_manager;
mod config;
mod denoiser_manager;
mod duplex_audio;
mod engine;
mod engine_handle;
mod errors;
mod realtime_pipeline;
mod tts_sink;
mod tts_fx;
mod tts_manager;
mod vad_manager;
mod wakeword;
mod viot;

use config::EngineConfig;
use config::ConfigWatcher;
use engine::VocomEngine;
use errors::VocomError;
use std::time::{Duration, Instant};
use wakeword::MariaWakewordSpotter;

fn main() {
    if let Err(err) = run() {
        eprintln!("fatal error: {err}");
    }
}

fn run() -> Result<(), VocomError> {
    let config = EngineConfig::from_env()?;
    let wakeword_config = config.wakeword.clone();
    let mut engine = VocomEngine::start(config)?;
    let mut watcher = ConfigWatcher::new_from_env()?;
    let mut last_metrics_print = Instant::now();
    let mut wakeword = MariaWakewordSpotter::new_with_targets(
        Duration::from_millis(wakeword_config.cooldown_ms),
        wakeword_config.keyword,
        wakeword_config.variants,
    );
    let viot = viot::ViotController::new("http://192.168.71.1");
    engine.keep_alive();

    println!("Realtime transcription started. Press Ctrl+C to stop.");
    println!("Config profile can be changed with VOCOM_PROFILE=balanced|low_latency|noisy_room|laptop|laptop_earbud|close_speaker_edge");
    if watcher.has_path() {
        println!("Config hot reload is enabled via VOCOM_CONFIG_FILE");
    }

    loop {
        if last_metrics_print.elapsed() >= Duration::from_secs(10) {
            let m = engine.barge_in_telemetry();
            let p50 = m
                .latency_p50_ms
                .map(|v| v.to_string())
                .unwrap_or_else(|| "n/a".to_string());
            let p95 = m
                .latency_p95_ms
                .map(|v| v.to_string())
                .unwrap_or_else(|| "n/a".to_string());
            let rolling_p50 = m
                .rolling_latency_p50_ms
                .map(|v| v.to_string())
                .unwrap_or_else(|| "n/a".to_string());
            let rolling_p95 = m
                .rolling_latency_p95_ms
                .map(|v| v.to_string())
                .unwrap_or_else(|| "n/a".to_string());
            println!(
                "[barge-metrics] state={} requested={} rejected_low_rms={} rejected_render_ratio={} rejected_persistence={} rejected_confidence={} ducked={} stopped={} latency_samples={} latency_p50_ms={} latency_p95_ms={} rolling_requested={} rolling_rejected_low_rms={} rolling_rejected_render_ratio={} rolling_rejected_persistence={} rolling_rejected_confidence={} rolling_ducked={} rolling_stopped={} rolling_latency_samples={} rolling_latency_p50_ms={} rolling_latency_p95_ms={}",
                m.state,
                m.requested,
                m.rejected_low_rms,
                m.rejected_render_ratio,
                m.rejected_persistence,
                m.rejected_confidence,
                m.ducked,
                m.stopped,
                m.latency_samples,
                p50,
                p95,
                m.rolling_requested,
                m.rolling_rejected_low_rms,
                m.rolling_rejected_render_ratio,
                m.rolling_rejected_persistence,
                m.rolling_rejected_confidence,
                m.rolling_ducked,
                m.rolling_stopped,
                m.rolling_latency_samples,
                rolling_p50,
                rolling_p95,
            );
            if !m.recent_transitions.is_empty() {
                println!("[barge-transitions]");
                for t in &m.recent_transitions {
                    println!(
                        "  +{}ms  {}->{} ({})",
                        t.at_ms % 100_000, // relative ms tail for readability
                        t.from,
                        t.to,
                        t.reason
                    );
                }
            }
            last_metrics_print = Instant::now();
        }

        if let Some(action) = engine.handle_barge_in() {
            println!("[barge-in] {action}");
        }

        if watcher.changed()? {
            println!("Config file changed, reloading engine...");
            match EngineConfig::from_env() {
                Ok(new_config) => {
                    let next_wakeword = new_config.wakeword.clone();
                    match VocomEngine::start(new_config) {
                        Ok(new_engine) => {
                            wakeword = MariaWakewordSpotter::new_with_targets(
                                Duration::from_millis(next_wakeword.cooldown_ms),
                                next_wakeword.keyword,
                                next_wakeword.variants,
                            );
                            engine = new_engine;
                            engine.keep_alive();
                            println!("Engine reload completed.");
                        }
                        Err(err) => {
                            eprintln!("Engine reload failed, keeping previous config: {err}");
                        }
                    }
                }
                Err(err) => {
                    eprintln!("Config reload failed, keeping previous config: {err}");
                }
            }
        }

        if let Some(event) = engine.recv_timeout(Duration::from_millis(40))? {
            let audio_ms = (event.sample_count as f32 / event.sample_rate as f32) * 1000.0;
            println!(
                "[audio={audio_ms:.0}ms decode={}ms] {}",
                event.decode_latency_ms, event.text
            );

            if let Some(m) = viot.check(&event.text) {
                println!("[viot] matched='{}' score={:.2} → {}", m.label, m.score, m.endpoint);
                viot.execute(&m);
            }

            if let Some(hit) = wakeword.detect(&event.text) {
                println!(
                    "[wakeword] keyword={} matched={} confidence={:.2} transcript='{}'",
                    hit.keyword,
                    hit.matched_token,
                    hit.confidence,
                    hit.transcript
                );
            }

            if engine.synthesize_mock_reply(&event.text)? {
                println!("[tts] played mock response");
            }
        }
    }
}
