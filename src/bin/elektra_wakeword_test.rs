use std::time::Duration;

use vocom::wakeword::MariaWakewordSpotter;
use vocom::{EngineConfig, VocomEngine};

fn main() {
    if let Err(err) = run() {
        eprintln!("wakeword test failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = EngineConfig::from_env()?;
    let wakeword_config = config.wakeword.clone();
    let mut engine = VocomEngine::start(config)?;
    let mut spotter = MariaWakewordSpotter::new_with_targets(
        Duration::from_millis(wakeword_config.cooldown_ms),
        wakeword_config.keyword,
        wakeword_config.variants,
    );

    println!("Wakeword realtime test started using runtime VOCOM_WAKEWORD/VOCOM_WAKEWORD_VARIANTS.");
    println!("Press Ctrl+C to stop.");

    loop {
        if let Some(event) = engine.recv_timeout(Duration::from_millis(40))? {
            println!(
                "[asr] {} (decode={}ms)",
                event.text,
                event.decode_latency_ms
            );

            if let Some(hit) = spotter.detect(&event.text) {
                println!(
                    "[wakeword-hit] keyword={} matched={} confidence={:.2}",
                    hit.keyword,
                    hit.matched_token,
                    hit.confidence
                );
            }
        }
    }
}