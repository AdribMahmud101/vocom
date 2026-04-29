use std::time::{Duration, Instant};

use vocom::wakeword::MariaWakewordSpotter;
use vocom::{EngineCommand, EngineConfig, EngineEvent, VocomEngine, VocomEngineHandle};

const WAKEWORD_LISTEN_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_SOAK_SECONDS: u64 = 30 * 60;

fn main() {
    if let Err(err) = run() {
        eprintln!("runtime workflow test failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let full_config = EngineConfig::from_env()?;
    let soak_duration = soak_duration_from_env();

    // Bootstrap listener: mic + ASR only, no TTS, to detect wakeword hands-free.
    let mut bootstrap_config = full_config.clone();
    bootstrap_config.wakeword.entrypoint_enabled = false;
    bootstrap_config.tts.enabled = false;

    let mut spotter = MariaWakewordSpotter::new_with_targets(
        Duration::from_millis(full_config.wakeword.cooldown_ms),
        full_config.wakeword.keyword.clone(),
        full_config.wakeword.variants.clone(),
    );

    println!("starting live runtime workflow test");
    println!("phase 1: listening from mic for wakeword (no typing required)");
    println!("say wakeword variants out loud, e.g. 'maria' or 'mariam'");

    let mut bootstrap_engine = VocomEngine::start(bootstrap_config)?;
    bootstrap_engine.keep_alive();

    let phase1_start = Instant::now();
    loop {
        if phase1_start.elapsed() > WAKEWORD_LISTEN_TIMEOUT {
            return Err(format!(
                "wakeword not detected within {}s",
                WAKEWORD_LISTEN_TIMEOUT.as_secs()
            )
            .into());
        }

        if let Some(event) = bootstrap_engine.recv_timeout(Duration::from_millis(40))? {
            println!("[bootstrap-asr] {}", event.text);
            if let Some(hit) = spotter.detect(&event.text) {
                println!(
                    "[wakeword] keyword={} matched={} confidence={:.2}",
                    hit.keyword, hit.matched_token, hit.confidence
                );
                break;
            }
        }
    }

    drop(bootstrap_engine);

    println!("phase 2: wakeword detected, starting full engine workflow");
    println!("you can now speak naturally; press Ctrl+C to stop");

    let mut main_config = full_config;
    main_config.wakeword.entrypoint_enabled = false;
    let handle = VocomEngineHandle::start(main_config)?;
    let soak_started_at = Instant::now();

    let mut last_poll = Instant::now();
    let mut last_telemetry = Instant::now();
    let mut half_duplex_gate_active = false;
    let mut leak_violations = 0u64;
    let mut soak_stats = SoakLeakStats::default();
    loop {
        if let Some(dur) = soak_duration {
            if soak_started_at.elapsed() >= dur {
                println!(
                    "[soak-summary] elapsed_s={} pressure_samples={} max_asr_epoch={} half_duplex_gate_engaged={} stale_asr_tasks_dropped={} stale_asr_events_dropped={} leak_violations={}",
                    dur.as_secs(),
                    soak_stats.pressure_samples,
                    soak_stats.max_asr_epoch,
                    soak_stats.half_duplex_gate_engaged,
                    soak_stats.stale_asr_tasks_dropped,
                    soak_stats.stale_asr_events_dropped,
                    leak_violations,
                );
                return Ok(());
            }
        }
        if last_poll.elapsed() >= Duration::from_millis(40) {
            handle.send_command(EngineCommand::PollTranscript)?;
            last_poll = Instant::now();
        }
        if last_telemetry.elapsed() >= Duration::from_millis(250) {
            handle.send_command(EngineCommand::GetTelemetry)?;
            last_telemetry = Instant::now();
        }

        if let Some(event) = handle.poll_event_timeout(Duration::from_millis(120)) {
            match event {
                EngineEvent::TranscriptReady(t) => {
                    if half_duplex_gate_active {
                        leak_violations = leak_violations.saturating_add(1);
                        return Err(format!(
                            "half-duplex leak assertion failed: transcript emitted while mute gate active (violations={leak_violations}, text='{}')",
                            t.text
                        )
                        .into());
                    }
                    println!("[asr] {} (decode={}ms)", t.text, t.decode_latency_ms);
                }
                EngineEvent::BargeInTransition(v) => {
                    if let Some(active) = parse_pressure_bool(&v, "half_duplex_gate_active") {
                        half_duplex_gate_active = active;
                    }
                    soak_stats.observe_pressure(&v);
                    println!("[barge] {v}");
                }
                EngineEvent::TelemetrySnapshot(t) => {
                    println!("[telemetry] state={} requested={}", t.state, t.requested);
                }
                EngineEvent::DuplexModeChanged(mode) => {
                    println!("[duplex] {mode}");
                }
                EngineEvent::BootStateChanged(state) => {
                    println!("[boot] {state}");
                }
                EngineEvent::Error(err) => {
                    println!("[error] {err}");
                }
                EngineEvent::Shutdown => {
                    println!("[event] shutdown");
                    return Ok(());
                }
            }
        }
    }
}

#[derive(Default)]
struct SoakLeakStats {
    pressure_samples: u64,
    max_asr_epoch: u64,
    half_duplex_gate_engaged: u64,
    stale_asr_tasks_dropped: u64,
    stale_asr_events_dropped: u64,
}

impl SoakLeakStats {
    fn observe_pressure(&mut self, line: &str) {
        if !line.starts_with("pressure ") {
            return;
        }
        self.pressure_samples = self.pressure_samples.saturating_add(1);
        if let Some(v) = parse_pressure_u64(line, "asr_epoch") {
            self.max_asr_epoch = self.max_asr_epoch.max(v);
        }
        if let Some(v) = parse_pressure_u64(line, "half_duplex_gate_engaged") {
            self.half_duplex_gate_engaged = v;
        }
        if let Some(v) = parse_pressure_u64(line, "stale_asr_tasks_dropped") {
            self.stale_asr_tasks_dropped = v;
        }
        if let Some(v) = parse_pressure_u64(line, "stale_asr_events_dropped") {
            self.stale_asr_events_dropped = v;
        }
    }
}

fn soak_duration_from_env() -> Option<Duration> {
    match std::env::var("VOCOM_SOAK_SECONDS") {
        Ok(raw) => {
            let s = raw.trim().parse::<u64>().ok()?;
            if s == 0 {
                None
            } else {
                Some(Duration::from_secs(s))
            }
        }
        Err(_) => Some(Duration::from_secs(DEFAULT_SOAK_SECONDS)),
    }
}

fn parse_pressure_bool(line: &str, key: &str) -> Option<bool> {
    let token = format!("{key}=");
    let start = line.find(&token)?;
    let value = &line[start + token.len()..];
    if value.starts_with("true") {
        Some(true)
    } else if value.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn parse_pressure_u64(line: &str, key: &str) -> Option<u64> {
    let token = format!("{key}=");
    let start = line.find(&token)?;
    let value = &line[start + token.len()..];
    let end = value
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(value.len());
    if end == 0 {
        return None;
    }
    value[..end].parse::<u64>().ok()
}
