use std::time::{Duration, Instant};

use vocom::{EngineCommand, EngineConfig, EngineEvent, VocomEngineHandle};

const WAIT_FOR_WAKEWORD_STATE: Duration = Duration::from_secs(5);
const WAIT_FOR_ENGINE_INIT: Duration = Duration::from_secs(45);

fn main() {
    if let Err(err) = run() {
        eprintln!("wakeword entrypoint workflow test failed: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = EngineConfig::from_env()?;
    config.wakeword.entrypoint_enabled = true;

    println!("[1/6] starting engine handle in wakeword entrypoint mode");
    let mut handle = VocomEngineHandle::start(config)?;

    println!("[2/6] waiting for waiting_for_wakeword state");
    wait_for_boot_state(&handle, "waiting_for_wakeword", WAIT_FOR_WAKEWORD_STATE)?;

    println!("[3/6] feeding non-wakeword transcript");
    handle.send_command(EngineCommand::SubmitWakewordTranscript {
        text: "hello assistant what time is it".to_string(),
    })?;

    std::thread::sleep(Duration::from_millis(120));

    println!("[4/6] feeding wakeword transcript to trigger hands-free startup");
    handle.send_command(EngineCommand::SubmitWakewordTranscript {
        text: "hey maria please wake up".to_string(),
    })?;

    println!("[5/6] waiting for engine_initialized state");
    wait_for_boot_state(&handle, "engine_initialized", WAIT_FOR_ENGINE_INIT)?;

    println!("[6/6] verifying command path and shutting down");
    handle.send_command(EngineCommand::GetDuplexMode)?;
    if let Some(event) = handle.poll_event_timeout(Duration::from_secs(2)) {
        println!("[event] {}", describe_event(&event));
    }

    handle.shutdown()?;
    println!("workflow test completed successfully");
    Ok(())
}

fn wait_for_boot_state(
    handle: &VocomEngineHandle,
    expected: &str,
    timeout: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if let Some(event) = handle.poll_event_timeout(Duration::from_millis(200)) {
            println!("[event] {}", describe_event(&event));
            match event {
                EngineEvent::BootStateChanged(state) if state == expected => return Ok(()),
                EngineEvent::Error(err) => {
                    return Err(format!("engine emitted error while waiting for {expected}: {err}").into())
                }
                _ => {}
            }
        }
    }

    Err(format!("timed out waiting for boot state '{expected}'").into())
}

fn describe_event(event: &EngineEvent) -> String {
    match event {
        EngineEvent::BootStateChanged(state) => format!("BootStateChanged({state})"),
        EngineEvent::TranscriptReady(t) => format!("TranscriptReady(text='{}')", t.text),
        EngineEvent::BargeInTransition(v) => format!("BargeInTransition({v})"),
        EngineEvent::TelemetrySnapshot(t) => {
            format!("TelemetrySnapshot(state={}, requested={})", t.state, t.requested)
        }
        EngineEvent::DuplexModeChanged(mode) => format!("DuplexModeChanged({mode})"),
        EngineEvent::Error(err) => format!("Error({err})"),
        EngineEvent::Shutdown => "Shutdown".to_string(),
    }
}