1. Organize architecture boundaries - completed
2. Implement efficient threading for capture and processing - completed
3. Add typed error handling across modules - completed
4. Compile and verify refactor - completed
5. Document refactor and runtime requirements - completed
6. Add robust unified engine configuration system - completed
7. Add startup profile + env override support - completed
8. Add config validation and docs - completed
9. Add JSON config loader support - completed
10. Add runtime config hot reload support - completed
11. Refactor TTS to non-blocking worker playback - completed
12. Add render reference bus for duplex audio - completed
13. Add feature-gated WebRTC AEC manager scaffold - completed
14. Wire AEC stage before VAD/ASR in realtime pipeline - completed
15. Document full duplex + AEC phase implementation - completed
16. Add interruptible TTS stop path for barge-in - completed
17. Wire duplex barge-in request from near-end speech detection - completed
18. Add graded barge-in policy (duck then stop) - completed
19. Expose barge-in duck/stop tuning knobs in runtime config - completed
20. Add anti-self-barge filter using render-vs-near-end RMS ratio - completed
21. Add persistence-based barge-in confidence gate - completed
22. Add reason-coded barge-in metrics and periodic telemetry - completed
23. Add interruption latency telemetry (p50/p95) - completed
24. Add robustness profiles for laptop_earbud and close_speaker_edge - completed
25. Implement dedicated interruption confidence scorer - completed
31. Smooth TTS gain transitions to remove visible ducking and start burst - completed
26. Add explicit barge-in state machine transitions - completed
27. Add rolling 60s metrics and latency telemetry - completed
28. Build replay-based robustness regression harness - planned
29. Harden Flutter/Kotlin Android audio service boundary - planned
30. Tune and lock device-class production profile baselines - planned
32. Audit voice UX regressions and document root causes - completed
33. Fix render-ratio false-trigger leak in barge-in path - completed
34. Improve barge-in FSM reaction cadence in main loop - completed
35. Mitigate online first-word clipping with suppression replay buffer - completed
36. Reduce render-reference frame loss under queue pressure - completed
37. Add dynamic VAD thresholding (noise-adaptive) - completed
38. Add denoise stage (offline/online speech denoiser) - completed
39. Add input overflow telemetry + mitigation policy - completed
40. Add offline post-roll + short-silence merge for segments - completed
41. Add model-specific online endpointing profiles - completed
42. Add input normalization + clip guard - completed
43. Split barge-in VAD vs segmentation VAD - completed
44. Fix online ASR support error message for nemotron - completed
45. Evaluate sherpa-onnx auxiliary components for robustness - completed

-- Sherpa-onnx components to evaluate (docs.rs source)
- Offline/online punctuation
- Offline/online speech denoiser
- Spoken language identification
- Speaker embedding + diarization
- Keyword spotting
- Audio tagging