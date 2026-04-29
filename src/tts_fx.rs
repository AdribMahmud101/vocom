use std::f32::consts::PI;

use crate::config::TtsFxConfig;

pub fn apply_tts_fx_in_place(samples: &mut [f32], sample_rate: i32, cfg: &TtsFxConfig) {
    if !cfg.enabled || samples.is_empty() || sample_rate <= 0 {
        return;
    }

    let mut chain = FxChain::new(sample_rate as f32, cfg);
    chain.process(samples);
}

struct FxChain {
    robot_mix: f32,
    ring_phase: f32,
    ring_step: f32,
    distortion_drive: f32,
    distortion_norm: f32,
    output_gain: f32,
    lp_alpha: f32,
    hp_alpha: f32,
    lp_state: f32,
    hp_state: f32,
    hp_prev_x: f32,
    // Clarity EQ stages.
    eq_presence: Biquad,
    eq_air: Biquad,
    // 3-band dynamics.
    mb_lp_state: f32,
    mb_lp_alpha: f32,
    mb_hp_state: f32,
    mb_hp_prev_x: f32,
    mb_hp_alpha: f32,
    low_env: f32,
    mid_env: f32,
    high_env: f32,
    comp_attack: f32,
    comp_release: f32,
    comp_threshold: f32,
    comp_ratio: f32,
    delay: Vec<f32>,
    delay_idx: usize,
    echo_mix: f32,
    echo_feedback: f32,
    // Short room reverb.
    room_mix: f32,
    room_comb: [Vec<f32>; 3],
    room_comb_idx: [usize; 3],
    room_ap: [Vec<f32>; 2],
    room_ap_idx: [usize; 2],
    room_feedback: f32,
    room_ap_feedback: f32,
    // Subtle short-delay modulation for "sci-fi" motion.
    micro_buf: Vec<f32>,
    micro_write_idx: usize,
    micro_phase: f32,
    micro_step: f32,
    micro_base_samples: f32,
    micro_depth_samples: f32,
    micro_mix: f32,
    // Harmonic shimmer layer (2nd harmonic blend).
    shimmer_mix: f32,
    // Dynamic presence lift to keep articulation after heavy FX.
    presence_env: f32,
    presence_release: f32,
    presence_amount: f32,
    // Final de-esser.
    de_ess_env: f32,
    de_ess_release: f32,
    de_ess_threshold: f32,
    de_ess_strength: f32,
    de_ess_hp_state: f32,
    de_ess_hp_prev_x: f32,
    de_ess_hp_alpha: f32,
}

impl FxChain {
    fn new(sample_rate: f32, cfg: &TtsFxConfig) -> Self {
        let sr = sample_rate.max(8_000.0);
        let low_cut_hz = cfg.low_cut_hz.clamp(20.0, 1_200.0);
        let high_cut_hz = cfg.high_cut_hz.clamp(low_cut_hz + 10.0, 8_000.0);
        let robot_mix = cfg.robot_mix.clamp(0.0, 1.0);
        let ring_mod_hz = cfg.ring_mod_hz.clamp(0.0, 240.0);
        let distortion_drive = cfg.distortion_drive.clamp(0.5, 4.0);
        let echo_delay_ms = cfg.echo_delay_ms.clamp(0, 1_000);
        let echo_mix = cfg.echo_mix.clamp(0.0, 1.0);
        let echo_feedback = cfg.echo_feedback.clamp(0.0, 0.98);
        let output_gain = cfg.output_gain.clamp(0.0, 2.0);

        let lp_alpha = one_pole_lowpass_alpha(high_cut_hz, sr);
        let hp_alpha = one_pole_highpass_alpha(low_cut_hz, sr);
        let ring_step = if ring_mod_hz <= 0.0 {
            0.0
        } else {
            2.0 * PI * ring_mod_hz / sr
        };
        let delay_len = ((echo_delay_ms as f32 / 1_000.0) * sr).round() as usize;
        let delay = if delay_len == 0 || echo_mix <= 0.0 {
            Vec::new()
        } else {
            vec![0.0; delay_len]
        };
        let distortion_norm = distortion_drive.tanh().max(1e-6);
        let eq_presence = Biquad::peaking(sr, 3_200.0, 0.9, 2.4);
        let eq_air = Biquad::high_shelf(sr, 10_500.0, 0.7, 1.6);
        let comp_attack = envelope_coeff(2.0, sr);
        let comp_release = envelope_coeff(85.0, sr);
        let comp_threshold = 0.18;
        let comp_ratio = 3.6;
        let mb_lp_alpha = one_pole_lowpass_alpha(250.0, sr);
        let mb_hp_alpha = one_pole_highpass_alpha(3_200.0, sr);
        let micro_mix = (0.08 + robot_mix * 0.18).clamp(0.0, 0.28);
        let micro_lfo_hz = (0.15 * ring_mod_hz + 0.35).clamp(0.25, 9.0);
        let micro_step = 2.0 * PI * micro_lfo_hz / sr;
        let micro_base_samples = ((1.5 + robot_mix * 2.2) * sr / 1_000.0).clamp(1.0, 128.0);
        let micro_depth_samples = ((0.5 + robot_mix * 1.8) * sr / 1_000.0).clamp(0.0, 128.0);
        let micro_len = ((micro_base_samples + micro_depth_samples + 3.0).ceil() as usize).max(8);
        let micro_buf = vec![0.0; micro_len];
        let shimmer_mix = (0.10 + (distortion_drive - 1.0).max(0.0) * 0.12).clamp(0.0, 0.34);
        let presence_release = (1.0 - (12.0 / sr)).clamp(0.90, 0.9995);
        let presence_amount = (0.10 + robot_mix * 0.18).clamp(0.0, 0.34);
        let room_mix = (echo_mix * 0.55).clamp(0.0, 0.16);
        let room_feedback = 0.58;
        let room_ap_feedback = 0.45;
        let room_comb = [
            vec![0.0; ((sr * 0.019).round() as usize).max(8)],
            vec![0.0; ((sr * 0.023).round() as usize).max(8)],
            vec![0.0; ((sr * 0.029).round() as usize).max(8)],
        ];
        let room_ap = [
            vec![0.0; ((sr * 0.0047).round() as usize).max(4)],
            vec![0.0; ((sr * 0.0033).round() as usize).max(4)],
        ];
        let de_ess_release = envelope_coeff(45.0, sr);
        let de_ess_threshold = 0.11;
        let de_ess_strength = 0.42;
        let de_ess_hp_alpha = one_pole_highpass_alpha(5_500.0, sr);

        Self {
            robot_mix,
            ring_phase: 0.0,
            ring_step,
            distortion_drive,
            distortion_norm,
            output_gain,
            lp_alpha,
            hp_alpha,
            lp_state: 0.0,
            hp_state: 0.0,
            hp_prev_x: 0.0,
            eq_presence,
            eq_air,
            mb_lp_state: 0.0,
            mb_lp_alpha,
            mb_hp_state: 0.0,
            mb_hp_prev_x: 0.0,
            mb_hp_alpha,
            low_env: 0.0,
            mid_env: 0.0,
            high_env: 0.0,
            comp_attack,
            comp_release,
            comp_threshold,
            comp_ratio,
            delay,
            delay_idx: 0,
            echo_mix,
            echo_feedback,
            room_mix,
            room_comb,
            room_comb_idx: [0, 0, 0],
            room_ap,
            room_ap_idx: [0, 0],
            room_feedback,
            room_ap_feedback,
            micro_buf,
            micro_write_idx: 0,
            micro_phase: 0.0,
            micro_step,
            micro_base_samples,
            micro_depth_samples,
            micro_mix,
            shimmer_mix,
            presence_env: 0.0,
            presence_release,
            presence_amount,
            de_ess_env: 0.0,
            de_ess_release,
            de_ess_threshold,
            de_ess_strength,
            de_ess_hp_state: 0.0,
            de_ess_hp_prev_x: 0.0,
            de_ess_hp_alpha,
        }
    }

    fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            let mut x = s.clamp(-1.0, 1.0);

            // High-pass to remove low-end rumble.
            let hp = self.hp_alpha * (self.hp_state + x - self.hp_prev_x);
            self.hp_state = hp;
            self.hp_prev_x = x;
            x = hp;

            // Low-pass to narrow bandwidth into a "radio/assistant" tone.
            self.lp_state += self.lp_alpha * (x - self.lp_state);
            x = self.lp_state;

            // Presence + air EQ for crisp articulation.
            x = self.eq_presence.process(x);
            x = self.eq_air.process(x);

            // 3-band split: low (<250Hz), high (>3.2kHz), mid = rest.
            self.mb_lp_state += self.mb_lp_alpha * (x - self.mb_lp_state);
            let low = self.mb_lp_state;
            let hp = self.mb_hp_alpha * (self.mb_hp_state + x - self.mb_hp_prev_x);
            self.mb_hp_state = hp;
            self.mb_hp_prev_x = x;
            let high = hp;
            let mid = x - low - high;
            let low_c = compress_band(
                low,
                &mut self.low_env,
                0.92,
                self.comp_attack,
                self.comp_release,
                self.comp_threshold,
                self.comp_ratio,
            );
            let mid_c = compress_band(
                mid,
                &mut self.mid_env,
                1.00,
                self.comp_attack,
                self.comp_release,
                self.comp_threshold,
                self.comp_ratio,
            );
            let high_c = compress_band(
                high,
                &mut self.high_env,
                1.08,
                self.comp_attack,
                self.comp_release,
                self.comp_threshold,
                self.comp_ratio,
            );
            x = low_c + mid_c + high_c;

            // Ring-mod blend to introduce subtle robotic edge.
            let ring = self.ring_phase.sin();
            self.ring_phase += self.ring_step;
            if self.ring_phase > 2.0 * PI {
                self.ring_phase -= 2.0 * PI;
            }
            let robot = x * ring;
            x = x * (1.0 - self.robot_mix) + robot * self.robot_mix;

            // Harmonic shimmer to push the voice into "synthetic assistant" space.
            let shimmer = (2.0 * x).tanh();
            x = x * (1.0 - self.shimmer_mix) + shimmer * self.shimmer_mix;

            // Micro-modulated short delay (flanger-ish) for subtle futuristic motion.
            if !self.micro_buf.is_empty() && self.micro_mix > 0.0 {
                let lfo = self.micro_phase.sin();
                self.micro_phase += self.micro_step;
                if self.micro_phase > 2.0 * PI {
                    self.micro_phase -= 2.0 * PI;
                }
                let variable_delay = self.micro_base_samples + self.micro_depth_samples * (0.5 + 0.5 * lfo);
                let read_offset = variable_delay.max(0.0) as usize;
                let len = self.micro_buf.len();
                let read_idx = (self.micro_write_idx + len - (read_offset % len)) % len;
                let delayed = self.micro_buf[read_idx];
                self.micro_buf[self.micro_write_idx] = x;
                self.micro_write_idx = (self.micro_write_idx + 1) % len;
                x = x * (1.0 - self.micro_mix) + delayed * self.micro_mix;
            }

            // Controlled non-linearity for body and "mechanical" bite.
            x = (x * self.distortion_drive).tanh() / self.distortion_norm;

            // Short echo for cinematic tail.
            if !self.delay.is_empty() {
                let delayed = self.delay[self.delay_idx];
                let input_for_delay = x + delayed * self.echo_feedback;
                self.delay[self.delay_idx] = input_for_delay.clamp(-1.0, 1.0);
                self.delay_idx = (self.delay_idx + 1) % self.delay.len();
                x += delayed * self.echo_mix;
            }

            // Short room reverb (felt space, low wetness).
            if self.room_mix > 0.0 {
                let mut r = 0.0f32;
                for i in 0..self.room_comb.len() {
                    let idx = self.room_comb_idx[i];
                    let d = self.room_comb[i][idx];
                    self.room_comb[i][idx] = (x + d * self.room_feedback).clamp(-1.0, 1.0);
                    self.room_comb_idx[i] = (idx + 1) % self.room_comb[i].len();
                    r += d;
                }
                r /= self.room_comb.len() as f32;
                for i in 0..self.room_ap.len() {
                    let idx = self.room_ap_idx[i];
                    let buf = self.room_ap[i][idx];
                    let y = -r + buf;
                    self.room_ap[i][idx] = (r + buf * self.room_ap_feedback).clamp(-1.0, 1.0);
                    self.room_ap_idx[i] = (idx + 1) % self.room_ap[i].len();
                    r = y;
                }
                x = x * (1.0 - self.room_mix) + r * self.room_mix;
            }

            // Dynamic presence compensation after heavy processing to keep consonants crisp.
            let abs_x = x.abs();
            self.presence_env = (self.presence_env * self.presence_release).max(abs_x);
            let presence_boost = 1.0 + (1.0 - self.presence_env).clamp(0.0, 1.0) * self.presence_amount;
            x *= presence_boost;

            // Final de-esser to tame boosted sibilance.
            let sib = self.de_ess_hp_alpha * (self.de_ess_hp_state + x - self.de_ess_hp_prev_x);
            self.de_ess_hp_state = sib;
            self.de_ess_hp_prev_x = x;
            let sib_abs = sib.abs();
            self.de_ess_env = (self.de_ess_env * self.de_ess_release).max(sib_abs);
            if self.de_ess_env > self.de_ess_threshold {
                let over = (self.de_ess_env - self.de_ess_threshold) / (1.0 - self.de_ess_threshold);
                let cut = (over * self.de_ess_strength).clamp(0.0, 0.7);
                x *= 1.0 - cut;
            }

            x *= self.output_gain;
            *s = x.clamp(-1.0, 1.0);
        }
    }

}

fn compress_band(
    x: f32,
    env: &mut f32,
    makeup: f32,
    attack: f32,
    release: f32,
    threshold: f32,
    ratio: f32,
) -> f32 {
    let abs_x = x.abs();
    let coeff = if abs_x > *env { attack } else { release };
    *env = *env * coeff + abs_x * (1.0 - coeff);
    if *env <= threshold {
        return x * makeup;
    }
    let over = (*env / threshold).max(1.0);
    let gain = over.powf(-(ratio - 1.0) / ratio);
    (x * gain * makeup).clamp(-1.0, 1.0)
}

#[derive(Clone, Copy)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn process(&mut self, x: f32) -> f32 {
        let y = x * self.b0 + self.z1;
        self.z1 = x * self.b1 + self.z2 - self.a1 * y;
        self.z2 = x * self.b2 - self.a2 * y;
        y
    }

    fn peaking(sample_rate: f32, freq: f32, q: f32, gain_db: f32) -> Self {
        let a = 10.0f32.powf(gain_db / 40.0);
        let nyq = (sample_rate * 0.5).max(1.0);
        let f = freq.clamp(20.0, nyq * 0.45);
        let w0 = 2.0 * PI * (f / sample_rate.max(1.0));
        let alpha = w0.sin() / (2.0 * q.max(0.1));
        let cos = w0.cos();
        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos;
        let a2 = 1.0 - alpha / a;
        Self::norm(b0, b1, b2, a0, a1, a2)
    }

    fn high_shelf(sample_rate: f32, freq: f32, slope: f32, gain_db: f32) -> Self {
        let a = 10.0f32.powf(gain_db / 40.0);
        let nyq = (sample_rate * 0.5).max(1.0);
        let f = freq.clamp(100.0, nyq * 0.45);
        let w0 = 2.0 * PI * (f / sample_rate.max(1.0));
        let cos = w0.cos();
        let sin = w0.sin();
        let s = slope.max(0.1);
        let alpha = sin / 2.0 * (((a + 1.0 / a) * (1.0 / s - 1.0) + 2.0).max(0.0)).sqrt();
        let beta = 2.0 * a.sqrt() * alpha;
        let b0 = a * ((a + 1.0) + (a - 1.0) * cos + beta);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos - beta);
        let a0 = (a + 1.0) - (a - 1.0) * cos + beta;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos);
        let a2 = (a + 1.0) - (a - 1.0) * cos - beta;
        Self::norm(b0, b1, b2, a0, a1, a2)
    }

    fn norm(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        let a0n = if a0.abs() < 1e-8 { 1.0 } else { a0 };
        Self {
            b0: b0 / a0n,
            b1: b1 / a0n,
            b2: b2 / a0n,
            a1: a1 / a0n,
            a2: a2 / a0n,
            z1: 0.0,
            z2: 0.0,
        }
    }
}

fn one_pole_lowpass_alpha(cutoff_hz: f32, sample_rate: f32) -> f32 {
    let cutoff = cutoff_hz.max(1.0);
    let dt = 1.0 / sample_rate.max(1.0);
    let rc = 1.0 / (2.0 * PI * cutoff);
    (dt / (rc + dt)).clamp(0.0, 1.0)
}

fn one_pole_highpass_alpha(cutoff_hz: f32, sample_rate: f32) -> f32 {
    let cutoff = cutoff_hz.max(1.0);
    let dt = 1.0 / sample_rate.max(1.0);
    let rc = 1.0 / (2.0 * PI * cutoff);
    (rc / (rc + dt)).clamp(0.0, 1.0)
}

fn envelope_coeff(time_ms: f32, sample_rate: f32) -> f32 {
    let t = (time_ms / 1_000.0).max(1e-4);
    (-1.0 / (sample_rate.max(1.0) * t)).exp().clamp(0.0, 0.9999)
}

#[cfg(test)]
mod tests {
    use super::apply_tts_fx_in_place;
    use crate::config::TtsFxConfig;

    #[test]
    fn tts_fx_keeps_samples_finite_and_bounded() {
        let mut samples = vec![0.0f32; 4_000];
        for (i, s) in samples.iter_mut().enumerate() {
            let t = i as f32 / 16_000.0;
            *s = (2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.6;
        }

        let mut cfg = TtsFxConfig::default();
        cfg.enabled = true;
        apply_tts_fx_in_place(&mut samples, 16_000, &cfg);

        assert!(samples.iter().all(|v| v.is_finite()));
        assert!(samples.iter().all(|v| *v >= -1.0 && *v <= 1.0));
    }

    #[test]
    fn disabled_tts_fx_is_noop() {
        let mut samples = vec![0.1f32, -0.2, 0.3, -0.4, 0.5];
        let original = samples.clone();
        let cfg = TtsFxConfig::default();
        apply_tts_fx_in_place(&mut samples, 16_000, &cfg);
        assert_eq!(samples, original);
    }

    #[test]
    fn enabled_tts_fx_has_material_delta() {
        let mut samples = vec![0.0f32; 8_000];
        for (i, s) in samples.iter_mut().enumerate() {
            let t = i as f32 / 16_000.0;
            let a = (2.0 * std::f32::consts::PI * 180.0 * t).sin() * 0.55;
            let b = (2.0 * std::f32::consts::PI * 360.0 * t).sin() * 0.25;
            *s = a + b;
        }
        let original = samples.clone();

        let mut cfg = TtsFxConfig::default();
        cfg.enabled = true;
        apply_tts_fx_in_place(&mut samples, 16_000, &cfg);

        let mut sum = 0.0f32;
        for (a, b) in original.iter().zip(samples.iter()) {
            sum += (a - b).abs();
        }
        let mean_abs_delta = sum / original.len() as f32;
        assert!(
            mean_abs_delta > 0.01,
            "expected material FX delta, got mean_abs_delta={mean_abs_delta}"
        );
    }
}
