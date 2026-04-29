use sherpa_onnx::{
    LinearResampler, OfflineSpeechDenoiserConfig, OfflineSpeechDenoiserModelConfig,
    OnlineSpeechDenoiser, OnlineSpeechDenoiserConfig,
};

use crate::config::{DenoiserConfig, DenoiserModelFamily};
use crate::errors::VocomError;

pub struct DenoiserProcessor {
    denoiser: OnlineSpeechDenoiser,
    to_denoiser: Option<LinearResampler>,
    from_denoiser: Option<LinearResampler>,
    denoiser_sample_rate: i32,
}

impl DenoiserProcessor {
    pub fn new(cfg: &DenoiserConfig, input_sample_rate: i32) -> Result<Option<Self>, VocomError> {
        if !cfg.enabled {
            return Ok(None);
        }

        if cfg.model_path.is_empty() {
            return Err(VocomError::DenoiserConfig(
                "denoiser.model_path must be set when denoiser is enabled".to_string(),
            ));
        }

        let mut model = OfflineSpeechDenoiserModelConfig::default();
        model.num_threads = cfg.num_threads;
        model.debug = cfg.debug;
        model.provider = Some(cfg.provider.clone());

        match cfg.family {
            DenoiserModelFamily::Gtcrn => {
                model.gtcrn.model = Some(cfg.model_path.clone());
            }
            DenoiserModelFamily::Dpdfnet => {
                model.dpdfnet.model = Some(cfg.model_path.clone());
            }
        }

        let offline = OfflineSpeechDenoiserConfig { model };
        let online_cfg = OnlineSpeechDenoiserConfig {
            model: offline.model,
        };

        let denoiser = OnlineSpeechDenoiser::create(&online_cfg).ok_or_else(|| {
            VocomError::DenoiserConfig("failed to create online speech denoiser".to_string())
        })?;

        let denoiser_sample_rate = denoiser.sample_rate();
        let to_denoiser = if input_sample_rate != denoiser_sample_rate {
            Some(LinearResampler::create(input_sample_rate, denoiser_sample_rate).ok_or_else(
                || {
                    VocomError::DenoiserConfig(format!(
                        "failed to create denoiser resampler: {input_sample_rate} -> {denoiser_sample_rate}"
                    ))
                },
            )?)
        } else {
            None
        };

        let from_denoiser = if input_sample_rate != denoiser_sample_rate {
            Some(LinearResampler::create(denoiser_sample_rate, input_sample_rate).ok_or_else(
                || {
                    VocomError::DenoiserConfig(format!(
                        "failed to create denoiser resampler: {denoiser_sample_rate} -> {input_sample_rate}"
                    ))
                },
            )?)
        } else {
            None
        };

        Ok(Some(Self {
            denoiser,
            to_denoiser,
            from_denoiser,
            denoiser_sample_rate,
        }))
    }

    pub fn process_chunk(&mut self, input: Vec<f32>, sample_rate: i32) -> Result<Vec<f32>, VocomError> {
        if input.is_empty() {
            return Ok(input);
        }

        let samples = if let Some(r) = &self.to_denoiser {
            r.resample(&input, false)
        } else {
            input
        };

        let denoised = self.denoiser.run(&samples, self.denoiser_sample_rate);
        let output = if let Some(r) = &self.from_denoiser {
            r.resample(&denoised.samples, false)
        } else {
            denoised.samples
        };

        if sample_rate != self.denoiser_sample_rate && output.is_empty() {
            return Ok(Vec::new());
        }

        Ok(output)
    }
}
