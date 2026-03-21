
use std::path::Path;

use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, Wave};

#[allow(unused)]
pub enum ASRVariant {
    Moonshinev2,
    Whisper,
    Unknown,
}

#[allow(unused)]
pub struct ASRModelBuilder{
    variant: ASRVariant,
    encoder: Option<String>,
    decoder: Option<String>,
    tokens: Option<String>,
    num_threads: i32,
    provider: Option<String>,

    // whisper specific
    language: Option<String>,
    task: Option<String>,
    tail_paddings: i32,
    enable_token_timestamps: bool,
    enable_segment_timestamps: bool,

}


impl ASRVariant {

    pub fn get_variant(file_name: &str)-> Self {

        let file = file_name.to_lowercase();

        if file.contains("moonshine"){
            Self::Moonshinev2
        } else if file.contains("whisper"){
            Self::Whisper
        } else{
            Self::Unknown
        }
         
    }

}


impl ASRModelBuilder{

    pub fn new(variant: ASRVariant) -> Self{
        Self{
            variant,
            encoder: None,
            decoder: None,
            tokens: None,
            num_threads: 2, // defualt to 2 threads
            provider: Some("cpu".to_string()), // default provider
            // whisper specific
            language: Some("en".to_string()),
            task: Some("transcribe".to_string()),
            tail_paddings: -1,
            enable_token_timestamps: false, // false by default
            enable_segment_timestamps: false,
        }
    }

    pub fn encoder(mut self, path: &str)-> Self{
        self.encoder = Some(path.to_string());
        self
    }

    pub fn decoder(mut self, path: &str)-> Self{
        self.decoder = Some(path.to_string());
        self
    }

    pub fn tokens(mut self, path: &str)-> Self{
        self.tokens = Some(path.to_string());
        self
    }

    pub fn num_threads(mut self, threads: i32)-> Self{
        self.num_threads = threads;
        self
    }

    pub fn provider(mut self, value: &str)-> Self{
        self.provider = Some(value.to_string());
        self
    }


    pub fn build(self)-> Result<OfflineRecognizer,String>{
        let encoder = self.encoder;
        let decoder = self.decoder;
        let tokens = self.tokens;

        // if !Path::new(&encoder).exists() {
        //     return Err(format!("Encoder file does not exist: {encoder}"));
        // }
        // if !Path::new(&decoder).exists() {
        //     return Err(format!("Decoder file does not exist: {decoder}"));
        // }
        // if !Path::new(&tokens).exists() {
        //     return Err(format!("Tokens file does not exist: {tokens}"));
        // }

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.tokens = tokens;
        config.model_config.num_threads = self.num_threads;
        config.model_config.provider = self.provider;
   
        match self.variant {
            ASRVariant::Moonshinev2 => {
                config.model_config.moonshine.encoder = encoder;
                config.model_config.moonshine.merged_decoder = decoder;
                
            },
            ASRVariant::Whisper => {
                config.model_config.whisper.encoder = encoder;
                config.model_config.whisper.decoder = decoder;
                // config.model_config.whisper.language = self.language;
                // config.model_config.whisper.task = self.task;
                // config.model_config.whisper.tail_paddings = self.tail_paddings;
                // config.model_config.whisper.enable_token_timestamps = self.enable_token_timestamps;
                // config.model_config.whisper.enable_segment_timestamps = self.enable_segment_timestamps;
       
            },
            ASRVariant::Unknown => {
                return Err("ASR model is not recognized".to_string());
            }
        }

        

        if let Some(recognizer) = OfflineRecognizer::create(&config)  {
            Ok(recognizer)
        } else {
            Err("Failed to create recognizer from config".to_string())
        }

    }

    pub fn transcribe(
        offline_recognizer: Result<OfflineRecognizer, String>,
        file_path: &str,
    ) -> Result<String, String> {
        let recognizer = offline_recognizer?;
        let stream = recognizer.create_stream();

        let wave = Wave::read(file_path)
            .ok_or_else(|| format!("Failed to read wave path: {file_path}"))?;
        stream.accept_waveform(wave.sample_rate(), wave.samples());
        recognizer.decode(&stream);

        if let Some(result) = stream.get_result() {
            Ok(result.text)
        } else {
            Err("Failed to get transcription result".to_string())
        }
    }



}
