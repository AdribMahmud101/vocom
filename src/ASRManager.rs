
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig};

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

    pub fn get_variant(file_name: &str)-> &Self {

        let file = file_name.to_lowercase();

        if file.contains("moonshine"){
            &Self::Moonshinev2 
        } else if file.contains("whisper"){
            &Self::Whisper
        } else{
            &Self::Unknown
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
            language: None,
            task: None,
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
        let encoder = self.encoder.ok_or("Encoder path is missing");
        let decoder = self.decoder.ok_or("Decoder path is missing");
        let tokens = self.tokens.ok_or("Tokens path is missing"); // returns Result<String, &str>

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.tokens = tokens.ok();
   
        match self.variant {
            ASRVariant::Moonshinev2 => {
                config.model_config.moonshine.encoder = encoder.ok();
                config.model_config.moonshine.merged_decoder = decoder.ok();
                
            },
            ASRVariant::Whisper => {
                config.model_config.whisper.encoder = encoder.ok();
                config.model_config.whisper.decoder = decoder.ok();
       
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


}



fn main() {


    println!("Hello, world!");
}
