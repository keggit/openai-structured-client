mod openai;

use crate::openai::OpenAiClient;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[tokio::main]
async fn main() {
    let http_client = reqwest::Client::new();
    let endpoint = "https://api.openai.com/v1/chat/completions";
    let model = "gpt-4o-2024-08-06";
    let api_key = "apikey";

    let openai = OpenAiClient::new(http_client, endpoint, model, api_key)
        .with_system_role("You are a helpful English tutor.");

    let prompt = "Explain the errors in the following sentences... yada yada";
    let review: Review = openai
        .call_schema(prompt)
        .await
        .expect("openai responded and parsed into Review");
    println!("{:#?}", review);
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
pub struct Review {
    #[schemars(description = "An explanation of the incorrect vocabulary and grammar points.")]
    pub explanation: String,
    pub incorrect_words: Option<Vec<String>>,
    pub incorrect_grammars: Option<Vec<String>>,
}
