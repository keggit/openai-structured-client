use reqwest::Client;
use schemars::schema_for;
use schemars::JsonSchema;
use serde::de::{self, DeserializeOwned, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;
use serde_json;
use serde_json::{json, Value};
use std::any::type_name;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

#[derive(Clone)]
pub struct OpenAiClient {
    http_client: Client,
    endpoint: String,
    model: String,
    api_key: String,
    system_role: Option<String>,
}

impl OpenAiClient {
    pub fn new(
        http_client: Client,
        endpoint: impl Into<String>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            http_client,
            endpoint: endpoint.into(),
            model: model.into(),
            api_key: api_key.into(),
            system_role: None,
        }
    }

    pub fn with_system_role(mut self, role: impl Into<String>) -> Self {
        self.system_role = Some(role.into());
        self
    }

    fn schema_name_for_type<T>() -> String {
        // Get the full type name, e.g. "my_crate::Review"
        let full_type_name = type_name::<T>();
        // Extract the last segment after "::"
        let type_name = full_type_name.rsplit("::").next().unwrap_or(full_type_name);
        // Convert to lowercase and append "_response"
        // Adjust case formatting if desired.
        format!("{}_response", type_name.to_lowercase())
    }

    fn generate_schema<T: JsonSchema>() -> Result<Value, Box<dyn std::error::Error>> {
        let schema = schema_for!(T);
        let mut schema_value = serde_json::to_value(&schema.schema)?;

        if let Value::Object(ref mut obj) = schema_value {
            // Ensure additionalProperties is false
            obj.insert("additionalProperties".to_string(), Value::Bool(false));

            // Ensure all properties are required
            if let Some(Value::Object(props)) = obj.get("properties") {
                let all_keys: Vec<String> = props.keys().cloned().collect();
                obj.insert(
                    "required".to_string(),
                    Value::Array(all_keys.into_iter().map(Value::String).collect()),
                );
            } else {
                // If no properties found, just ensure required is an empty array
                obj.insert("required".to_string(), Value::Array(vec![]));
            }
        }

        Ok(schema_value)
    }

    pub async fn call_schema<T: DeserializeOwned + JsonSchema + Clone>(
        &self,
        user_prompt: &str,
    ) -> Result<T, Box<dyn std::error::Error>> {
        let schema_value = Self::generate_schema::<T>()?;
        let schema_name = Self::schema_name_for_type::<T>();

        // Construct messages
        let mut messages = Vec::new();
        if let Some(system_content) = &self.system_role {
            messages.push(json!({
                "role": "system",
                "content": system_content
            }));
        }
        messages.push(json!({
            "role": "user",
            "content": user_prompt
        }));

        let body = json!({
            "model": self.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": true,
                    "schema": schema_value
                }
            }
        });

        let res = self
            .http_client
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let response: OpenAIResponse<T> = res.json().await?;

        match response {
            OpenAIResponse::Ok(res) => match res.choices[0].message.clone() {
                Message::Ok(content) => Ok(content.content),
                Message::Err(refusal) => Err(Box::new(refusal)),
            },
            OpenAIResponse::Err(err) => Err(Box::new(err)),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
#[serde(bound(deserialize = "T: DeserializeOwned"))]
pub enum OpenAIResponse<T> {
    Ok(ChatGPTResponse<T>),
    Err(OpenAIError),
}

#[derive(Debug, Deserialize)]
pub struct OpenAIError {
    error: OpenAIErrorDetails,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIErrorDetails {
    pub message: String,
    // pub r#type: String,
    // pub param: Option<String>,
    // pub code: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(bound(deserialize = "T: DeserializeOwned"))]
pub struct ChatGPTResponse<T> {
    pub choices: Vec<Choice<T>>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(bound(deserialize = "T: DeserializeOwned"))]
pub struct Choice<T> {
    pub message: Message<T>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
#[serde(bound(deserialize = "T: DeserializeOwned"))]
pub enum Message<T> {
    Ok(Content<T>),
    Err(Refusal),
}

#[derive(Debug, Deserialize, Clone)]
#[serde(bound(deserialize = "T: DeserializeOwned"))]
pub struct Content<T> {
    // pub role: String,
    #[serde(deserialize_with = "deserialize_content")]
    pub content: T,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Refusal {
    // role: String,
    refusal: String,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LLM refusal: {}", self.refusal)
    }
}

impl Error for Refusal {}

impl fmt::Display for OpenAIError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OpenAI Error: {}", self.error.message)
    }
}

impl Error for OpenAIError {}

fn deserialize_content<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: DeserializeOwned, // Ensure T owns all its data
    D: Deserializer<'de>,
{
    struct ContentVisitor<T>(PhantomData<T>);

    impl<'de, T> Visitor<'de> for ContentVisitor<T>
    where
        T: DeserializeOwned,
    {
        type Value = T;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string containing JSON data or an object")
        }

        fn visit_string<E>(self, v: String) -> Result<T, E>
        where
            E: de::Error,
        {
            // v is a String containing JSON data
            serde_json::from_str(&v).map_err(E::custom)
        }

        fn visit_str<E>(self, v: &str) -> Result<T, E>
        where
            E: de::Error,
        {
            // v is a &str containing JSON data
            serde_json::from_str(v).map_err(E::custom)
        }

        fn visit_map<M>(self, map: M) -> Result<T, M::Error>
        where
            M: MapAccess<'de>,
        {
            // Deserialize the map directly into T
            T::deserialize(de::value::MapAccessDeserializer::new(map))
        }

        fn visit_seq<A>(self, seq: A) -> Result<T, A::Error>
        where
            A: SeqAccess<'de>,
        {
            // Deserialize the sequence directly into T
            T::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }
    }

    deserializer.deserialize_any(ContentVisitor(PhantomData))
}
