use regex::Regex;
use reqwest::Client;
use schemars::schema::{RootSchema, Schema, SchemaObject};
use schemars::schema_for;
use schemars::JsonSchema;
use serde::de::{self, DeserializeOwned, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;
use serde_json;
use serde_json::{json, Value};
use std::any::type_name;
use std::collections::BTreeSet;
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
        let full_type_name = type_name::<T>();

        // Replace anything not in [a-zA-Z0-9_-] with underscores.
        let re = Regex::new("[^a-zA-Z0-9_-]+").unwrap();
        let sanitized = re.replace_all(full_type_name, "_").to_string();

        format!("{}_response", sanitized.to_lowercase())
    }

    /// Generates a JSON schema for T, ensuring additionalProperties=false
    /// for all nested object types.
    fn generate_schema<T: JsonSchema>() -> Result<Value, Box<dyn Error>> {
        Self::generate_schema_with_no_additional::<T>()
    }

    fn generate_schema_with_no_additional<T: JsonSchema>() -> Result<Value, Box<dyn Error>> {
        let mut root_schema: RootSchema = schema_for!(T);

        // 1. Update the top-level schema
        Self::set_no_additional_properties(&mut root_schema.schema);

        // 2. Update each schema in "definitions"
        for (_name, def_schema) in root_schema.definitions.iter_mut() {
            if let Schema::Object(ref mut obj) = def_schema {
                Self::set_no_additional_properties(obj);
            }
        }

        // 3. Convert RootSchema => JSON
        let schema_value = serde_json::to_value(&root_schema)?;
        Ok(schema_value)
    }

    fn set_no_additional_properties(schema_obj: &mut SchemaObject) {
        // 1. If this schema is "type: object", set additionalProperties = false
        //    and force all properties to appear in the 'required' list.
        if let Some(schemars::schema::SingleOrVec::Single(ref t)) = schema_obj.instance_type {
            if **t == schemars::schema::InstanceType::Object {
                // If there's no ObjectValidation yet, create one
                let ov = schema_obj
                    .object
                    .get_or_insert_with(|| Box::new(schemars::schema::ObjectValidation::default()));

                // Disallow unknown fields
                ov.additional_properties = Some(Box::new(schemars::schema::Schema::Bool(false)));

                let prop_names: BTreeSet<String> = ov.properties.keys().cloned().collect();
                ov.required = prop_names;
            }
        }

        // 2. Recurse into each property
        if let Some(ref mut box_obj_validation) = schema_obj.object {
            // Now that we have forced them to be required, also
            // descend into each property’s schema if it's an object.
            for (_prop_name, prop_schema) in box_obj_validation.properties.iter_mut() {
                if let schemars::schema::Schema::Object(ref mut nested_obj) = prop_schema {
                    Self::set_no_additional_properties(nested_obj);
                }
            }
        }

        // 3. If it's an array, handle items
        if let Some(ref mut array_box) = schema_obj.array {
            // array_box is Box<ArrayValidation>
            if let Some(items_schema) = &mut array_box.items {
                match items_schema {
                    // single schema
                    schemars::schema::SingleOrVec::Single(boxed_schema) => {
                        if let schemars::schema::Schema::Object(ref mut nested_obj) = **boxed_schema
                        {
                            Self::set_no_additional_properties(nested_obj);
                        }
                    }
                    // tuple variant
                    schemars::schema::SingleOrVec::Vec(ref mut schemas) => {
                        for schema in schemas.iter_mut() {
                            if let schemars::schema::Schema::Object(ref mut nested_obj) = schema {
                                Self::set_no_additional_properties(nested_obj);
                            }
                        }
                    }
                }
            }
        }

        // 4. Check "subschemas" (allOf, anyOf, oneOf, not)
        if let Some(ref mut subs) = schema_obj.subschemas {
            // allOf
            if let Some(ref mut all_of_vec) = subs.all_of {
                for sub_schema in all_of_vec.iter_mut() {
                    if let schemars::schema::Schema::Object(ref mut nested_obj) = sub_schema {
                        Self::set_no_additional_properties(nested_obj);
                    }
                }
            }
            // anyOf
            if let Some(ref mut any_of_vec) = subs.any_of {
                for sub_schema in any_of_vec.iter_mut() {
                    if let schemars::schema::Schema::Object(ref mut nested_obj) = sub_schema {
                        Self::set_no_additional_properties(nested_obj);
                    }
                }
            }
            // oneOf
            if let Some(ref mut one_of_vec) = subs.one_of {
                for sub_schema in one_of_vec.iter_mut() {
                    if let schemars::schema::Schema::Object(ref mut nested_obj) = sub_schema {
                        Self::set_no_additional_properties(nested_obj);
                    }
                }
            }
            // not
            if let Some(ref mut not_box) = subs.not {
                if let schemars::schema::Schema::Object(ref mut nested_obj) = **not_box {
                    Self::set_no_additional_properties(nested_obj);
                }
            }
        }
    }

    /// Calls the OpenAI endpoint, passing the JSON schema in 'response_format.json_schema.schema'.
    /// Expects a typed response conforming to T.
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

        // Build the request body
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
