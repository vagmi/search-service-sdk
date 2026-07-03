//! Search and document response types.

use serde::Deserialize;
use serde_json::Value;

/// A `_search` response.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchResponse {
    pub took: u128,
    pub hits: Hits,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Hits {
    pub total: Total,
    pub hits: Vec<Hit>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Total {
    pub value: usize,
}

/// A single hit. Document hits (`match` / `multi_match`) carry `source`/`meta`;
/// chunk hits (`knn` / `hybrid`) carry `chunk_id`, `seq`, and `content`.
#[derive(Debug, Clone, Deserialize)]
pub struct Hit {
    #[serde(rename = "_id")]
    pub id: String,
    #[serde(rename = "_score")]
    pub score: f64,
    #[serde(rename = "_source", default)]
    pub source: Option<Value>,
    #[serde(rename = "_meta", default)]
    pub meta: Option<Value>,
    #[serde(default)]
    pub chunk_id: Option<i64>,
    #[serde(default)]
    pub seq: Option<i32>,
    #[serde(default)]
    pub content: Option<String>,
}

impl Hit {
    /// Deserialize a document hit's `_source` into a typed value.
    pub fn source_as<T: serde::de::DeserializeOwned>(&self) -> Option<crate::Result<T>> {
        self.source
            .clone()
            .map(|v| serde_json::from_value(v).map_err(Into::into))
    }
}

/// A document fetched via `GET /{index}/_doc/{id}`.
#[derive(Debug, Clone, Deserialize)]
pub struct Document {
    #[serde(rename = "_id")]
    pub id: String,
    #[serde(default)]
    pub found: bool,
    #[serde(rename = "_source", default)]
    pub source: Value,
    #[serde(rename = "_meta", default)]
    pub meta: Value,
}

impl Document {
    /// Deserialize `_source` into a typed value.
    pub fn source_as<T: serde::de::DeserializeOwned>(&self) -> crate::Result<T> {
        serde_json::from_value(self.source.clone()).map_err(Into::into)
    }
}

/// Internal: the `{ "_id": ... }` shape returned when creating a document.
#[derive(Debug, Deserialize)]
pub(crate) struct DocAck {
    #[serde(rename = "_id")]
    pub id: String,
}

/// Internal: `GET /_indices` envelope.
#[derive(Debug, Deserialize)]
pub(crate) struct IndicesList {
    pub indices: Vec<crate::IndexInfo>,
}
