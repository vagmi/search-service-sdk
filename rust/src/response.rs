//! Search and document response types.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

/// A `_search` response.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchResponse {
    pub took: u128,
    pub hits: Hits,
    /// Facet results, keyed by aggregation name. Present only when the request had
    /// `aggs`. For `match`/`multi_match`/`bool` the counts reflect the full matching
    /// set (respecting the query and its filters, but not `post_filter`).
    #[serde(default)]
    pub aggregations: Option<BTreeMap<String, AggResult>>,
}

impl SearchResponse {
    /// The buckets of a named aggregation, if present.
    pub fn agg(&self, name: &str) -> Option<&AggResult> {
        self.aggregations.as_ref().and_then(|a| a.get(name))
    }
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
    /// The relevance score, or `None` for filter-only matches (a `bool` query with
    /// no scoring `must`), which are ordered by recency rather than relevance.
    #[serde(rename = "_score", default)]
    pub score: Option<f64>,
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

/// The result of one aggregation: its buckets, ordered as the service returned them
/// (`terms` by descending count, `range` in request order).
#[derive(Debug, Clone, Deserialize)]
pub struct AggResult {
    pub buckets: Vec<Bucket>,
}

/// A single facet bucket. `key` is a string for `terms`/`range` (a derived or
/// explicit label) or a scalar for numeric/boolean terms; `from`/`to` are set on
/// `range` buckets.
#[derive(Debug, Clone, Deserialize)]
pub struct Bucket {
    pub key: Value,
    pub doc_count: i64,
    #[serde(default)]
    pub from: Option<f64>,
    #[serde(default)]
    pub to: Option<f64>,
}

impl Bucket {
    /// The bucket key as a string, if it is one (e.g. a keyword/range label).
    pub fn key_str(&self) -> Option<&str> {
        self.key.as_str()
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
