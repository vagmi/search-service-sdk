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
    /// What embedding this query cost, when the query embedded anything.
    ///
    /// `None` covers two different situations and the caller cannot tell them
    /// apart: a BM25-only query, which genuinely embedded nothing, and a server
    /// too old to report usage. Both are safely treated as "nothing to bill" —
    /// under-reporting a cost is recoverable, inventing one is not.
    #[serde(default)]
    pub usage: Option<EmbedUsage>,
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

/// What a provider billed for the embedding done on a request's behalf.
///
/// The service is tenant-blind — it knows index names, not who owns them — so it
/// reports the meter reading and leaves attribution to the caller, which is the
/// only party that knows whose document this was.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct EmbedUsage {
    /// Provider that served the call, e.g. `"gemini"`.
    pub provider: String,
    /// Model id, e.g. `"gemini-embedding-2"`.
    pub model: String,
    /// Input tokens billed.
    #[serde(default)]
    pub prompt_tokens: u32,
    /// Total tokens as the provider reported them.
    #[serde(default)]
    pub total_tokens: u32,
    /// Provider round-trips. Lets a caller sanity-check batching — 400 chunks
    /// should be a dozen or so requests, not 400.
    #[serde(default)]
    pub requests: u32,
}

/// The acknowledgement returned by an indexing write.
#[derive(Debug, Clone, Deserialize)]
pub struct IndexAck {
    #[serde(rename = "_index", default)]
    pub index: String,
    #[serde(rename = "_id", default)]
    pub id: String,
    /// What embedding the write cost. `None` when the document had no `_embed`
    /// field, or when the server predates usage reporting.
    #[serde(default)]
    pub usage: Option<EmbedUsage>,
}

/// Internal: `GET /_indices` envelope.
#[derive(Debug, Deserialize)]
pub(crate) struct IndicesList {
    pub indices: Vec<crate::IndexInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Usage reporting is newer than the SDK's oldest deployed server. A client
    /// that refused to parse a response without it would take the whole service
    /// down during a rollout, which is a far worse outcome than an unbilled
    /// document.
    #[test]
    fn an_ack_from_a_server_without_usage_still_parses() {
        let ack: IndexAck = serde_json::from_str(
            r#"{"_index":"docs_abc","_id":"doc-1","result":"indexed"}"#,
        )
        .expect("older servers must stay readable");
        assert_eq!(ack.id, "doc-1");
        assert!(ack.usage.is_none());
    }

    #[test]
    fn an_ack_with_usage_parses_it() {
        let ack: IndexAck = serde_json::from_str(
            r#"{"_index":"docs_abc","_id":"doc-1","result":"indexed",
                "usage":{"provider":"gemini","model":"gemini-embedding-2",
                         "prompt_tokens":48120,"total_tokens":48120,"requests":13}}"#,
        )
        .expect("parses");
        let usage = ack.usage.expect("present");
        assert_eq!(usage.prompt_tokens, 48_120);
        assert_eq!(usage.requests, 13);
        assert_eq!(usage.model, "gemini-embedding-2");
    }

    /// A BM25-only search embeds nothing, so the server omits the field.
    #[test]
    fn a_search_response_without_usage_still_parses() {
        let resp: SearchResponse = serde_json::from_str(
            r#"{"took":3,"hits":{"total":{"value":0},"hits":[]}}"#,
        )
        .expect("parses");
        assert!(resp.usage.is_none());
    }
}
