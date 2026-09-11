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
/// chunk hits (`knn` / `hybrid`) carry `chunk_id`, `seq`, `content`, and `rank`.
#[derive(Debug, Clone, Deserialize)]
pub struct Hit {
    #[serde(rename = "_id")]
    pub id: String,
    /// The relevance score — **a different quantity per query type**, and not
    /// comparable across them:
    ///
    /// | Query | `score` is | Range |
    /// | --- | --- | --- |
    /// | `match` / `multi_match` | summed BM25 over the queried fields | positive, unbounded |
    /// | `knn` | cosine similarity (also in [`HitScores::cosine`]) | `[-1, 1]` |
    /// | `hybrid` | RRF score, `Σ wᵢ/(rrf_k + rankᵢ)` | `(0, Σw/(rrf_k+1)]` — 0.0328 at the defaults |
    /// | filter-only `bool` | `None` — nothing scored it | — |
    ///
    /// For a chunk hit, prefer [`Hit::rank`] over this value when judging *why* the
    /// hit is here: RRF fuses rank into a scalar, so a chunk ranked first by BM25
    /// and one ranked first by the vector leg land on the identical score.
    #[serde(rename = "_score", default)]
    pub score: Option<f64>,
    /// Which retrieval legs produced this chunk hit, and where it placed in each.
    /// `None` for document hits, and for chunk hits from a server predating
    /// provenance reporting.
    #[serde(rename = "_rank", default)]
    pub rank: Option<HitRank>,
    /// Raw component scores, when the query computed any (`knn` reports `cosine`).
    #[serde(rename = "_scores", default)]
    pub scores: Option<HitScores>,
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

    /// Whether the BM25 leg matched this chunk — i.e. the query terms genuinely
    /// occur in it, rather than it merely being a near neighbour in vector space.
    ///
    /// This is the question a `hybrid` `_score` cannot answer on its own. `false`
    /// for a vector-only hit, a document hit, or a server that reports no `_rank`;
    /// use [`Hit::rank`] directly if you need to tell those cases apart.
    pub fn is_lexical_match(&self) -> bool {
        self.rank.as_ref().is_some_and(|r| r.bm25.is_some())
    }

    /// Cosine similarity for a `knn` hit, when the server reported it.
    pub fn cosine(&self) -> Option<f64> {
        self.scores.as_ref().and_then(|s| s.cosine)
    }
}

/// Where a chunk hit came from: its 1-based place in each retrieval leg, or `None`
/// where that leg did not produce it.
///
/// Reciprocal Rank Fusion sums `w/(k + rank)` across the legs, which collapses two
/// very different hits onto the same number — top of the BM25 list and top of the
/// vector list both fuse to `1/(k+1)`. These ranks are what separate them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
pub struct HitRank {
    /// Rank in the BM25 leg. `Some` only when the query terms occur in the chunk,
    /// since that leg is gated on a real lexical match.
    #[serde(default)]
    pub bm25: Option<usize>,
    /// Rank in the vector leg — the chunk's place in the ANN candidate window.
    /// An ANN scan has no notion of "no match", so this is populated for any query
    /// at all unless the query set a similarity floor.
    #[serde(default)]
    pub vector: Option<usize>,
}

/// Raw component scores behind a hit, in their own units.
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
pub struct HitScores {
    /// Cosine similarity in `[-1, 1]`, reported for `knn` hits.
    #[serde(default)]
    pub cosine: Option<f64>,
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
        let ack: IndexAck =
            serde_json::from_str(r#"{"_index":"docs_abc","_id":"doc-1","result":"indexed"}"#)
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

    /// Provenance separates the two hits an RRF score cannot tell apart: both fuse
    /// to 1/61, but only one is a real lexical match.
    #[test]
    fn rank_distinguishes_a_lexical_hit_from_vector_filler() {
        let resp: SearchResponse = serde_json::from_str(
            r#"{"took":4,"hits":{"total":{"value":2},"hits":[
                {"_id":"lex","chunk_id":1,"seq":0,"_score":0.016393,
                 "_rank":{"bm25":1,"vector":null},"content":"terms occur here"},
                {"_id":"nn","chunk_id":2,"seq":0,"_score":0.016393,
                 "_rank":{"bm25":null,"vector":1},"content":"merely nearby"}
            ]}}"#,
        )
        .expect("parses");
        let (lex, nn) = (&resp.hits.hits[0], &resp.hits.hits[1]);
        assert_eq!(lex.score, nn.score, "RRF gives both the identical score");
        assert!(lex.is_lexical_match(), "...but only one matched lexically");
        assert!(!nn.is_lexical_match());
        assert_eq!(lex.rank.unwrap().bm25, Some(1));
        assert_eq!(nn.rank.unwrap().vector, Some(1));
    }

    /// `knn` reports the raw cosine alongside the score.
    #[test]
    fn knn_hit_exposes_its_cosine() {
        let resp: SearchResponse = serde_json::from_str(
            r#"{"took":1,"hits":{"total":{"value":1},"hits":[
                {"_id":"d","chunk_id":7,"seq":2,"_score":0.83,
                 "_rank":{"bm25":null,"vector":null},"_scores":{"cosine":0.83},
                 "content":"a passage"}
            ]}}"#,
        )
        .expect("parses");
        assert_eq!(resp.hits.hits[0].cosine(), Some(0.83));
    }

    /// Provenance is newer than the SDK's oldest deployed server, and document hits
    /// never carry it. Neither may break parsing — same reasoning as usage above.
    #[test]
    fn a_hit_without_rank_still_parses() {
        let resp: SearchResponse = serde_json::from_str(
            r#"{"took":2,"hits":{"total":{"value":1},"hits":[
                {"_id":"doc-1","_score":12.5,"_source":{"title":"t"},"_meta":{}}
            ]}}"#,
        )
        .expect("older servers and document hits must stay readable");
        let hit = &resp.hits.hits[0];
        assert!(hit.rank.is_none());
        assert!(hit.scores.is_none());
        assert!(
            !hit.is_lexical_match(),
            "absent provenance is not a claim of a lexical match"
        );
        assert_eq!(hit.cosine(), None);
    }

    /// A BM25-only search embeds nothing, so the server omits the field.
    #[test]
    fn a_search_response_without_usage_still_parses() {
        let resp: SearchResponse =
            serde_json::from_str(r#"{"took":3,"hits":{"total":{"value":0},"hits":[]}}"#)
                .expect("parses");
        assert!(resp.usage.is_none());
    }
}
