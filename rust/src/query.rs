//! Search request types. These serialize to exactly the JSON the service expects.

use std::collections::BTreeMap;

use serde::Serialize;

/// A `_search` request body.
#[derive(Debug, Clone, Serialize)]
pub struct SearchRequest {
    pub query: QueryClause,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<i64>,
}

impl SearchRequest {
    pub fn new(query: QueryClause) -> Self {
        Self {
            query,
            size: None,
            from: None,
        }
    }

    /// Page size (defaults to the service default, 10, when unset).
    pub fn size(mut self, size: i64) -> Self {
        self.size = Some(size);
        self
    }

    /// Offset (defaults to 0 when unset).
    pub fn from(mut self, from: i64) -> Self {
        self.from = Some(from);
        self
    }
}

/// A query clause. Serializes externally-tagged, e.g. `{ "match": { "body": "..." } }`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryClause {
    /// Single text field, document-level BM25.
    Match(BTreeMap<String, String>),
    /// Several text fields, document-level BM25.
    MultiMatch(MultiMatch),
    /// Vector (kNN) over a vector field's chunks.
    Knn(VectorQuery),
    /// BM25-on-chunk-content ⊕ vector, fused by RRF.
    Hybrid(VectorQuery),
}

impl QueryClause {
    /// `{ "match": { field: query } }`.
    pub fn match_field(field: &str, query: &str) -> Self {
        let mut map = BTreeMap::new();
        map.insert(field.to_string(), query.to_string());
        QueryClause::Match(map)
    }

    /// `{ "multi_match": { "query": ..., "fields": [...] } }`.
    pub fn multi_match<I, S>(fields: I, query: &str) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        QueryClause::MultiMatch(MultiMatch {
            query: query.to_string(),
            fields: fields.into_iter().map(Into::into).collect(),
        })
    }

    /// `{ "knn": { "field": ..., "query": ... } }`.
    pub fn knn(field: &str, query: &str) -> Self {
        QueryClause::Knn(VectorQuery::new(field, query))
    }

    /// `{ "hybrid": { "field": ..., "query": ... } }`.
    pub fn hybrid(field: &str, query: &str) -> Self {
        QueryClause::Hybrid(VectorQuery::new(field, query))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MultiMatch {
    pub query: String,
    pub fields: Vec<String>,
}

/// A vector field query, used by both `knn` and `hybrid`.
#[derive(Debug, Clone, Serialize)]
pub struct VectorQuery {
    pub field: String,
    pub query: String,
    /// Candidate depth per modality (defaults to `size`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k: Option<i64>,
    /// RRF constant for `hybrid` (defaults to 60).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rrf_k: Option<f64>,
}

impl VectorQuery {
    pub fn new(field: &str, query: &str) -> Self {
        Self {
            field: field.to_string(),
            query: query.to_string(),
            k: None,
            rrf_k: None,
        }
    }

    pub fn k(mut self, k: i64) -> Self {
        self.k = Some(k);
        self
    }

    pub fn rrf_k(mut self, rrf_k: f64) -> Self {
        self.rrf_k = Some(rrf_k);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn match_serializes_externally_tagged() {
        let req = SearchRequest::new(QueryClause::match_field("body", "bm25 ranking"));
        assert_eq!(
            serde_json::to_value(&req).unwrap(),
            json!({ "query": { "match": { "body": "bm25 ranking" } } })
        );
    }

    #[test]
    fn multi_match_and_paging() {
        let req = SearchRequest::new(QueryClause::multi_match(["title", "body"], "hybrid"))
            .size(5)
            .from(10);
        assert_eq!(
            serde_json::to_value(&req).unwrap(),
            json!({
                "query": { "multi_match": { "query": "hybrid", "fields": ["title", "body"] } },
                "size": 5, "from": 10
            })
        );
    }

    #[test]
    fn knn_and_hybrid() {
        let knn = SearchRequest::new(QueryClause::knn("embedding", "rank fusion"));
        assert_eq!(
            serde_json::to_value(&knn).unwrap(),
            json!({ "query": { "knn": { "field": "embedding", "query": "rank fusion" } } })
        );
        let hybrid = SearchRequest::new(QueryClause::Hybrid(
            VectorQuery::new("embedding", "x").k(50).rrf_k(60.0),
        ));
        assert_eq!(
            serde_json::to_value(&hybrid).unwrap(),
            json!({ "query": { "hybrid": { "field": "embedding", "query": "x", "k": 50, "rrf_k": 60.0 } } })
        );
    }
}
