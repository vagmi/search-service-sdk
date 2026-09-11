//! Search request types. These serialize to exactly the JSON the service expects.
//!
//! - [`QueryClause`] — the scoring query (`match`, `multi_match`, `knn`, `hybrid`,
//!   or a `bool` combinator).
//! - [`Filter`] — filter context (`term`, `terms`, `range`, `exists`, nested `bool`).
//!   Filters restrict which documents match but never affect the relevance score.
//! - [`Agg`] — aggregations (facets): `terms` and `range` bucket counts.

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;

/// A `_search` request body.
#[derive(Debug, Clone, Serialize)]
pub struct SearchRequest {
    pub query: QueryClause,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<i64>,
    /// Drop hits scoring below this. Only meaningful *per query type* — see
    /// [`crate::Hit::score`] for what `_score` holds in each case. A value tuned for
    /// `match` means nothing for `hybrid`, whose scores are bounded by
    /// `Σweights/(rrf_k + 1)` (0.0328 at the defaults).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_score: Option<f64>,
    /// Facets, computed over the full matching set (independent of `size`/`from`).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub aggs: BTreeMap<String, Agg>,
    /// Filters applied to the returned hits *after* aggregations are computed — so
    /// selecting a facet narrows results without collapsing that facet's own counts.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub post_filter: Vec<Filter>,
}

impl SearchRequest {
    pub fn new(query: impl Into<QueryClause>) -> Self {
        Self {
            query: query.into(),
            size: None,
            from: None,
            min_score: None,
            aggs: BTreeMap::new(),
            post_filter: Vec::new(),
        }
    }

    /// Page size (defaults to the service default, 10, when unset). Use `0` to
    /// return only aggregations.
    pub fn size(mut self, size: i64) -> Self {
        self.size = Some(size);
        self
    }

    /// Offset (defaults to 0 when unset).
    pub fn from(mut self, from: i64) -> Self {
        self.from = Some(from);
        self
    }

    /// Drop hits scoring below `min_score`. The threshold is in whatever units the
    /// query type's `_score` uses, so it is not portable between query types. For
    /// `knn`, prefer [`VectorQuery::min_similarity`], which the service pushes into
    /// SQL rather than applying to an already-fetched candidate window.
    pub fn min_score(mut self, min_score: f64) -> Self {
        self.min_score = Some(min_score);
        self
    }

    /// Add a named aggregation (facet).
    pub fn agg(mut self, name: &str, agg: impl Into<Agg>) -> Self {
        self.aggs.insert(name.to_string(), agg.into());
        self
    }

    /// Add a `post_filter` (applied to hits only, not to aggregations).
    pub fn post_filter(mut self, filter: impl Into<Filter>) -> Self {
        self.post_filter.push(filter.into());
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
    /// A scoring `must` (text only) restricted by filter context.
    Bool(BoolQuery),
}

impl QueryClause {
    /// `{ "match": { field: query } }`.
    pub fn match_field(field: &str, query: &str) -> Self {
        QueryClause::Match(one(field, query.to_string()))
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

    /// `{ "knn": { "field": ..., "query": ... } }`. For a filtered kNN, build a
    /// [`VectorQuery`] directly: `QueryClause::Knn(VectorQuery::new(..).filter(..))`.
    pub fn knn(field: &str, query: &str) -> Self {
        QueryClause::Knn(VectorQuery::new(field, query))
    }

    /// `{ "hybrid": { "field": ..., "query": ... } }`.
    pub fn hybrid(field: &str, query: &str) -> Self {
        QueryClause::Hybrid(VectorQuery::new(field, query))
    }

    /// Start a `bool` query: `QueryClause::bool().must(..).filter(..)`.
    pub fn bool() -> BoolQuery {
        BoolQuery::default()
    }
}

impl From<BoolQuery> for QueryClause {
    fn from(b: BoolQuery) -> Self {
        QueryClause::Bool(b)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MultiMatch {
    pub query: String,
    pub fields: Vec<String>,
}

/// A vector field query, used by both `knn` and `hybrid`. An optional `filter`
/// restricts the search to documents matching the given filter context (ES
/// `knn.filter`): the service pre-filters to the matching parent ids before the
/// ANN scan over chunks.
#[derive(Debug, Clone, Serialize)]
pub struct VectorQuery {
    pub field: String,
    pub query: String,
    /// Candidate *depth* — how many chunks each modality retrieves before fusion
    /// and paging. `size` truncates the page; `k` does not set it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k: Option<i64>,
    /// RRF constant for `hybrid` (defaults to 60).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rrf_k: Option<f64>,
    /// Cosine-similarity floor in `[-1, 1]`. Unset ⇒ no floor, which is the ANN
    /// default: nearest neighbours always exist, so an unrelated query still comes
    /// back with a full page. Setting it gates the vector leg the way BM25 already
    /// is, letting "nothing is close enough" return no hits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_similarity: Option<f64>,
    /// Relative trust in each `hybrid` leg. Unset ⇒ an even 1.0/1.0 fusion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights: Option<HybridWeights>,
    /// Keep only the best-scoring chunk per parent document, so one long document
    /// cannot fill the page with its own chunks.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub collapse: bool,
    /// Filter context on the parent document's fields.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub filter: Vec<Filter>,
}

/// Per-leg weights for `hybrid` fusion. Unweighted RRF locks lexical and semantic
/// evidence at 50/50; raising `bm25` biases toward chunks where the query terms
/// genuinely occur. Both must be non-negative and not both zero, or the service
/// returns a 400.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct HybridWeights {
    pub vector: f64,
    pub bm25: f64,
}

impl HybridWeights {
    pub fn new(vector: f64, bm25: f64) -> Self {
        Self { vector, bm25 }
    }
}

impl Default for HybridWeights {
    fn default() -> Self {
        Self {
            vector: 1.0,
            bm25: 1.0,
        }
    }
}

impl VectorQuery {
    pub fn new(field: &str, query: &str) -> Self {
        Self {
            field: field.to_string(),
            query: query.to_string(),
            k: None,
            rrf_k: None,
            min_similarity: None,
            weights: None,
            collapse: false,
            filter: Vec::new(),
        }
    }

    /// Candidate depth per modality. This is *not* the page size — set that with
    /// [`SearchRequest::size`], which truncates the fused page.
    pub fn k(mut self, k: i64) -> Self {
        self.k = Some(k);
        self
    }

    pub fn rrf_k(mut self, rrf_k: f64) -> Self {
        self.rrf_k = Some(rrf_k);
        self
    }

    /// Require at least this cosine similarity (in `[-1, 1]`) from the vector leg.
    /// Without it an ANN scan returns its whole window for any query at all.
    pub fn min_similarity(mut self, min_similarity: f64) -> Self {
        self.min_similarity = Some(min_similarity);
        self
    }

    /// Weight the two `hybrid` legs relative to each other (no effect on `knn`).
    pub fn weights(mut self, vector: f64, bm25: f64) -> Self {
        self.weights = Some(HybridWeights::new(vector, bm25));
        self
    }

    /// Return only the best-scoring chunk per parent document.
    pub fn collapse(mut self, collapse: bool) -> Self {
        self.collapse = collapse;
        self
    }

    /// Restrict the vector search to documents matching this filter.
    pub fn filter(mut self, filter: impl Into<Filter>) -> Self {
        self.filter.push(filter.into());
        self
    }
}

/// A `bool` query: an optional scoring `must` (text only) plus filter context.
/// `filter`/`must_not` never affect the score; they only restrict which documents
/// match. With no `must`, this is a filtered `match_all` (hits carry no score).
#[derive(Debug, Clone, Default, Serialize)]
pub struct BoolQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub must: Option<Box<QueryClause>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub filter: Vec<Filter>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub must_not: Vec<Filter>,
}

impl BoolQuery {
    /// Set the scoring query (`match` / `multi_match`).
    pub fn must(mut self, query: impl Into<QueryClause>) -> Self {
        self.must = Some(Box::new(query.into()));
        self
    }

    /// Add a filter that documents must match (does not affect score).
    pub fn filter(mut self, filter: impl Into<Filter>) -> Self {
        self.filter.push(filter.into());
        self
    }

    /// Add a filter that documents must NOT match.
    pub fn must_not(mut self, filter: impl Into<Filter>) -> Self {
        self.must_not.push(filter.into());
        self
    }
}

// --- filter context -------------------------------------------------------

/// A filter clause. Filters restrict matches but never contribute to the score.
/// Serializes externally-tagged, e.g. `{ "term": { "tags": "rust" } }`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Filter {
    /// `{ "term": { field: value } }` — exact match (keyword membership, or `=`).
    Term(BTreeMap<String, Value>),
    /// `{ "terms": { field: [v1, v2] } }` — match any of the values.
    Terms(BTreeMap<String, Vec<Value>>),
    /// `{ "range": { field: { "gte": .., "lt": .. } } }` — numeric/date bounds.
    Range(BTreeMap<String, RangeBounds>),
    /// `{ "exists": { "field": name } }` — the field is present (non-null/non-empty).
    Exists(ExistsClause),
    /// `{ "bool": { .. } }` — nested boolean combinator.
    Bool(BoolFilter),
}

impl Filter {
    /// `term` — exact match. For a keyword field, matches documents containing the
    /// value; for a scalar/date field, an equality match.
    pub fn term(field: &str, value: impl Into<Value>) -> Self {
        Filter::Term(one(field, value.into()))
    }

    /// `terms` — match any of the values.
    pub fn terms<I, V>(field: &str, values: I) -> Self
    where
        I: IntoIterator<Item = V>,
        V: Into<Value>,
    {
        Filter::Terms(one(field, values.into_iter().map(Into::into).collect()))
    }

    /// `range` — build numeric/date bounds: `Filter::range("views").gte(100).lt(1000)`.
    pub fn range(field: &str) -> RangeFilter {
        RangeFilter::new(field)
    }

    /// `exists` — the field has a value.
    pub fn exists(field: &str) -> Self {
        Filter::Exists(ExistsClause {
            field: field.to_string(),
        })
    }

    /// Start a nested `bool` filter: `Filter::bool().should(..).must_not(..)`.
    pub fn bool() -> BoolFilter {
        BoolFilter::default()
    }
}

/// Builder for a `range` filter. Any subset of bounds may be set; `[from, to)`
/// semantics match the service (`gte`/`gt` lower, `lte`/`lt` upper).
#[derive(Debug, Clone)]
pub struct RangeFilter {
    field: String,
    bounds: RangeBounds,
}

impl RangeFilter {
    fn new(field: &str) -> Self {
        Self {
            field: field.to_string(),
            bounds: RangeBounds::default(),
        }
    }

    pub fn gte(mut self, v: impl Into<Value>) -> Self {
        self.bounds.gte = Some(v.into());
        self
    }
    pub fn gt(mut self, v: impl Into<Value>) -> Self {
        self.bounds.gt = Some(v.into());
        self
    }
    pub fn lte(mut self, v: impl Into<Value>) -> Self {
        self.bounds.lte = Some(v.into());
        self
    }
    pub fn lt(mut self, v: impl Into<Value>) -> Self {
        self.bounds.lt = Some(v.into());
        self
    }
}

impl From<RangeFilter> for Filter {
    fn from(r: RangeFilter) -> Self {
        Filter::Range(one(&r.field, r.bounds))
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RangeBounds {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gte: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lte: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lt: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExistsClause {
    pub field: String,
}

/// Nested boolean filter combinator: `must`/`filter` AND, `should` OR, `must_not` NOT.
#[derive(Debug, Clone, Default, Serialize)]
pub struct BoolFilter {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub must: Vec<Filter>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub should: Vec<Filter>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub must_not: Vec<Filter>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub filter: Vec<Filter>,
}

impl BoolFilter {
    pub fn must(mut self, f: impl Into<Filter>) -> Self {
        self.must.push(f.into());
        self
    }
    pub fn should(mut self, f: impl Into<Filter>) -> Self {
        self.should.push(f.into());
        self
    }
    pub fn must_not(mut self, f: impl Into<Filter>) -> Self {
        self.must_not.push(f.into());
        self
    }
    pub fn filter(mut self, f: impl Into<Filter>) -> Self {
        self.filter.push(f.into());
        self
    }
}

impl From<BoolFilter> for Filter {
    fn from(b: BoolFilter) -> Self {
        Filter::Bool(b)
    }
}

// --- aggregations ---------------------------------------------------------

/// An aggregation (facet). Serializes externally-tagged, e.g.
/// `{ "terms": { "field": "tags", "size": 20 } }`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Agg {
    Terms(TermsAgg),
    Range(RangeAgg),
}

impl Agg {
    /// A `terms` facet over a keyword/numeric/boolean/date field.
    pub fn terms(field: &str) -> TermsAgg {
        TermsAgg {
            field: field.to_string(),
            size: None,
        }
    }

    /// A `range` facet over a numeric field: `Agg::range("views").below(100.0).above(100.0)`.
    pub fn range(field: &str) -> RangeAgg {
        RangeAgg {
            field: field.to_string(),
            ranges: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TermsAgg {
    pub field: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<i64>,
}

impl TermsAgg {
    /// Maximum number of buckets to return (top values by count).
    pub fn size(mut self, size: i64) -> Self {
        self.size = Some(size);
        self
    }
}

impl From<TermsAgg> for Agg {
    fn from(a: TermsAgg) -> Self {
        Agg::Terms(a)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RangeAgg {
    pub field: String,
    pub ranges: Vec<RangeBucketDef>,
}

impl RangeAgg {
    /// A bucket `[from, to)`; pass `None` for an open end.
    pub fn bucket(mut self, from: Option<f64>, to: Option<f64>) -> Self {
        self.ranges.push(RangeBucketDef {
            from,
            to,
            key: None,
        });
        self
    }

    /// A bucket for values below `to` (exclusive).
    pub fn below(self, to: f64) -> Self {
        self.bucket(None, Some(to))
    }

    /// A bucket for values in `[from, to)`.
    pub fn between(self, from: f64, to: f64) -> Self {
        self.bucket(Some(from), Some(to))
    }

    /// A bucket for values at or above `from`.
    pub fn above(self, from: f64) -> Self {
        self.bucket(Some(from), None)
    }
}

impl From<RangeAgg> for Agg {
    fn from(a: RangeAgg) -> Self {
        Agg::Range(a)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RangeBucketDef {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
}

/// Insert a single (field, value) pair into a fresh map.
fn one<V>(field: &str, value: V) -> BTreeMap<String, V> {
    let mut map = BTreeMap::new();
    map.insert(field.to_string(), value);
    map
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

    #[test]
    fn bool_query_with_filter_context() {
        let req = SearchRequest::new(
            QueryClause::bool()
                .must(QueryClause::match_field("body", "postgres"))
                .filter(Filter::terms("tags", ["rust", "db"]))
                .filter(Filter::range("views").gte(100))
                .must_not(Filter::term("published", false)),
        );
        assert_eq!(
            serde_json::to_value(&req).unwrap(),
            json!({ "query": { "bool": {
                "must": { "match": { "body": "postgres" } },
                "filter": [
                    { "terms": { "tags": ["rust", "db"] } },
                    { "range": { "views": { "gte": 100 } } }
                ],
                "must_not": [ { "term": { "published": false } } ]
            }}})
        );
    }

    #[test]
    fn filter_only_bool_is_match_all() {
        let req = SearchRequest::new(QueryClause::bool().filter(Filter::term("tags", "db")));
        assert_eq!(
            serde_json::to_value(&req).unwrap(),
            json!({ "query": { "bool": { "filter": [ { "term": { "tags": "db" } } ] } } })
        );
    }

    #[test]
    fn aggs_and_post_filter() {
        let req = SearchRequest::new(QueryClause::bool())
            .size(0)
            .agg("tags", Agg::terms("tags").size(20))
            .agg(
                "views",
                Agg::range("views")
                    .below(100.0)
                    .between(100.0, 1000.0)
                    .above(1000.0),
            )
            .post_filter(Filter::term("tags", "rust"));
        assert_eq!(
            serde_json::to_value(&req).unwrap(),
            json!({
                "query": { "bool": {} },
                "size": 0,
                "aggs": {
                    "tags": { "terms": { "field": "tags", "size": 20 } },
                    "views": { "range": { "field": "views", "ranges": [
                        { "to": 100.0 }, { "from": 100.0, "to": 1000.0 }, { "from": 1000.0 }
                    ] } }
                },
                "post_filter": [ { "term": { "tags": "rust" } } ]
            })
        );
    }

    #[test]
    fn knn_with_filter() {
        let req = SearchRequest::new(QueryClause::Knn(
            VectorQuery::new("embedding", "q")
                .k(20)
                .filter(Filter::term("tags", "db")),
        ));
        assert_eq!(
            serde_json::to_value(&req).unwrap(),
            json!({ "query": { "knn": {
                "field": "embedding", "query": "q", "k": 20,
                "filter": [ { "term": { "tags": "db" } } ]
            }}})
        );
    }

    #[test]
    fn scoring_controls_serialize() {
        let req = SearchRequest::new(QueryClause::Hybrid(
            VectorQuery::new("embedding", "x")
                .k(200)
                .min_similarity(0.6)
                .weights(1.0, 3.0)
                .collapse(true),
        ))
        .size(10)
        .min_score(0.02);
        assert_eq!(
            serde_json::to_value(&req).unwrap(),
            json!({
                "query": { "hybrid": {
                    "field": "embedding", "query": "x", "k": 200,
                    "min_similarity": 0.6,
                    "weights": { "vector": 1.0, "bm25": 3.0 },
                    "collapse": true
                }},
                "size": 10, "min_score": 0.02
            })
        );
    }

    /// The new knobs are opt-in: an untouched query must serialize exactly as it did
    /// before they existed, or every existing caller starts sending new fields.
    #[test]
    fn scoring_controls_are_omitted_when_unset() {
        let req = SearchRequest::new(QueryClause::knn("embedding", "rank fusion"));
        assert_eq!(
            serde_json::to_value(&req).unwrap(),
            json!({ "query": { "knn": { "field": "embedding", "query": "rank fusion" } } }),
            "collapse=false and the unset options must not appear in the body"
        );
    }

    #[test]
    fn nested_bool_filter() {
        let f: Filter = Filter::bool()
            .should(Filter::term("tags", "rust"))
            .should(Filter::term("tags", "db"))
            .must_not(Filter::exists("archived_at"))
            .into();
        assert_eq!(
            serde_json::to_value(&f).unwrap(),
            json!({ "bool": {
                "should": [ { "term": { "tags": "rust" } }, { "term": { "tags": "db" } } ],
                "must_not": [ { "exists": { "field": "archived_at" } } ]
            }})
        );
    }
}
