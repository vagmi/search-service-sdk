# search_service (Rust SDK)

Async Rust client for the [search-service](https://github.com/vagmi/search-service)
HTTP API — BM25 full-text plus vector / hybrid search, loosely Elasticsearch-shaped.

## Install

```toml
[dependencies]
search_service = { path = "../search-service-sdk/rust" } # or a git/version dep
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
serde_json = "1"
```

## Quickstart

```rust
use search_service::{Client, Schema, FieldType, SearchRequest, QueryClause};

#[tokio::main]
async fn main() -> search_service::Result<()> {
    // Point at wherever the router is mounted (root, or a prefix like /search).
    let client = Client::new("http://localhost:3000/search")?;

    // Create an index.
    let schema = Schema::builder()
        .text("title", Some("english"))
        .text("body", None)
        .scalar("views", FieldType::Integer)
        .vector("embedding", "gemini", "gemini-embedding-2", 1536, Some("retrieval"))
        .build();
    client.create_index("posts", &schema).await?;

    // Index a document: `_meta` is stored/returned; `_embed` chunks are embedded.
    client.index_document("posts", "p1", &serde_json::json!({
        "title": "Hybrid search", "body": "BM25 plus vectors", "views": 10,
        "_meta": { "slug": "hybrid" },
        "_embed": { "embedding": ["a passage to embed", "another passage"] }
    })).await?;

    // Search (match / multi_match → documents; knn / hybrid → chunks).
    let res = client.search("posts",
        &SearchRequest::new(QueryClause::hybrid("embedding", "retire elasticsearch"))).await?;
    for hit in res.hits.hits {
        // chunk hits carry chunk_id / seq / content; document hits carry source / meta.
        // `score` is None for filter-only queries (a `bool` with no scoring `must`),
        // and means something different per query type — see "Scores and ranking".
        // `is_lexical_match()` says whether the query terms actually occur in it.
        println!("{} ({:.4}) lexical={} {:?}",
            hit.id, hit.score.unwrap_or(0.0), hit.is_lexical_match(), hit.content);
    }
    Ok(())
}
```

## API

All methods are `async` on [`Client`].

| Method | Endpoint |
|---|---|
| `create_index(index, &Schema)` | `PUT /{index}` |
| `get_index(index) -> IndexInfo` | `GET /{index}` |
| `list_indices() -> Vec<IndexInfo>` | `GET /_indices` |
| `delete_index(index)` | `DELETE /{index}` |
| `update_mapping(index, &Schema) -> MappingChanges` | `PUT /{index}/_mapping` |
| `index_document(index, id, &doc)` | `PUT /{index}/_doc/{id}` |
| `create_document(index, &doc) -> String` | `POST /{index}/_doc` (returns the id) |
| `get_document(index, id) -> Document` | `GET /{index}/_doc/{id}` |
| `delete_document(index, id)` | `DELETE /{index}/_doc/{id}` |
| `search(index, &SearchRequest) -> SearchResponse` | `POST /{index}/_search` |

Build mappings with [`Schema::builder`] and queries with the [`QueryClause`]
constructors (`match_field`, `multi_match`, `knn`, `hybrid`, `bool`).
`SearchRequest::new(..).size(n).from(m)` sets paging. Document bodies are any
`Serialize` value (e.g. `serde_json::json!`), so `_meta` / `_embed` are passed inline.

## Filtering

`keyword` fields are multi-valued (a string or an array of strings) and filterable.
A `bool` query pairs a scoring `must` (`match`/`multi_match`) with a filter context —
`filter`/`must_not` restrict which documents match but never change the score:

```rust
use search_service::{Filter, QueryClause, SearchRequest};

let query = QueryClause::bool()
    .must(QueryClause::match_field("body", "postgres"))
    .filter(Filter::terms("tags", ["rust", "db"]))   // keyword membership (any)
    .filter(Filter::range("views").gte(100))          // numeric/date bounds
    .must_not(Filter::term("published", false));
let res = client.search("posts", &SearchRequest::new(query)).await?;
```

Filter clauses: `Filter::term`, `Filter::terms`, `Filter::range(..).gte/gt/lte/lt`,
`Filter::exists`, and `Filter::bool()` (nested `should`/`must`/`must_not`/`filter`).
A `bool` with no `must` is a filtered `match_all` — hits are ordered by recency and
carry `score == None`.

## Faceted search

Add aggregations to get bucket counts over the whole matching set (independent of
paging — use `.size(0)` for a facets-only response). Use `post_filter` for faceted
navigation: it narrows the returned hits *after* aggregations are computed, so
selecting a facet value doesn't collapse that facet's own counts.

```rust
use search_service::{Agg, Filter, QueryClause, SearchRequest};

let req = SearchRequest::new(QueryClause::bool())
    .size(0)
    .agg("tags",  Agg::terms("tags").size(20))
    .agg("views", Agg::range("views").below(100.0).between(100.0, 1000.0).above(1000.0))
    .post_filter(Filter::term("tags", "rust"));

let res = client.search("posts", &req).await?;
if let Some(tags) = res.agg("tags") {
    for b in &tags.buckets {
        println!("{:?}: {}", b.key, b.doc_count);
    }
}
```

## Scores and ranking

`hit.score` holds **a different quantity per query type**, and the values are not
comparable across them:

| Query | `score` is | Range |
|---|---|---|
| `match` / `multi_match` | summed BM25 over the queried fields | positive, unbounded, corpus-dependent |
| `knn` | cosine similarity (also `hit.cosine()`) | `[-1, 1]` |
| `hybrid` | RRF score, `Σ wᵢ/(rrf_k + rankᵢ)` | `(0, Σw/(rrf_k+1)]` — **0.0328** at the defaults |
| filter-only `bool` | `None` — nothing scored it | — |

### Why a `hybrid` score can't tell you if a hit is real

RRF fuses rank into a scalar, which discards *which* leg produced a chunk. A chunk
ranked first by BM25 and absent from the vector list, and one ranked first by the
vector leg and absent from BM25, both score exactly `1/(rrf_k + 1)`. `hit.rank`
carries the per-leg ranks so you can separate them:

```rust
for hit in &res.hits.hits {
    match hit.rank.unwrap_or_default() {
        r if r.bm25.is_some() => println!("{}: query terms genuinely occur", hit.id),
        r if r.vector.is_some() => println!("{}: nearest-neighbour only", hit.id),
        _ => {}
    }
}

// Or the short form — true only when the BM25 leg matched:
let genuine: Vec<_> = res.hits.hits.iter().filter(|h| h.is_lexical_match()).collect();
```

`rank.bm25` is `Some` only when the query terms occur in the chunk, because that
leg is gated on a real lexical match. `rank.vector` is the chunk's place in the ANN
candidate window — populated for *any* query at all, which is the next point.

### Gating the vector leg

An ANN scan has no notion of "no match": nearest neighbours always exist, so an
unrelated query still returns a full page of plausible-looking hits.
`min_similarity` gates it the way BM25 is already gated (the service pushes the
floor into SQL rather than filtering fetched rows):

```rust
let q = VectorQuery::new("embedding", "wholly unrelated text").min_similarity(0.7);
let res = client.search("posts", &SearchRequest::new(QueryClause::Hybrid(q))).await?;
assert!(res.hits.hits.is_empty()); // nothing was close enough
```

### Tuning the fusion

```rust
let q = VectorQuery::new("embedding", "retire elasticsearch")
    .k(200)               // candidate depth per leg — NOT the page size
    .weights(1.0, 3.0)    // (vector, bm25): trust lexical evidence 3x
    .collapse(true)       // one best chunk per parent document
    .min_similarity(0.6);
let res = client.search("posts",
    &SearchRequest::new(QueryClause::Hybrid(q)).size(10).min_score(0.01)).await?;
```

`min_score` drops hits below a threshold, in whatever units that query type's score
uses — a value tuned for `match` is meaningless for `hybrid`. On document queries
the service folds it into the match predicate, so `total` and the facet counts stay
consistent with the returned hits.

> **`k` is candidate depth, not page size.** `size` truncates the page. A recent
> service change made this so: `k(100)` with the default `size` now yields 10 hits
> drawn from 100 candidates, where it previously returned 100.

> `hits.total.value` is the total matching documents for `match`/`multi_match`/`bool`.
> For `knn`/`hybrid` it is the size of the retained candidate set (after
> `min_similarity`, `min_score` and `collapse`) — a top-k ANN scan has no cheap exact
> count to report.

## Filtered vector search

`knn`/`hybrid` accept a `filter` on the parent document's fields (ES `knn.filter`);
the service pre-filters to the matching documents before scanning chunk embeddings.
Build a [`VectorQuery`] directly to attach one:

```rust
use search_service::{Filter, QueryClause, SearchRequest, VectorQuery};

let knn = VectorQuery::new("embedding", "combine lexical and vector search")
    .k(20)
    .filter(Filter::term("tags", "db"));
let res = client.search("posts", &SearchRequest::new(QueryClause::Knn(knn))).await?;
```

## Examples

Runnable, commented examples live in [`examples/`](examples/). Point
`SEARCH_SERVICE_URL` at a running service and:

```sh
SEARCH_SERVICE_URL=http://127.0.0.1:3000/search cargo run --example quickstart
SEARCH_SERVICE_URL=http://127.0.0.1:3000/search cargo run --example filtering
SEARCH_SERVICE_URL=http://127.0.0.1:3000/search cargo run --example faceted_search
SEARCH_SERVICE_URL=http://127.0.0.1:3000/search cargo run --example vector_filtered_search
SEARCH_SERVICE_URL=http://127.0.0.1:3000/search cargo run --example score_semantics
```

| Example | Shows |
|---|---|
| `quickstart` | create index, index docs, full-text search |
| `filtering` | `bool` query with `term`/`terms`/`range`/`must_not` and nested `bool` |
| `faceted_search` | `terms`/`range` aggregations + `post_filter` navigation |
| `vector_filtered_search` | `knn`/`hybrid` restricted by a document filter |
| `score_semantics` | what `_score` means per query type, hit provenance, similarity floors, fusion weights |

## Errors

Non-2xx responses become [`Error::Api`] carrying the service's `type` and `reason`.
Helpers: `err.status()`, `err.kind()`, `err.is_not_found()`.

```rust
match client.get_index("missing").await {
    Err(e) if e.is_not_found() => { /* 404 */ }
    other => { other?; }
}
```

## Testing

```sh
cargo test                      # unit tests (offline): serialization + URL building

# live end-to-end test against a running service (ignored by default):
SEARCH_SERVICE_URL=http://127.0.0.1:3000/search \
  cargo test --test live -- --ignored --nocapture
```

The live test exercises every endpoint; its `knn`/`hybrid` steps require the
server to have a working embedder.
