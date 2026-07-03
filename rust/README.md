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
        // chunk hits carry chunk_id / seq / content; document hits carry source / meta
        println!("{} ({:.3}) {:?}", hit.id, hit.score, hit.content);
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

Build mappings with [`Schema::builder`] and queries with the
[`QueryClause`] constructors (`match_field`, `multi_match`, `knn`, `hybrid`).
`SearchRequest::new(..).size(n).from(m)` sets paging. Document bodies are any
`Serialize` value (e.g. `serde_json::json!`), so `_meta` / `_embed` are passed inline.

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
