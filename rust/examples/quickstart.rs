//! Quickstart: create an index, add documents, run a full-text search.
//!
//! Run against a running search-service:
//!
//! ```sh
//! SEARCH_SERVICE_URL=http://127.0.0.1:3000/search cargo run --example quickstart
//! ```
//!
//! `SEARCH_SERVICE_URL` defaults to `http://localhost:3000` if unset.

use search_service::{Client, FieldType, QueryClause, Schema, SearchRequest};
use serde_json::json;

const INDEX: &str = "sdk_example_quickstart";

#[tokio::main]
async fn main() -> search_service::Result<()> {
    let base =
        std::env::var("SEARCH_SERVICE_URL").unwrap_or_else(|_| "http://localhost:3000".into());
    let client = Client::new(&base)?;

    // Start from a clean slate (ignore "index not found").
    let _ = client.delete_index(INDEX).await;

    // 1. Create an index. `keyword` fields are multi-valued and filterable/facetable;
    //    `text` fields are analyzed for BM25 full-text search.
    let schema = Schema::builder()
        .text("title", Some("english"))
        .text("body", None)
        .keyword("tags")
        .scalar("views", FieldType::Integer)
        .build();
    client.create_index(INDEX, &schema).await?;
    println!("created index {INDEX}");

    // 2. Index a few documents. `tags` accepts a string or an array of strings.
    let docs = [
        (
            "1",
            json!({ "title": "Postgres full-text search", "body": "BM25 ranking in Postgres", "tags": ["db", "search"], "views": 1200 }),
        ),
        (
            "2",
            json!({ "title": "Rust web services",         "body": "Building APIs with axum",   "tags": ["rust"],          "views": 340 }),
        ),
        (
            "3",
            json!({ "title": "Hybrid retrieval",          "body": "Combine BM25 and vectors",  "tags": ["db", "ai"],      "views": 890 }),
        ),
    ];
    for (id, doc) in &docs {
        client.index_document(INDEX, id, doc).await?;
    }
    println!("indexed {} documents", docs.len());

    // 3. Full-text search. `match`/`multi_match` return document hits with `_source`.
    let res = client
        .search(
            INDEX,
            &SearchRequest::new(QueryClause::match_field("body", "postgres bm25")),
        )
        .await?;
    println!("\n{} hit(s) for \"postgres bm25\":", res.hits.total.value);
    for hit in &res.hits.hits {
        // Document hits carry a relevance score and the stored `_source`.
        let title = hit
            .source
            .as_ref()
            .and_then(|s| s["title"].as_str())
            .unwrap_or("?");
        println!("  {} ({:.3}) — {title}", hit.id, hit.score.unwrap_or(0.0));
    }

    // Clean up.
    client.delete_index(INDEX).await?;
    Ok(())
}
