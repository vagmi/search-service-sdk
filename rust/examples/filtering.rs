//! Filtering: restrict search results with a `bool` query's filter context.
//!
//! Filters (`term`, `terms`, `range`, `exists`, nested `bool`) narrow which
//! documents match but never affect the relevance score. A `bool` query pairs a
//! scoring `must` (a `match`/`multi_match`) with `filter`/`must_not` context; with
//! no `must` it becomes a filtered `match_all` (hits ordered by recency, no score).
//!
//! ```sh
//! SEARCH_SERVICE_URL=http://127.0.0.1:3000/search cargo run --example filtering
//! ```

use search_service::{Client, FieldType, Filter, QueryClause, Schema, SearchRequest};
use serde_json::json;

const INDEX: &str = "sdk_example_filtering";

#[tokio::main]
async fn main() -> search_service::Result<()> {
    let base =
        std::env::var("SEARCH_SERVICE_URL").unwrap_or_else(|_| "http://localhost:3000".into());
    let client = Client::new(&base)?;
    let _ = client.delete_index(INDEX).await;

    let schema = Schema::builder()
        .text("body", None)
        .keyword("tags")
        .scalar("views", FieldType::Integer)
        .scalar("published", FieldType::Boolean)
        .build();
    client.create_index(INDEX, &schema).await?;

    let docs = [
        (
            "1",
            json!({ "body": "postgres search bm25", "tags": ["rust", "db"], "views": 1200, "published": true }),
        ),
        (
            "2",
            json!({ "body": "postgres vector search", "tags": ["db", "ai"], "views": 50,   "published": true }),
        ),
        (
            "3",
            json!({ "body": "rust web services",      "tags": ["rust"],      "views": 800,  "published": false }),
        ),
        (
            "4",
            json!({ "body": "postgres tuning tips",   "tags": ["db"],        "views": 300,  "published": true }),
        ),
    ];
    for (id, doc) in &docs {
        client.index_document(INDEX, id, doc).await?;
    }

    // 1. Scoring query + filter context: match "postgres", restricted to tag "db"
    //    with views >= 100. Filters don't change the BM25 score, only the matches.
    let q = QueryClause::bool()
        .must(QueryClause::match_field("body", "postgres"))
        .filter(Filter::term("tags", "db"))
        .filter(Filter::range("views").gte(100));
    let res = client.search(INDEX, &SearchRequest::new(q)).await?;
    println!("match + filter (db, views>=100): {:?}", ids(&res));

    // 2. `terms` matches any of several tag values; `must_not` excludes.
    let q = QueryClause::bool()
        .filter(Filter::terms("tags", ["rust", "db"]))
        .must_not(Filter::term("published", false));
    let res = client.search(INDEX, &SearchRequest::new(q)).await?;
    // Filter-only query: hits are ordered by recency and carry no score.
    println!("terms(rust|db) AND published: {:?}", ids(&res));
    println!(
        "  (filter-only → score is None: {})",
        res.hits.hits.iter().all(|h| h.score.is_none())
    );

    // 3. A nested `bool` filter expresses OR / NOT logic within one clause.
    let q = QueryClause::bool().filter(
        Filter::bool()
            .should(Filter::term("tags", "ai"))
            .should(Filter::range("views").gte(1000)),
    );
    let res = client.search(INDEX, &SearchRequest::new(q)).await?;
    println!("tag=ai OR views>=1000: {:?}", ids(&res));

    client.delete_index(INDEX).await?;
    Ok(())
}

/// Collect the sorted hit ids for printing.
fn ids(res: &search_service::SearchResponse) -> Vec<&str> {
    let mut v: Vec<&str> = res.hits.hits.iter().map(|h| h.id.as_str()).collect();
    v.sort_unstable();
    v
}
