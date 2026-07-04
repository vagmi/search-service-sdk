//! Filtered vector search: `knn` / `hybrid` restricted to documents matching a
//! filter (ES `knn.filter`).
//!
//! The filter references the parent document's fields (e.g. `tags`); the service
//! pre-filters to the matching document ids before scanning chunk embeddings. Build
//! a [`VectorQuery`] directly to attach a filter.
//!
//! NOTE: this requires the server to have a working embedder configured.
//!
//! ```sh
//! SEARCH_SERVICE_URL=http://127.0.0.1:3000/search cargo run --example vector_filtered_search
//! ```

use search_service::{Client, Filter, QueryClause, Schema, SearchRequest, VectorQuery};
use serde_json::json;

const INDEX: &str = "sdk_example_vector_filtered";

#[tokio::main]
async fn main() -> search_service::Result<()> {
    let base =
        std::env::var("SEARCH_SERVICE_URL").unwrap_or_else(|_| "http://localhost:3000".into());
    let client = Client::new(&base)?;
    let _ = client.delete_index(INDEX).await;

    // A vector field alongside a keyword field to filter on.
    let schema = Schema::builder()
        .keyword("tags")
        .vector(
            "embedding",
            "gemini",
            "gemini-embedding-2",
            1536,
            Some("retrieval"),
        )
        .build();
    client.create_index(INDEX, &schema).await?;

    // `_embed` chunks are embedded server-side and stored per document.
    client
        .index_document(
            INDEX,
            "a",
            &json!({ "tags": ["rust", "db"], "_embed": { "embedding": ["how to combine bm25 with vector search in postgres"] } }),
        )
        .await?;
    client
        .index_document(
            INDEX,
            "b",
            &json!({ "tags": ["ai"], "_embed": { "embedding": ["a distinct passage about semantic retrieval"] } }),
        )
        .await?;

    let query = "combining lexical and vector search";

    // 1. Unfiltered kNN over all chunks.
    let res = client
        .search(
            INDEX,
            &SearchRequest::new(QueryClause::knn("embedding", query)),
        )
        .await?;
    println!("knn (unfiltered): {:?}", chunk_docs(&res));

    // 2. Filtered kNN — restrict the ANN scan to documents tagged "db".
    let knn = VectorQuery::new("embedding", query)
        .k(20)
        .filter(Filter::term("tags", "db"));
    let res = client
        .search(INDEX, &SearchRequest::new(QueryClause::Knn(knn)))
        .await?;
    println!("knn (filter tags=db): {:?}", chunk_docs(&res));

    // 3. Filtered hybrid (BM25-on-content ⊕ vector, fused by RRF) with the same filter.
    let hybrid = VectorQuery::new("embedding", query).filter(Filter::term("tags", "ai"));
    let res = client
        .search(INDEX, &SearchRequest::new(QueryClause::Hybrid(hybrid)))
        .await?;
    println!("hybrid (filter tags=ai): {:?}", chunk_docs(&res));

    client.delete_index(INDEX).await?;
    Ok(())
}

/// The distinct parent document ids behind the returned chunk hits.
fn chunk_docs(res: &search_service::SearchResponse) -> Vec<&str> {
    let mut v: Vec<&str> = res.hits.hits.iter().map(|h| h.id.as_str()).collect();
    v.sort_unstable();
    v.dedup();
    v
}
