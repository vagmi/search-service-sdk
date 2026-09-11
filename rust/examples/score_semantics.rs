//! Reading scores correctly: what `_score` means per query type, how to tell a real
//! lexical hit from vector filler, and how to stop an unrelated query returning a
//! confident-looking page.
//!
//! The trap this example exists for: `hybrid` fuses two rankings with Reciprocal
//! Rank Fusion, which sums `w/(k + rank)` and throws away *which* list a chunk came
//! from. A chunk ranked first by BM25 and one ranked first by the vector leg fuse to
//! exactly the same score — and since an ANN scan always returns its window, a query
//! matching nothing lexically still fills the page. `Hit::rank` is what tells them
//! apart; `min_similarity` is what gates the vector leg.
//!
//! NOTE: this requires the server to have a working embedder configured.
//!
//! ```sh
//! SEARCH_SERVICE_URL=http://127.0.0.1:3000/search cargo run --example score_semantics
//! ```

use search_service::{Client, Hit, QueryClause, Schema, SearchRequest, VectorQuery};
use serde_json::json;

const INDEX: &str = "sdk_example_score_semantics";

#[tokio::main]
async fn main() -> search_service::Result<()> {
    let base =
        std::env::var("SEARCH_SERVICE_URL").unwrap_or_else(|_| "http://localhost:3000".into());
    let client = Client::new(&base)?;
    let _ = client.delete_index(INDEX).await;

    let schema = Schema::builder()
        .text("body", None)
        .vector(
            "embedding",
            "gemini",
            "gemini-embedding-2",
            1536,
            Some("retrieval"),
        )
        .build();
    client.create_index(INDEX, &schema).await?;

    client
        .index_document(
            INDEX,
            "marsupials",
            &json!({
                "body": "marsupials",
                "_embed": { "embedding": ["kangaroos and wallabies are marsupials native to australia"] }
            }),
        )
        .await?;
    client
        .index_document(
            INDEX,
            "consensus",
            &json!({
                "body": "consensus",
                "_embed": { "embedding": ["raft and paxos achieve distributed consensus over quorum writes"] }
            }),
        )
        .await?;

    // 1. Document BM25: `_score` is a summed BM25 value — positive and unbounded,
    //    so its magnitude only means something relative to the same query.
    let res = client
        .search(
            INDEX,
            &SearchRequest::new(QueryClause::match_field("body", "marsupials")),
        )
        .await?;
    for hit in &res.hits.hits {
        println!(
            "[match]  {} score={:?} (bm25, unbounded)",
            hit.id, hit.score
        );
    }

    // 2. kNN: `_score` is cosine similarity, echoed raw via `Hit::cosine()`. Note
    //    that *both* documents come back — an ANN scan has no notion of "no match".
    let res = client
        .search(
            INDEX,
            &SearchRequest::new(QueryClause::knn("embedding", "kangaroo")),
        )
        .await?;
    for hit in &res.hits.hits {
        println!(
            "[knn]    {} score={:?} cosine={:?}",
            hit.id,
            hit.score,
            hit.cosine()
        );
    }

    // 3. Hybrid: `_score` is an RRF value bounded by Σw/(rrf_k+1) — 0.0328 at the
    //    defaults. Read `rank`, not `score`, to judge why a hit is here.
    let res = client
        .search(
            INDEX,
            &SearchRequest::new(QueryClause::hybrid("embedding", "kangaroo wallaby")),
        )
        .await?;
    for hit in &res.hits.hits {
        println!("[hybrid] {}", describe(hit));
    }

    // 4. The failure mode. A query that matches nothing lexically still returns a
    //    full page — every hit vector-only filler, as the provenance shows.
    let nonsense = "zzqx entirely unrelated gibberish token";
    let res = client
        .search(
            INDEX,
            &SearchRequest::new(QueryClause::hybrid("embedding", nonsense)),
        )
        .await?;
    println!(
        "[nonsense, no floor] {} hits, {} of them genuine lexical matches",
        res.hits.hits.len(),
        res.hits
            .hits
            .iter()
            .filter(|h| h.is_lexical_match())
            .count()
    );

    // 5. The fix: gate the vector leg on a similarity floor, so "nothing is close
    //    enough" can mean no hits. The service pushes this into SQL.
    let floored = VectorQuery::new("embedding", nonsense).min_similarity(0.8);
    let res = client
        .search(INDEX, &SearchRequest::new(QueryClause::Hybrid(floored)))
        .await?;
    println!("[nonsense, floor 0.8] {} hits", res.hits.hits.len());

    // 6. Tilt the fusion toward lexical evidence, keep one chunk per document, and
    //    drop anything under a threshold — note the threshold is in RRF units here.
    let tuned = VectorQuery::new("embedding", "kangaroo wallaby")
        .k(200)
        .weights(1.0, 3.0)
        .collapse(true);
    let res = client
        .search(
            INDEX,
            &SearchRequest::new(QueryClause::Hybrid(tuned)).min_score(0.01),
        )
        .await?;
    println!("[tuned] total={}", res.hits.total.value);
    for hit in &res.hits.hits {
        println!("[tuned]  {}", describe(hit));
    }

    client.delete_index(INDEX).await?;
    Ok(())
}

/// A chunk hit with its provenance spelled out.
fn describe(hit: &Hit) -> String {
    let rank = hit.rank.unwrap_or_default();
    let origin = match (rank.bm25, rank.vector) {
        (Some(b), Some(v)) => format!("both legs (bm25 #{b}, vector #{v})"),
        (Some(b), None) => format!("lexical only (bm25 #{b})"),
        (None, Some(v)) => format!("vector only (#{v}) — no query term occurs in it"),
        (None, None) => "no provenance reported".to_string(),
    };
    format!(
        "{} score={:.5} — {origin}",
        hit.id,
        hit.score.unwrap_or(0.0)
    )
}
