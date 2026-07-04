//! Faceted search: aggregations + `post_filter` for faceted navigation.
//!
//! Aggregations (`terms`, `range`) return bucket counts over the full matching set,
//! independent of pagination — so `size(0)` gives a facets-only response. `post_filter`
//! narrows the returned hits *after* aggregations are computed, so selecting a facet
//! value doesn't collapse that facet's own counts (the faceted-navigation invariant).
//!
//! ```sh
//! SEARCH_SERVICE_URL=http://127.0.0.1:3000/search cargo run --example faceted_search
//! ```

use search_service::{
    Agg, Client, FieldType, Filter, QueryClause, Schema, SearchRequest, SearchResponse,
};
use serde_json::json;

const INDEX: &str = "sdk_example_faceted";

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
        .build();
    client.create_index(INDEX, &schema).await?;

    let docs = [
        (
            "1",
            json!({ "body": "postgres search", "tags": ["rust", "db"], "views": 1200 }),
        ),
        (
            "2",
            json!({ "body": "postgres vector", "tags": ["db", "ai"],   "views": 50 }),
        ),
        (
            "3",
            json!({ "body": "rust services",   "tags": ["rust"],       "views": 800 }),
        ),
        (
            "4",
            json!({ "body": "db tuning",        "tags": ["db"],         "views": 300 }),
        ),
    ];
    for (id, doc) in &docs {
        client.index_document(INDEX, id, doc).await?;
    }

    // 1. Facets-only request: `size(0)` returns no hits, just the aggregation buckets
    //    over every matching document. `terms` counts tags; `range` buckets views.
    let req = SearchRequest::new(QueryClause::bool())
        .size(0)
        .agg("tags", Agg::terms("tags").size(10))
        .agg(
            "views",
            Agg::range("views")
                .below(100.0)
                .between(100.0, 1000.0)
                .above(1000.0),
        );
    let res = client.search(INDEX, &req).await?;
    println!("facets over all {} docs:", res.hits.total.value);
    print_facets(&res);

    // 2. Faceted navigation: the user clicks tag "rust". Use `post_filter` so the hits
    //    narrow to rust-tagged docs while the `tags` facet still shows every tag's
    //    count — letting the user pivot to another tag.
    let req = SearchRequest::new(QueryClause::bool())
        .agg("tags", Agg::terms("tags"))
        .post_filter(Filter::term("tags", "rust"));
    let res = client.search(INDEX, &req).await?;
    println!(
        "\nafter clicking tag=rust — {} hit(s):",
        res.hits.total.value
    );
    for hit in &res.hits.hits {
        println!("  doc {}", hit.id);
    }
    println!("tags facet still reflects the full set (not just rust):");
    print_facets(&res);

    client.delete_index(INDEX).await?;
    Ok(())
}

/// Pretty-print every aggregation's buckets.
fn print_facets(res: &SearchResponse) {
    let Some(aggs) = &res.aggregations else {
        return;
    };
    for (name, agg) in aggs {
        println!("  [{name}]");
        for b in &agg.buckets {
            match (b.from, b.to) {
                (None, None) => println!("    {:?}: {}", b.key, b.doc_count),
                _ => println!(
                    "    {} [{:?}..{:?}]: {}",
                    b.key_str().unwrap_or("?"),
                    b.from,
                    b.to,
                    b.doc_count
                ),
            }
        }
    }
}
