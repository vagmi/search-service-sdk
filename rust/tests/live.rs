//! Live end-to-end test of the SDK against a running search-service.
//!
//! `#[ignore]`d (needs a running server). Point `SEARCH_SERVICE_URL` at wherever
//! the router is mounted, then run:
//!
//! ```sh
//! SEARCH_SERVICE_URL=http://127.0.0.1:3000/search \
//!   cargo test --test live -- --ignored --nocapture
//! ```
//!
//! The `vector` parts require the server to have a working embedder.

use search_service::{Agg, Client, FieldType, Filter, QueryClause, Schema, SearchRequest};
use serde_json::json;

const INDEX: &str = "sdk_live_test";

fn client() -> Option<Client> {
    let url = std::env::var("SEARCH_SERVICE_URL").ok()?;
    Some(Client::new(&url).expect("valid SEARCH_SERVICE_URL"))
}

#[tokio::test]
#[ignore = "needs a running search-service at SEARCH_SERVICE_URL"]
async fn full_sdk_lifecycle() {
    let Some(client) = client() else {
        eprintln!("skipping: SEARCH_SERVICE_URL not set");
        return;
    };

    // Clean slate (ignore "not found").
    let _ = client.delete_index(INDEX).await;

    // create_index
    let schema = Schema::builder()
        .text("title", Some("english"))
        .text("body", Some("english"))
        .scalar("views", FieldType::Integer)
        .vector(
            "embedding",
            "gemini",
            "gemini-embedding-2",
            1536,
            Some("retrieval"),
        )
        .build();
    client
        .create_index(INDEX, &schema)
        .await
        .expect("create_index");

    // get_index round-trips the mapping
    let info = client.get_index(INDEX).await.expect("get_index");
    assert_eq!(info.name, INDEX);
    assert!(info.mappings.fields.contains_key("embedding"));

    // index_document with _meta + _embed chunks
    client
        .index_document(
            INDEX,
            "p1",
            &json!({
                "title": "Hybrid search in Postgres",
                "body": "Combine BM25 with vector embeddings for hybrid retrieval.",
                "views": 42,
                "_meta": { "slug": "hybrid" },
                "_embed": { "embedding": ["Hybrid retrieval fuses lexical and semantic scores."] }
            }),
        )
        .await
        .expect("index_document");

    // create_document returns a server-generated id
    let gen_id = client
        .create_document(
            INDEX,
            &json!({ "title": "Auto id", "body": "second doc", "views": 1 }),
        )
        .await
        .expect("create_document");
    assert!(!gen_id.is_empty());

    // get_document
    let doc = client
        .get_document(INDEX, "p1")
        .await
        .expect("get_document");
    assert!(doc.found);
    assert_eq!(doc.meta["slug"], json!("hybrid"));
    assert_eq!(doc.source["views"], json!(42));

    // match search → document hit
    let res = client
        .search(
            INDEX,
            &SearchRequest::new(QueryClause::match_field("body", "hybrid retrieval")),
        )
        .await
        .expect("match search");
    assert!(res.hits.hits.iter().any(|h| h.id == "p1"));
    assert!(
        res.hits.hits[0].source.is_some(),
        "document hit carries _source"
    );

    // knn search → chunk hit (requires the server's embedder)
    let res = client
        .search(
            INDEX,
            &SearchRequest::new(QueryClause::knn("embedding", "lexical and semantic fusion")),
        )
        .await
        .expect("knn search");
    assert!(!res.hits.hits.is_empty(), "knn returns chunks");
    let top = &res.hits.hits[0];
    assert!(
        top.chunk_id.is_some() && top.content.is_some(),
        "chunk hit shape"
    );

    // update_mapping (add a field)
    let evolved = Schema::builder()
        .text("title", Some("english"))
        .text("body", Some("english"))
        .scalar("views", FieldType::Integer)
        .vector(
            "embedding",
            "gemini",
            "gemini-embedding-2",
            1536,
            Some("retrieval"),
        )
        .keyword("tags")
        .build();
    let changes = client
        .update_mapping(INDEX, &evolved)
        .await
        .expect("update_mapping");
    assert_eq!(changes.added, vec!["tags".to_string()]);

    // list_indices includes ours
    let indices = client.list_indices().await.expect("list_indices");
    assert!(indices.iter().any(|i| i.name == INDEX));

    // delete_document
    client
        .delete_document(INDEX, "p1")
        .await
        .expect("delete_document");
    let missing = client.get_document(INDEX, "p1").await;
    assert!(
        missing.is_err() && missing.unwrap_err().is_not_found(),
        "deleted doc → 404"
    );

    // delete_index, then it's gone
    client.delete_index(INDEX).await.expect("delete_index");
    let gone = client.get_index(INDEX).await;
    assert!(
        gone.is_err() && gone.unwrap_err().is_not_found(),
        "deleted index → 404"
    );
}

const FACET_INDEX: &str = "sdk_live_facets";

#[tokio::test]
#[ignore = "needs a running search-service at SEARCH_SERVICE_URL"]
async fn filters_and_facets() {
    let Some(client) = client() else {
        eprintln!("skipping: SEARCH_SERVICE_URL not set");
        return;
    };
    let _ = client.delete_index(FACET_INDEX).await;

    let schema = Schema::builder()
        .text("body", None)
        .keyword("tags")
        .scalar("views", FieldType::Integer)
        .build();
    client
        .create_index(FACET_INDEX, &schema)
        .await
        .expect("create_index");

    for (id, doc) in [
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
            json!({ "body": "postgres tuning",  "tags": ["db"],         "views": 300 }),
        ),
    ] {
        client
            .index_document(FACET_INDEX, id, &doc)
            .await
            .expect("index");
    }

    // bool query: filter to tag "db" with views >= 100 → docs 1 and 4.
    let q = QueryClause::bool()
        .must(QueryClause::match_field("body", "postgres"))
        .filter(Filter::term("tags", "db"))
        .filter(Filter::range("views").gte(100));
    let res = client
        .search(FACET_INDEX, &SearchRequest::new(q))
        .await
        .expect("bool search");
    let mut got: Vec<&str> = res.hits.hits.iter().map(|h| h.id.as_str()).collect();
    got.sort_unstable();
    assert_eq!(got, vec!["1", "4"]);
    assert_eq!(res.hits.total.value, 2, "total is the real matched count");

    // facets-only + post_filter: hits narrow to rust, tags facet stays full.
    let req = SearchRequest::new(QueryClause::bool())
        .agg("tags", Agg::terms("tags").size(10))
        .post_filter(Filter::term("tags", "rust"));
    let res = client
        .search(FACET_INDEX, &req)
        .await
        .expect("facet search");
    assert_eq!(
        res.hits.total.value, 2,
        "post_filter narrows hits to rust-tagged"
    );
    let tags = res.agg("tags").expect("tags agg present");
    let db = tags
        .buckets
        .iter()
        .find(|b| b.key_str() == Some("db"))
        .expect("db bucket");
    assert_eq!(
        db.doc_count, 3,
        "facet counts ignore post_filter (db still 3)"
    );

    client
        .delete_index(FACET_INDEX)
        .await
        .expect("delete_index");
}
