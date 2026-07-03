//! Async Rust client for the **search-service** HTTP API (BM25 full-text +
//! vector / hybrid search, loosely Elasticsearch-shaped).
//!
//! ```no_run
//! use search_service::{Client, Schema, FieldType, SearchRequest, QueryClause};
//! # async fn demo() -> search_service::Result<()> {
//! let client = Client::new("http://localhost:3000/search")?;
//!
//! // Create an index.
//! let schema = Schema::builder()
//!     .text("title", Some("english"))
//!     .text("body", None)
//!     .scalar("views", FieldType::Integer)
//!     .vector("embedding", "gemini", "gemini-embedding-2", 1536, Some("retrieval"))
//!     .build();
//! client.create_index("posts", &schema).await?;
//!
//! // Index a document (with chunks to embed + metadata).
//! client.index_document("posts", "p1", &serde_json::json!({
//!     "title": "Hybrid search", "body": "...", "views": 10,
//!     "_meta": { "slug": "hybrid-search" },
//!     "_embed": { "embedding": ["passage one", "passage two"] }
//! })).await?;
//!
//! // Search.
//! let res = client.search("posts",
//!     &SearchRequest::new(QueryClause::hybrid("embedding", "retire elasticsearch"))).await?;
//! for hit in res.hits.hits {
//!     println!("{} ({:.3})", hit.id, hit.score);
//! }
//! # Ok(())
//! # }
//! ```

mod client;
mod error;
mod query;
mod response;
mod schema;

pub use client::Client;
pub use error::{Error, Result};
pub use query::{MultiMatch, QueryClause, SearchRequest, VectorQuery};
pub use response::{Document, Hit, Hits, SearchResponse, Total};
pub use schema::{FieldDef, FieldType, IndexInfo, MappingChanges, Schema, SchemaBuilder};
