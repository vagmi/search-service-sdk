//! The async HTTP client.

use serde::Serialize;
use serde::de::DeserializeOwned;
use url::Url;

use crate::error::{Error, Result};
use crate::response::{DocAck, IndicesList};
use crate::schema::{IndexInfo, MappingChanges, Schema};
use crate::{Document, SearchRequest, SearchResponse};

/// Async client for the search-service HTTP API.
///
/// `base_url` is wherever the router is mounted — the service root if mounted at
/// `/`, or e.g. `http://host:3000/search` if nested under `/search`.
#[derive(Debug, Clone)]
pub struct Client {
    http: reqwest::Client,
    base: Url,
}

impl Client {
    /// Create a client pointed at the given base URL.
    pub fn new(base_url: &str) -> Result<Self> {
        Ok(Self {
            http: reqwest::Client::new(),
            base: parse_base(base_url)?,
        })
    }

    /// Create a client with a caller-provided [`reqwest::Client`] (timeouts, proxies, ...).
    pub fn with_http_client(base_url: &str, http: reqwest::Client) -> Result<Self> {
        Ok(Self {
            http,
            base: parse_base(base_url)?,
        })
    }

    fn url(&self, segments: &[&str]) -> Url {
        let mut url = self.base.clone();
        {
            // `parse_base` guarantees a base URL, so this never errors.
            let mut path = url.path_segments_mut().expect("base url");
            path.pop_if_empty();
            path.extend(segments);
        }
        url
    }

    // --- indices ---

    /// `PUT /{index}` — create an index from a mapping. Errors `409` if it exists.
    pub async fn create_index(&self, index: &str, schema: &Schema) -> Result<()> {
        self.send_discard(self.http.put(self.url(&[index])).json(schema))
            .await
    }

    /// `GET /{index}` — fetch an index's mapping + metadata.
    pub async fn get_index(&self, index: &str) -> Result<IndexInfo> {
        self.send_json(self.http.get(self.url(&[index]))).await
    }

    /// `GET /_indices` — list all indices.
    pub async fn list_indices(&self) -> Result<Vec<IndexInfo>> {
        let list: IndicesList = self.send_json(self.http.get(self.url(&["_indices"]))).await?;
        Ok(list.indices)
    }

    /// `DELETE /{index}` — drop an index (and its chunk tables).
    pub async fn delete_index(&self, index: &str) -> Result<()> {
        self.send_discard(self.http.delete(self.url(&[index]))).await
    }

    /// `PUT /{index}/_mapping` — evolve the mapping; returns what changed.
    pub async fn update_mapping(&self, index: &str, schema: &Schema) -> Result<MappingChanges> {
        #[derive(serde::Deserialize)]
        struct Resp {
            changes: MappingChanges,
        }
        let resp: Resp = self
            .send_json(self.http.put(self.url(&[index, "_mapping"])).json(schema))
            .await?;
        Ok(resp.changes)
    }

    // --- documents ---

    /// `PUT /{index}/_doc/{id}` — index (upsert) a document with a client-supplied id.
    pub async fn index_document(
        &self,
        index: &str,
        id: &str,
        document: &impl Serialize,
    ) -> Result<()> {
        self.send_discard(
            self.http
                .put(self.url(&[index, "_doc", id]))
                .json(document),
        )
        .await
    }

    /// `POST /{index}/_doc` — index a document with a server-generated id (returned).
    pub async fn create_document(&self, index: &str, document: &impl Serialize) -> Result<String> {
        let ack: DocAck = self
            .send_json(self.http.post(self.url(&[index, "_doc"])).json(document))
            .await?;
        Ok(ack.id)
    }

    /// `GET /{index}/_doc/{id}` — fetch a document.
    pub async fn get_document(&self, index: &str, id: &str) -> Result<Document> {
        self.send_json(self.http.get(self.url(&[index, "_doc", id])))
            .await
    }

    /// `DELETE /{index}/_doc/{id}` — delete a document.
    pub async fn delete_document(&self, index: &str, id: &str) -> Result<()> {
        self.send_discard(self.http.delete(self.url(&[index, "_doc", id])))
            .await
    }

    // --- search ---

    /// `POST /{index}/_search` — run a `match` / `multi_match` / `knn` / `hybrid` query.
    pub async fn search(&self, index: &str, request: &SearchRequest) -> Result<SearchResponse> {
        self.send_json(self.http.post(self.url(&[index, "_search"])).json(request))
            .await
    }

    // --- helpers ---

    async fn send_json<T: DeserializeOwned>(&self, req: reqwest::RequestBuilder) -> Result<T> {
        let resp = req.send().await?;
        let status = resp.status();
        let body = resp.text().await?;
        if status.is_success() {
            Ok(serde_json::from_str(&body)?)
        } else {
            Err(Error::from_response(status.as_u16(), &body))
        }
    }

    async fn send_discard(&self, req: reqwest::RequestBuilder) -> Result<()> {
        let resp = req.send().await?;
        let status = resp.status();
        if status.is_success() {
            Ok(())
        } else {
            let body = resp.text().await.unwrap_or_default();
            Err(Error::from_response(status.as_u16(), &body))
        }
    }
}

/// Parse + normalize a base URL (must be absolute http/https).
fn parse_base(base_url: &str) -> Result<Url> {
    let url = Url::parse(base_url)?;
    if url.cannot_be_a_base() || !matches!(url.scheme(), "http" | "https") {
        return Err(Error::Url(url::ParseError::RelativeUrlWithoutBase));
    }
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_urls_under_a_nested_base() {
        let c = Client::new("http://localhost:3000/search").unwrap();
        assert_eq!(c.url(&["posts"]).as_str(), "http://localhost:3000/search/posts");
        assert_eq!(
            c.url(&["posts", "_doc", "id-1"]).as_str(),
            "http://localhost:3000/search/posts/_doc/id-1"
        );
        assert_eq!(c.url(&["_indices"]).as_str(), "http://localhost:3000/search/_indices");
    }

    #[test]
    fn builds_urls_at_root_and_encodes() {
        let c = Client::new("http://localhost:3000").unwrap();
        assert_eq!(c.url(&["posts", "_search"]).as_str(), "http://localhost:3000/posts/_search");
        // path segments are percent-encoded.
        assert_eq!(
            c.url(&["posts", "_doc", "a b/c"]).as_str(),
            "http://localhost:3000/posts/_doc/a%20b%2Fc"
        );
    }

    #[test]
    fn rejects_non_http_base() {
        assert!(Client::new("not a url").is_err());
        assert!(Client::new("ftp://x").is_err());
    }
}
