use serde::Deserialize;
use thiserror::Error;

/// Errors returned by the client.
#[derive(Debug, Error)]
pub enum Error {
    /// The service returned a non-2xx response with its `{ "error": { ... } }` body.
    #[error("api error {status} ({kind}): {reason}")]
    Api {
        status: u16,
        kind: String,
        reason: String,
    },

    /// The service returned a non-2xx response we couldn't parse as an API error.
    #[error("unexpected {status} response: {body}")]
    Unexpected { status: u16, body: String },

    /// Transport / request failure.
    #[error(transparent)]
    Http(#[from] reqwest::Error),

    /// Invalid base URL.
    #[error("invalid base url: {0}")]
    Url(#[from] url::ParseError),

    /// (De)serialization failure.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// The body shape the service uses for errors: `{ "error": { "type", "reason" } }`.
#[derive(Debug, Deserialize)]
pub(crate) struct ApiErrorBody {
    pub error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ApiErrorDetail {
    #[serde(rename = "type")]
    pub kind: String,
    pub reason: String,
}

impl Error {
    /// Map a non-2xx status + raw body into a typed error.
    pub(crate) fn from_response(status: u16, body: &str) -> Self {
        match serde_json::from_str::<ApiErrorBody>(body) {
            Ok(parsed) => Error::Api {
                status,
                kind: parsed.error.kind,
                reason: parsed.error.reason,
            },
            Err(_) => Error::Unexpected {
                status,
                body: body.to_string(),
            },
        }
    }

    /// True if this is an API error with the given HTTP status.
    pub fn status(&self) -> Option<u16> {
        match self {
            Error::Api { status, .. } | Error::Unexpected { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// The service's machine-readable error `type`, if any (e.g. `index_not_found`).
    pub fn kind(&self) -> Option<&str> {
        match self {
            Error::Api { kind, .. } => Some(kind),
            _ => None,
        }
    }

    /// Convenience: was this a 404 from the service?
    pub fn is_not_found(&self) -> bool {
        self.status() == Some(404)
    }
}
