//! Index mapping types. These mirror the service's mapping JSON exactly.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The supported field types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    Text,
    Keyword,
    Integer,
    Long,
    Double,
    Boolean,
    Date,
    Vector,
}

/// A single field definition. Vector fields additionally set `provider`, `model`,
/// `dimensions`, and (optionally) `task`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDef {
    #[serde(rename = "type")]
    pub field_type: FieldType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analyzer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
}

impl Default for FieldDef {
    fn default() -> Self {
        Self {
            field_type: FieldType::Text,
            analyzer: None,
            provider: None,
            model: None,
            dimensions: None,
            task: None,
        }
    }
}

impl FieldDef {
    /// A text field with an optional analyzer (`text_config`).
    pub fn text(analyzer: Option<&str>) -> Self {
        Self {
            field_type: FieldType::Text,
            analyzer: analyzer.map(str::to_string),
            ..Default::default()
        }
    }

    /// A scalar/keyword field (no extra config).
    pub fn scalar(field_type: FieldType) -> Self {
        Self {
            field_type,
            ..Default::default()
        }
    }

    /// A vector field.
    pub fn vector(provider: &str, model: &str, dimensions: usize, task: Option<&str>) -> Self {
        Self {
            field_type: FieldType::Vector,
            provider: Some(provider.to_string()),
            model: Some(model.to_string()),
            dimensions: Some(dimensions),
            task: task.map(str::to_string),
            ..Default::default()
        }
    }
}

/// An index mapping: the set of named fields.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Schema {
    pub fields: BTreeMap<String, FieldDef>,
}

impl Schema {
    pub fn builder() -> SchemaBuilder {
        SchemaBuilder::default()
    }
}

/// Fluent builder for a [`Schema`].
#[derive(Debug, Default)]
pub struct SchemaBuilder {
    fields: BTreeMap<String, FieldDef>,
}

impl SchemaBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a field with an explicit definition.
    pub fn field(mut self, name: &str, def: FieldDef) -> Self {
        self.fields.insert(name.to_string(), def);
        self
    }

    pub fn text(self, name: &str, analyzer: Option<&str>) -> Self {
        self.field(name, FieldDef::text(analyzer))
    }

    pub fn keyword(self, name: &str) -> Self {
        self.field(name, FieldDef::scalar(FieldType::Keyword))
    }

    pub fn scalar(self, name: &str, field_type: FieldType) -> Self {
        self.field(name, FieldDef::scalar(field_type))
    }

    pub fn vector(
        self,
        name: &str,
        provider: &str,
        model: &str,
        dimensions: usize,
        task: Option<&str>,
    ) -> Self {
        self.field(name, FieldDef::vector(provider, model, dimensions, task))
    }

    pub fn build(self) -> Schema {
        Schema {
            fields: self.fields,
        }
    }
}

/// Index metadata as returned by `GET /{index}` and `GET /_indices`.
#[derive(Debug, Clone, Deserialize)]
pub struct IndexInfo {
    pub name: String,
    pub mappings: Schema,
    pub created_at: String,
}

/// What a `PUT /{index}/_mapping` changed.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct MappingChanges {
    #[serde(default)]
    pub added: Vec<String>,
    #[serde(default)]
    pub removed: Vec<String>,
    #[serde(default)]
    pub updated: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_serializes_to_service_shape() {
        let schema = Schema::builder()
            .text("title", Some("english"))
            .keyword("tags")
            .scalar("views", FieldType::Integer)
            .vector("embedding", "gemini", "gemini-embedding-2", 1536, Some("retrieval"))
            .build();
        let v = serde_json::to_value(&schema).unwrap();
        assert_eq!(v["fields"]["title"], serde_json::json!({"type":"text","analyzer":"english"}));
        assert_eq!(v["fields"]["tags"], serde_json::json!({"type":"keyword"}));
        assert_eq!(v["fields"]["views"], serde_json::json!({"type":"integer"}));
        assert_eq!(
            v["fields"]["embedding"],
            serde_json::json!({
                "type":"vector","provider":"gemini",
                "model":"gemini-embedding-2","dimensions":1536,"task":"retrieval"
            })
        );
    }
}
