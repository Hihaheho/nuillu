use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};

pub const COL_ID: &str = "id";
pub const COL_CONTENT: &str = "content";
pub const COL_RANK: &str = "rank";
pub const COL_VECTOR: &str = "vector";
pub const COL_CREATED_AT_MS: &str = "created_at_ms";
pub const COL_UPDATED_AT_MS: &str = "updated_at_ms";
pub const COL_SOURCE_IDS: &str = "source_ids";
pub const COL_METADATA_JSON: &str = "metadata_json";

pub fn memory_schema(vector_dims: i32) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(COL_ID, DataType::Utf8, false),
        Field::new(COL_CONTENT, DataType::Utf8, false),
        Field::new(COL_RANK, DataType::Int32, false),
        Field::new(
            COL_VECTOR,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                vector_dims,
            ),
            false,
        ),
        Field::new(COL_CREATED_AT_MS, DataType::Int64, false),
        Field::new(COL_UPDATED_AT_MS, DataType::Int64, false),
        Field::new(COL_SOURCE_IDS, DataType::Utf8, true),
        Field::new(COL_METADATA_JSON, DataType::Utf8, true),
    ]))
}
