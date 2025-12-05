//! Model-related error types.

use thiserror::Error;

/// Errors that can occur during model operations.
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Model has no layers defined")]
    NoLayers,

    #[error("Model has no input buffer defined")]
    NoInputBuffer,

    #[error("Invalid layer configuration: {message}")]
    InvalidLayerConfig { message: String },

    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },

    #[error("Export error: {message}")]
    ExportError { message: String },

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Training error: {message}")]
    TrainingError { message: String },

    #[error("Invalid activation: {name}")]
    InvalidActivation { name: String },

    #[error("Invalid graph structure: {0}")]
    InvalidGraph(String),
}
