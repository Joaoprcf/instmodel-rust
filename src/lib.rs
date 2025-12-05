//! # instmodel-rust
//!
//! A Rust library for building, training, and exporting instruction-based neural network models.
//!
//! This library provides a high-level API similar to the Python `instmodel` package,
//! allowing users to define neural network architectures, train them using optimizers
//! like Adam, and export them to a compact JSON instruction format for efficient inference.
//!
//! ## Features
//!
//! - **Burn Backend**: Uses the Burn framework with WGPU backend for GPU acceleration
//!   without external dependencies.
//! - **Familiar API**: Mirrors the Python `instmodel` API for easy migration.
//! - **Export to Instruction Model**: Trained models can be exported to JSON format
//!   compatible with `instmodel-rust-inference`.
//!
//! ## Example
//!
//! ```
//! use instmodel::prelude::*;
//! use burn::backend::NdArray;
//!
//! type Backend = NdArray;
//!
//! let device = <Backend as burn::tensor::backend::Backend>::Device::default();
//!
//! // Create a simple feed-forward model
//! let model: instmodel::model_graph::ModelGraph<Backend> = ModelGraphConfig::with_feature_size(4)
//!     .dense(8, Activation::Relu)
//!     .dense(1, Activation::Sigmoid)
//!     .build(&device)
//!     .expect("Failed to build model");
//!
//! // Export to instruction model format
//! let json = model.export_to_instruction_model().unwrap();
//! assert!(json.contains("DOT"));
//! ```

pub mod errors;
pub mod graph;
pub mod layers;
pub mod model_graph;
pub mod training;

// Re-exports for convenience
pub use errors::ModelError;
pub use layers::activation::Activation;
pub use model_graph::ModelGraph;
pub use training::{Loss, TrainingConfig};

/// Backend type alias for WGPU with autodiff support.
pub type Backend = burn::backend::Autodiff<burn::backend::Wgpu>;

/// Backend type for inference (no autodiff).
pub type InferenceBackend = burn::backend::Wgpu;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::errors::ModelError;
    pub use crate::layers::activation::Activation;
    pub use crate::model_graph::{ModelGraph, ModelGraphConfig};
    pub use crate::training::{Loss, TrainingConfig, train};
    pub use crate::{Backend, InferenceBackend};
}
