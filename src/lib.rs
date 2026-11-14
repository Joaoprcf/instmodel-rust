//! Build and train neural networks with Burn, then export them to the instmodel JSON
//! instruction format consumed by the instmodel inference engines.
//!
//! ```
//! use instmodel::prelude::*;
//! use burn::backend::NdArray;
//!
//! let model: ModelGraph<NdArray> = ModelGraphConfig::with_feature_size(4)
//!     .dense(8, Activation::Relu)
//!     .dense(1, Activation::Sigmoid)
//!     .build(&Default::default())
//!     .unwrap();
//! assert!(model.export_to_instruction_model().unwrap().contains("DOT"));
//! ```

pub mod errors;
pub mod graph;
pub mod layers;
pub mod model_graph;
pub mod training;

pub use errors::ModelError;
pub use layers::activation::Activation;
pub use model_graph::ModelGraph;
pub use training::{Loss, TrainingConfig};

/// WGPU with autodiff, for training.
pub type Backend = burn::backend::Autodiff<burn::backend::Wgpu>;

/// WGPU without autodiff, for inference.
pub type InferenceBackend = burn::backend::Wgpu;

pub mod prelude {
    pub use crate::errors::ModelError;
    pub use crate::layers::activation::Activation;
    pub use crate::model_graph::{ModelGraph, ModelGraphConfig};
    pub use crate::training::{Loss, TrainingConfig, train};
    pub use crate::{Backend, InferenceBackend};
}
