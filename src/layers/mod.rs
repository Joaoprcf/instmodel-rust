//! Neural network layer implementations.
//!
//! This module contains the building blocks for constructing neural networks,
//! including dense (fully connected) layers and activation functions.

pub mod activation;
pub mod dense;

pub use activation::Activation;
pub use dense::{Dense, DenseConfig};
