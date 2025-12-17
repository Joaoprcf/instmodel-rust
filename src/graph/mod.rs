//! Functional graph API for building neural networks.
//!
//! This module provides a functional API for building computation graphs.
//!
//! # Example
//!
//! ```
//! use instmodel::graph::{InputBuffer, ModelGraph, ops};
//! use instmodel::layers::Activation;
//! use burn::backend::NdArray;
//! use burn::tensor::backend::Backend;
//!
//! type TestBackend = NdArray;
//! let device = <TestBackend as Backend>::Device::default();
//!
//! // Create input and build graph (no device needed)
//! let input = InputBuffer::new(4);
//! let x = input.buffer();
//! let hidden = ops::dense(8, Activation::Relu, x.clone());
//! let output = ops::dense(1, Activation::Sigmoid, hidden);
//! let graph = ModelGraph::new(vec![input], output);
//!
//! // Compile to create weights on device
//! let model = graph.compile::<TestBackend>(&device).unwrap();
//! ```

mod buffer;
mod compile;
mod core;
mod model;
mod operation;

pub use buffer::{BufferId, DataBuffer, InputBuffer, Producer};
pub use compile::{CompileContext, InstructionExport, InstructionModelExport};
pub use core::{GraphId, ModelGraph};
pub use model::{CompiledModel, SubModel};
pub use operation::{OpId, Operation, ops};
