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
//! // Create input
//! let input = InputBuffer::new(4);
//!
//! // Build graph using convenience functions
//! let x = input.buffer();
//! let hidden = ops::dense(8, Activation::Relu, x.clone());
//! let output = ops::dense(1, Activation::Sigmoid, hidden);
//!
//! // Create model (layers initialized here)
//! let model = ModelGraph::<TestBackend>::new(vec![input], output, &device).unwrap();
//!
//! // Example with branching and Add
//! let input2 = InputBuffer::new(4);
//! let x2 = input2.buffer();
//! let branch1 = ops::dense(4, Activation::Relu, x2.clone());
//! let branch2 = ops::dense(4, Activation::Relu, x2);
//! let merged = ops::add(vec![branch1, branch2]);
//! let output2 = ops::dense(1, Activation::Sigmoid, merged);
//! let model2 = ModelGraph::<TestBackend>::new(vec![input2], output2, &device).unwrap();
//! ```

mod buffer;
mod compile;
mod model;
mod operation;

pub use buffer::{BufferId, DataBuffer, InputBuffer, Producer};
pub use compile::{CompileContext, InstructionExport, InstructionModelExport};
pub use model::ModelGraph;
pub use operation::{OpId, Operation, ops};
