//! Functional graph API: build a weight-free graph, then `compile` it on a device.
//!
//! ```
//! use instmodel::graph::{InputBuffer, ModelGraph, ops};
//! use instmodel::layers::Activation;
//! use burn::backend::NdArray;
//!
//! let input = InputBuffer::new(4);
//! let hidden = ops::dense(8, Activation::Relu, input.buffer());
//! let output = ops::dense(1, Activation::Sigmoid, hidden);
//! let model = ModelGraph::new(vec![input], output)
//!     .compile::<NdArray>(&Default::default())
//!     .unwrap();
//! assert!(model.export_to_instruction_model().unwrap().contains("DOT"));
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
