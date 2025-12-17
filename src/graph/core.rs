//! ModelGraph - graph structure without weights.
//!
//! This module contains the non-generic ModelGraph that represents
//! the computation graph structure. Weights are only created when
//! `compile()` is called.

use std::sync::atomic::{AtomicUsize, Ordering};

use burn::tensor::backend::Backend;

use super::buffer::{BufferId, DataBuffer, InputBuffer, Producer};
use super::model::CompiledModel;
use crate::errors::ModelError;

/// Global counter for unique graph IDs.
static GRAPH_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Unique identifier for a ModelGraph.
pub type GraphId = usize;

fn next_graph_id() -> GraphId {
    GRAPH_ID_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// ModelGraph represents the computation graph structure without weights.
///
/// This is a non-generic struct that can be freely composed and nested.
/// Call `compile()` to create a `CompiledModel` with actual weight tensors.
///
/// # Example
///
/// ```
/// use instmodel::graph::{InputBuffer, ModelGraph, ops};
/// use instmodel::layers::Activation;
/// use burn::backend::NdArray;
/// use burn::tensor::backend::Backend;
///
/// let device = <NdArray as Backend>::Device::default();
/// let input = InputBuffer::new(4);
/// let output = ops::dense(1, Activation::Sigmoid, input.buffer());
/// let graph = ModelGraph::new(vec![input], output);
///
/// // Compile to create weights
/// let model = graph.compile::<NdArray>(&device).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct ModelGraph {
    /// Unique identifier for this graph (used for sub-model extraction)
    id: GraphId,
    /// Input buffers
    inputs: Vec<InputBuffer>,
    /// Output buffer (carries the graph structure via Producer chain)
    output: DataBuffer,
}

impl ModelGraph {
    /// Creates a new ModelGraph from input buffers and output buffer.
    ///
    /// This only captures the graph structure - no weights are created.
    /// Call `compile()` to initialize weights on a specific device.
    pub fn new(inputs: Vec<InputBuffer>, output: DataBuffer) -> Self {
        Self {
            id: next_graph_id(),
            inputs,
            output,
        }
    }

    /// Returns the unique ID of this graph.
    pub fn id(&self) -> GraphId {
        self.id
    }

    /// Returns the input buffers.
    pub fn inputs(&self) -> &[InputBuffer] {
        &self.inputs
    }

    /// Returns the output buffer.
    pub fn output(&self) -> &DataBuffer {
        &self.output
    }

    /// Returns the number of input features.
    pub fn feature_size(&self) -> usize {
        self.inputs.iter().map(|i| i.size()).sum()
    }

    /// Returns the output size of the model.
    pub fn output_size(&self) -> usize {
        self.output.size()
    }

    /// Returns the feature names from all inputs.
    pub fn features(&self) -> Vec<String> {
        self.inputs
            .iter()
            .filter_map(|i| i.features())
            .flatten()
            .cloned()
            .collect()
    }

    /// Applies this graph to the given inputs, returning a DataBuffer.
    ///
    /// This enables nested model composition where a ModelGraph can be used
    /// like any other operation in the graph.
    pub fn apply(&self, inputs: Vec<DataBuffer>) -> DataBuffer {
        assert_eq!(
            inputs.len(),
            self.inputs.len(),
            "Number of inputs must match the graph's input count"
        );

        // Generate a unique ID for this sub-graph application
        let id = super::operation::next_op_id();

        // Collect input buffer IDs from the original graph
        let input_ids: Vec<BufferId> = self.inputs.iter().map(|i| i.id()).collect();

        DataBuffer::new_with_producer(
            self.output_size(),
            Some(Producer::SubGraph {
                id,
                input_ids,
                output: Box::new(self.output.clone()),
                graph_id: self.id,
            }),
            inputs,
        )
    }

    /// Applies this graph to a single input, returning a DataBuffer.
    ///
    /// Convenience method for graphs with a single input.
    pub fn apply_single(&self, input: DataBuffer) -> DataBuffer {
        self.apply(vec![input])
    }

    /// Compiles this graph into a CompiledModel with initialized weights.
    ///
    /// This is where the actual weight tensors are created on the specified device.
    pub fn compile<B: Backend>(&self, device: &B::Device) -> Result<CompiledModel<B>, ModelError> {
        CompiledModel::new(self.inputs.clone(), self.output.clone(), device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::operation::ops;
    use crate::layers::Activation;

    #[test]
    fn test_graph_creation() {
        let input = InputBuffer::new(4);
        let output = ops::dense(1, Activation::Sigmoid, input.buffer());
        let graph = ModelGraph::new(vec![input], output);

        assert_eq!(graph.feature_size(), 4);
        assert_eq!(graph.output_size(), 1);
    }

    #[test]
    fn test_graph_apply() {
        // Create a sub-graph
        let sub_input = InputBuffer::new(4);
        let sub_output = ops::dense(2, Activation::Relu, sub_input.buffer());
        let sub_graph = ModelGraph::new(vec![sub_input], sub_output);

        // Apply to a new input
        let main_input = InputBuffer::new(4);
        let applied = sub_graph.apply_single(main_input.buffer());

        assert_eq!(applied.size(), 2);
    }

    #[test]
    fn test_nested_graphs() {
        // Create inner graph
        let inner_input = InputBuffer::new(4);
        let inner_output = ops::dense(8, Activation::Relu, inner_input.buffer());
        let inner_graph = ModelGraph::new(vec![inner_input], inner_output);

        // Create outer graph using inner
        let outer_input = InputBuffer::new(4);
        let applied_inner = inner_graph.apply_single(outer_input.buffer());
        let outer_output = ops::dense(1, Activation::Sigmoid, applied_inner);
        let outer_graph = ModelGraph::new(vec![outer_input], outer_output);

        assert_eq!(outer_graph.feature_size(), 4);
        assert_eq!(outer_graph.output_size(), 1);
    }

    #[test]
    fn test_graph_ids_are_unique() {
        let input1 = InputBuffer::new(4);
        let output1 = ops::dense(1, Activation::None, input1.buffer());
        let graph1 = ModelGraph::new(vec![input1], output1);

        let input2 = InputBuffer::new(4);
        let output2 = ops::dense(1, Activation::None, input2.buffer());
        let graph2 = ModelGraph::new(vec![input2], output2);

        assert_ne!(graph1.id(), graph2.id());
    }
}
