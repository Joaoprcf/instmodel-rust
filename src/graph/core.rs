use std::sync::atomic::{AtomicUsize, Ordering};

use burn::tensor::backend::Backend;

use super::buffer::{DataBuffer, InputBuffer, Producer};
use super::model::CompiledModel;
use crate::errors::ModelError;

static GRAPH_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Unique identifier for a ModelGraph.
pub type GraphId = usize;

/// Weight-free graph structure; `compile()` creates a [`CompiledModel`] with weights.
///
/// ```
/// use instmodel::graph::{InputBuffer, ModelGraph, ops};
/// use instmodel::layers::Activation;
/// use burn::backend::NdArray;
///
/// let input = InputBuffer::new(4);
/// let output = ops::dense(1, Activation::Sigmoid, input.buffer());
/// let model = ModelGraph::new(vec![input], output)
///     .compile::<NdArray>(&Default::default())
///     .unwrap();
/// assert_eq!(model.output_size(), 1);
/// ```
#[derive(Clone, Debug)]
pub struct ModelGraph {
    id: GraphId,
    inputs: Vec<InputBuffer>,
    output: DataBuffer,
}

impl ModelGraph {
    pub fn new(inputs: Vec<InputBuffer>, output: DataBuffer) -> Self {
        Self {
            id: GRAPH_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            inputs,
            output,
        }
    }

    pub fn id(&self) -> GraphId {
        self.id
    }

    pub fn inputs(&self) -> &[InputBuffer] {
        &self.inputs
    }

    pub fn output(&self) -> &DataBuffer {
        &self.output
    }

    pub fn feature_size(&self) -> usize {
        self.inputs.iter().map(InputBuffer::size).sum()
    }

    pub fn output_size(&self) -> usize {
        self.output.size()
    }

    pub fn features(&self) -> Vec<String> {
        self.inputs
            .iter()
            .filter_map(InputBuffer::features)
            .flatten()
            .cloned()
            .collect()
    }

    /// Uses this graph as a sub-model; every application shares the same weights.
    pub fn apply(&self, inputs: Vec<DataBuffer>) -> DataBuffer {
        assert_eq!(
            inputs.len(),
            self.inputs.len(),
            "Number of inputs must match the graph's input count"
        );
        DataBuffer::new_with_producer(
            self.output_size(),
            Some(Producer::SubGraph {
                id: super::operation::next_op_id(),
                input_ids: self.inputs.iter().map(InputBuffer::id).collect(),
                output: Box::new(self.output.clone()),
                graph_id: self.id,
            }),
            inputs,
        )
    }

    pub fn apply_single(&self, input: DataBuffer) -> DataBuffer {
        self.apply(vec![input])
    }

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
    fn nested_graph_sizes() {
        let inner_input = InputBuffer::new(4);
        let inner_output = ops::dense(8, Activation::Relu, inner_input.buffer());
        let inner = ModelGraph::new(vec![inner_input], inner_output);

        let outer_input = InputBuffer::new(4);
        let applied = inner.apply_single(outer_input.buffer());
        assert_eq!(applied.size(), 8);
        let outer = ModelGraph::new(
            vec![outer_input],
            ops::dense(1, Activation::Sigmoid, applied),
        );
        assert_eq!((outer.feature_size(), outer.output_size()), (4, 1));
        assert_ne!(inner.id(), outer.id());
    }

    #[test]
    #[should_panic(expected = "Number of inputs must match")]
    fn apply_checks_input_count() {
        let input = InputBuffer::new(2);
        let graph = ModelGraph::new(vec![input.clone()], input.buffer());
        graph.apply(vec![]);
    }
}
