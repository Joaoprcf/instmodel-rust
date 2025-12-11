//! Operations for the computation graph.
//!
//! Uses a simple enum instead of trait objects for clarity.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::layers::Activation;

use super::buffer::DataBuffer;
use super::compile::{CompileContext, InstructionExport};

/// Unique identifier for an operation.
pub type OpId = usize;

/// Global counter for unique operation IDs.
static OP_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Generates a new unique operation ID.
fn next_op_id() -> OpId {
    OP_ID_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// A computation operation in the graph.
#[derive(Debug, Clone)]
pub enum Operation {
    /// Dense (fully connected) layer.
    Dense {
        id: OpId,
        output_size: usize,
        activation: Activation,
    },
    /// Element-wise addition of buffers.
    Add { id: OpId },
    /// Element-wise multiplication of buffers.
    Multiply { id: OpId },
    /// Concatenate buffers along feature dimension.
    Concat { id: OpId },
    /// Softmax activation (operates across entire buffer).
    Softmax { id: OpId },
    /// Element-wise multiply by a fixed parameter vector.
    Scale { id: OpId, vector: Vec<f32> },
    /// Element-wise add a fixed parameter vector.
    Shift { id: OpId, vector: Vec<f32> },
    /// Batch normalization layer with trainable parameters.
    BatchNorm { id: OpId, epsilon: f32 },
}

impl Operation {
    /// Creates a new Dense operation.
    pub fn dense(output_size: usize, activation: Activation) -> Self {
        Self::Dense {
            id: next_op_id(),
            output_size,
            activation,
        }
    }

    /// Creates a new Add operation.
    pub fn add() -> Self {
        Self::Add { id: next_op_id() }
    }

    /// Creates a new Multiply operation.
    pub fn multiply() -> Self {
        Self::Multiply { id: next_op_id() }
    }

    /// Creates a new Concat operation.
    pub fn concat() -> Self {
        Self::Concat { id: next_op_id() }
    }

    /// Creates a new Softmax operation.
    pub fn softmax() -> Self {
        Self::Softmax { id: next_op_id() }
    }

    /// Creates a new Scale operation (element-wise multiply by fixed vector).
    pub fn scale(vector: Vec<f32>) -> Self {
        Self::Scale {
            id: next_op_id(),
            vector,
        }
    }

    /// Creates a new Shift operation (element-wise add fixed vector).
    pub fn shift(vector: Vec<f32>) -> Self {
        Self::Shift {
            id: next_op_id(),
            vector,
        }
    }

    /// Creates a new BatchNorm operation with trainable parameters.
    pub fn batch_norm(epsilon: f32) -> Self {
        Self::BatchNorm {
            id: next_op_id(),
            epsilon,
        }
    }

    /// Creates a new BatchNorm operation with default epsilon (1e-3).
    pub fn batch_norm_default() -> Self {
        Self::batch_norm(1e-3)
    }

    /// Returns the unique ID of this operation.
    pub fn id(&self) -> OpId {
        match self {
            Self::Dense { id, .. } => *id,
            Self::Add { id } => *id,
            Self::Multiply { id } => *id,
            Self::Concat { id } => *id,
            Self::Softmax { id } => *id,
            Self::Scale { id, .. } => *id,
            Self::Shift { id, .. } => *id,
            Self::BatchNorm { id, .. } => *id,
        }
    }

    /// Returns the output size for this operation given input sizes.
    pub fn output_size(&self, input_sizes: &[usize]) -> usize {
        match self {
            Self::Dense { output_size, .. } => *output_size,
            Self::Add { .. } | Self::Multiply { .. } => input_sizes[0],
            Self::Concat { .. } => input_sizes.iter().sum(),
            Self::Softmax { .. } => input_sizes[0],
            Self::Scale { vector, .. } => vector.len(),
            Self::Shift { vector, .. } => vector.len(),
            Self::BatchNorm { .. } => input_sizes[0],
        }
    }

    /// Applies this operation to an input buffer, returning a new DataBuffer.
    pub fn apply(&self, input: DataBuffer) -> DataBuffer {
        let output_size = self.output_size(&[input.size()]);
        DataBuffer::new(output_size, Some(self.clone()), vec![input])
    }

    /// Applies this operation to multiple input buffers.
    pub fn apply_multi(&self, inputs: Vec<DataBuffer>) -> DataBuffer {
        let input_sizes: Vec<usize> = inputs.iter().map(|b| b.size()).collect();
        let output_size = self.output_size(&input_sizes);
        DataBuffer::new(output_size, Some(self.clone()), inputs)
    }
}

/// Convenience functions for creating operations and applying them in one step.
/// These provide a Python-like API.
pub mod ops {
    use super::*;

    /// Creates a Dense layer and applies it to the input.
    ///
    /// # Example
    /// ```
    /// use instmodel::graph::{InputBuffer, ops};
    /// use instmodel::layers::Activation;
    ///
    /// let input = InputBuffer::new(4);
    /// let output = ops::dense(8, Activation::Relu, input.buffer());
    /// assert_eq!(output.size(), 8);
    /// ```
    pub fn dense(output_size: usize, activation: Activation, input: DataBuffer) -> DataBuffer {
        Operation::dense(output_size, activation).apply(input)
    }

    /// Creates an Add operation and applies it to the inputs.
    ///
    /// # Example
    /// ```
    /// use instmodel::graph::{InputBuffer, ops};
    /// use instmodel::layers::Activation;
    ///
    /// let input = InputBuffer::new(4);
    /// let x = ops::dense(4, Activation::Relu, input.buffer());
    /// let y = ops::dense(4, Activation::Relu, x.clone());
    /// let sum = ops::add(vec![x, y]);
    /// assert_eq!(sum.size(), 4);
    /// ```
    pub fn add(inputs: Vec<DataBuffer>) -> DataBuffer {
        assert!(inputs.len() >= 2, "Add requires at least 2 inputs");
        let size = inputs[0].size();
        for input in &inputs[1..] {
            assert_eq!(
                input.size(),
                size,
                "All inputs to Add must have the same size"
            );
        }
        Operation::add().apply_multi(inputs)
    }

    /// Creates a Multiply operation and applies it to the inputs.
    ///
    /// # Example
    /// ```
    /// use instmodel::graph::{InputBuffer, ops};
    /// use instmodel::layers::Activation;
    ///
    /// let input = InputBuffer::new(4);
    /// let x = ops::dense(4, Activation::Relu, input.buffer());
    /// let y = ops::dense(4, Activation::Relu, x.clone());
    /// let product = ops::multiply(vec![x, y]);
    /// assert_eq!(product.size(), 4);
    /// ```
    pub fn multiply(inputs: Vec<DataBuffer>) -> DataBuffer {
        assert!(inputs.len() >= 2, "Multiply requires at least 2 inputs");
        let size = inputs[0].size();
        for input in &inputs[1..] {
            assert_eq!(
                input.size(),
                size,
                "All inputs to Multiply must have the same size"
            );
        }
        Operation::multiply().apply_multi(inputs)
    }

    /// Concatenates buffers along the feature dimension.
    ///
    /// # Example
    /// ```
    /// use instmodel::graph::{InputBuffer, ops};
    /// use instmodel::layers::Activation;
    ///
    /// let input = InputBuffer::new(4);
    /// let features_a = ops::dense(3, Activation::Relu, input.buffer());
    /// let features_b = ops::dense(2, Activation::Relu, features_a.clone());
    /// let combined = ops::concat(vec![features_a, features_b]);
    /// assert_eq!(combined.size(), 5);
    /// ```
    pub fn concat(inputs: Vec<DataBuffer>) -> DataBuffer {
        assert!(!inputs.is_empty(), "Concat requires at least 1 input");
        Operation::concat().apply_multi(inputs)
    }

    /// Applies softmax activation to the input buffer.
    ///
    /// # Example
    /// ```
    /// use instmodel::graph::{InputBuffer, ops};
    /// use instmodel::layers::Activation;
    ///
    /// let input = InputBuffer::new(4);
    /// let logits = ops::dense(3, Activation::None, input.buffer());
    /// let probs = ops::softmax(logits);
    /// assert_eq!(probs.size(), 3);
    /// ```
    pub fn softmax(input: DataBuffer) -> DataBuffer {
        Operation::softmax().apply(input)
    }

    /// Scales (element-wise multiply) the input by a fixed parameter vector.
    ///
    /// # Example
    /// ```
    /// use instmodel::graph::{InputBuffer, ops};
    ///
    /// let input = InputBuffer::new(3);
    /// let scaled = ops::scale(vec![1.0, 2.0, 3.0], input.buffer());
    /// assert_eq!(scaled.size(), 3);
    /// ```
    pub fn scale(vector: Vec<f32>, input: DataBuffer) -> DataBuffer {
        assert_eq!(
            vector.len(),
            input.size(),
            "Scale vector length must match input size"
        );
        Operation::scale(vector).apply(input)
    }

    /// Shifts (element-wise add) the input by a fixed parameter vector.
    ///
    /// # Example
    /// ```
    /// use instmodel::graph::{InputBuffer, ops};
    ///
    /// let input = InputBuffer::new(3);
    /// let shifted = ops::shift(vec![0.1, 0.2, 0.3], input.buffer());
    /// assert_eq!(shifted.size(), 3);
    /// ```
    pub fn shift(vector: Vec<f32>, input: DataBuffer) -> DataBuffer {
        assert_eq!(
            vector.len(),
            input.size(),
            "Shift vector length must match input size"
        );
        Operation::shift(vector).apply(input)
    }

    /// Applies batch normalization to the input.
    ///
    /// During training, tracks running mean/variance. At export, converts to
    /// element-wise ADD and MUL operations for efficient inference.
    ///
    /// # Example
    /// ```
    /// use instmodel::graph::{InputBuffer, ops};
    ///
    /// let input = InputBuffer::new(4);
    /// let normalized = ops::batch_norm(input.buffer());
    /// assert_eq!(normalized.size(), 4);
    /// ```
    pub fn batch_norm(input: DataBuffer) -> DataBuffer {
        Operation::batch_norm_default().apply(input)
    }

    /// Applies batch normalization with custom epsilon.
    ///
    /// # Example
    /// ```
    /// use instmodel::graph::{InputBuffer, ops};
    ///
    /// let input = InputBuffer::new(4);
    /// let normalized = ops::batch_norm_with_epsilon(1e-5, input.buffer());
    /// assert_eq!(normalized.size(), 4);
    /// ```
    pub fn batch_norm_with_epsilon(epsilon: f32, input: DataBuffer) -> DataBuffer {
        Operation::batch_norm(epsilon).apply(input)
    }
}

impl Operation {
    /// Compiles this operation to instruction format.
    pub fn compile(&self, input_indices: &[usize], ctx: &mut CompileContext) -> usize {
        match self {
            Self::Dense {
                id,
                output_size,
                activation,
            } => {
                let weight_index = ctx.get_or_create_weight_index(*id);
                let output_index = ctx.allocate_buffer(*output_size);

                let instruction = InstructionExport::Dot {
                    input: input_indices[0],
                    output: output_index,
                    weights: weight_index,
                    activation: activation.to_instruction_name().map(String::from),
                };
                ctx.add_instruction(instruction);

                output_index
            }
            Self::Add { .. } => {
                let output_size = ctx.buffer_sizes[input_indices[0]];
                let output_index = ctx.allocate_buffer(output_size);

                let instruction = InstructionExport::AddBuffers {
                    input: input_indices.to_vec(),
                    output: output_index,
                };
                ctx.add_instruction(instruction);

                output_index
            }
            Self::Multiply { .. } => {
                let output_size = ctx.buffer_sizes[input_indices[0]];
                let output_index = ctx.allocate_buffer(output_size);

                let instruction = InstructionExport::MultiplyBuffers {
                    input: input_indices.to_vec(),
                    output: output_index,
                };
                ctx.add_instruction(instruction);

                output_index
            }
            Self::Concat { .. } => {
                let total_size: usize = input_indices.iter().map(|&i| ctx.buffer_sizes[i]).sum();
                let output_index = ctx.allocate_buffer(total_size);

                let mut offset = 0;
                for &input_idx in input_indices {
                    let input_size = ctx.buffer_sizes[input_idx];

                    // Use COPY with internal_index (like Python Concatenate)
                    // This copies the entire input buffer to output at offset
                    let instruction = InstructionExport::Copy {
                        input: input_idx,
                        output: output_index,
                        internal_index: offset,
                    };
                    ctx.add_instruction(instruction);

                    offset += input_size;
                }

                output_index
            }
            Self::Softmax { .. } => {
                let output_size = ctx.buffer_sizes[input_indices[0]];
                let output_index = ctx.allocate_buffer(output_size);

                // First copy input to output
                let copy_instruction = InstructionExport::Copy {
                    input: input_indices[0],
                    output: output_index,
                    internal_index: 0,
                };
                ctx.add_instruction(copy_instruction);

                // Then apply softmax activation
                let activation_instruction = InstructionExport::Activation {
                    input: output_index,
                    activation: "SOFTMAX".to_string(),
                };
                ctx.add_instruction(activation_instruction);

                output_index
            }
            Self::Scale { id, vector } => {
                let output_size = vector.len();
                let output_index = ctx.allocate_buffer(output_size);

                // First copy input to output
                let copy_instruction = InstructionExport::Copy {
                    input: input_indices[0],
                    output: output_index,
                    internal_index: 0,
                };
                ctx.add_instruction(copy_instruction);

                // Store parameters and add MUL_ELEMENTWISE instruction
                let param_index = ctx.get_or_store_parameters(*id, vector);
                let mul_instruction = InstructionExport::MulElementwise {
                    input: output_index,
                    parameters: param_index,
                };
                ctx.add_instruction(mul_instruction);

                output_index
            }
            Self::Shift { id, vector } => {
                let output_size = vector.len();
                let output_index = ctx.allocate_buffer(output_size);

                // First copy input to output
                let copy_instruction = InstructionExport::Copy {
                    input: input_indices[0],
                    output: output_index,
                    internal_index: 0,
                };
                ctx.add_instruction(copy_instruction);

                // Store parameters and add ADD_ELEMENTWISE instruction
                let param_index = ctx.get_or_store_parameters(*id, vector);
                let add_instruction = InstructionExport::AddElementwise {
                    input: output_index,
                    parameters: param_index,
                };
                ctx.add_instruction(add_instruction);

                output_index
            }
            Self::BatchNorm { .. } => {
                panic!(
                    "BatchNorm must be compiled with compile_batch_norm to include trained parameters"
                );
            }
        }
    }

    /// Compiles BatchNorm operation with trained parameters.
    /// Parameters: (center, std, beta) where:
    /// - center = -running_mean
    /// - std = gamma / sqrt(running_variance + epsilon)
    /// - beta = learned bias
    pub fn compile_batch_norm(
        &self,
        input_indices: &[usize],
        ctx: &mut CompileContext,
        center: &[f32],
        std: &[f32],
        beta: &[f32],
    ) -> usize {
        let Self::BatchNorm { id, .. } = self else {
            panic!("compile_batch_norm called on non-BatchNorm operation");
        };

        let output_size = ctx.buffer_sizes[input_indices[0]];
        let output_index = ctx.allocate_buffer(output_size);

        // First copy input to output
        let copy_instruction = InstructionExport::Copy {
            input: input_indices[0],
            output: output_index,
            internal_index: 0,
        };
        ctx.add_instruction(copy_instruction);

        // 1. ADD_ELEMENTWISE: x + center (i.e., x - mean)
        let center_param_index = ctx.get_or_store_parameters(*id, center);
        let center_instruction = InstructionExport::AddElementwise {
            input: output_index,
            parameters: center_param_index,
        };
        ctx.add_instruction(center_instruction);

        // 2. MUL_ELEMENTWISE: x * std (i.e., x * gamma / sqrt(var + eps))
        let std_param_index = ctx.get_or_store_parameters(*id + 1_000_000, std);
        let mul_instruction = InstructionExport::MulElementwise {
            input: output_index,
            parameters: std_param_index,
        };
        ctx.add_instruction(mul_instruction);

        // 3. ADD_ELEMENTWISE: x + beta
        let beta_param_index = ctx.get_or_store_parameters(*id + 2_000_000, beta);
        let beta_instruction = InstructionExport::AddElementwise {
            input: output_index,
            parameters: beta_param_index,
        };
        ctx.add_instruction(beta_instruction);

        output_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::buffer::InputBuffer;

    #[test]
    fn test_dense_creation() {
        let dense = Operation::dense(8, Activation::Relu);
        assert!(matches!(dense, Operation::Dense { output_size: 8, .. }));
    }

    #[test]
    fn test_add_creation() {
        let add = Operation::add();
        assert!(matches!(add, Operation::Add { .. }));
    }

    #[test]
    fn test_multiply_creation() {
        let mul = Operation::multiply();
        assert!(matches!(mul, Operation::Multiply { .. }));
    }

    #[test]
    fn test_unique_ids() {
        let op1 = Operation::dense(4, Activation::None);
        let op2 = Operation::add();
        let op3 = Operation::multiply();

        assert_ne!(op1.id(), op2.id());
        assert_ne!(op2.id(), op3.id());
    }

    #[test]
    fn test_output_size() {
        let dense = Operation::dense(8, Activation::Relu);
        assert_eq!(dense.output_size(&[4]), 8);

        let add = Operation::add();
        assert_eq!(add.output_size(&[4, 4]), 4);
    }

    #[test]
    fn test_convenience_dense() {
        let input = InputBuffer::new(4);
        let output = ops::dense(8, Activation::Relu, input.buffer());
        assert_eq!(output.size(), 8);
        assert!(output.producer().is_some());
    }

    #[test]
    fn test_convenience_add() {
        let input1 = InputBuffer::new(4);
        let input2 = InputBuffer::new(4);
        let output = ops::add(vec![input1.buffer(), input2.buffer()]);
        assert_eq!(output.size(), 4);
    }

    #[test]
    fn test_convenience_multiply() {
        let input1 = InputBuffer::new(4);
        let input2 = InputBuffer::new(4);
        let output = ops::multiply(vec![input1.buffer(), input2.buffer()]);
        assert_eq!(output.size(), 4);
    }

    #[test]
    #[should_panic(expected = "Add requires at least 2 inputs")]
    fn test_add_requires_two_inputs() {
        let input = InputBuffer::new(4);
        ops::add(vec![input.buffer()]);
    }

    #[test]
    #[should_panic(expected = "All inputs to Add must have the same size")]
    fn test_add_requires_same_size() {
        let input1 = InputBuffer::new(4);
        let input2 = InputBuffer::new(5);
        ops::add(vec![input1.buffer(), input2.buffer()]);
    }

    #[test]
    fn test_concat_creation() {
        let concat = Operation::concat();
        assert!(matches!(concat, Operation::Concat { .. }));
    }

    #[test]
    fn test_softmax_creation() {
        let softmax = Operation::softmax();
        assert!(matches!(softmax, Operation::Softmax { .. }));
    }

    #[test]
    fn test_convenience_concat() {
        let input1 = InputBuffer::new(4);
        let input2 = InputBuffer::new(3);
        let output = ops::concat(vec![input1.buffer(), input2.buffer()]);
        assert_eq!(output.size(), 7);
    }

    #[test]
    fn test_convenience_softmax() {
        let input = InputBuffer::new(4);
        let output = ops::softmax(input.buffer());
        assert_eq!(output.size(), 4);
    }

    #[test]
    fn test_concat_output_size() {
        let concat = Operation::concat();
        assert_eq!(concat.output_size(&[4, 3, 2]), 9);
    }

    #[test]
    fn test_softmax_output_size() {
        let softmax = Operation::softmax();
        assert_eq!(softmax.output_size(&[10]), 10);
    }

    #[test]
    #[should_panic(expected = "Concat requires at least 1 input")]
    fn test_concat_requires_input() {
        ops::concat(vec![]);
    }

    #[test]
    fn test_scale_creation() {
        let scale = Operation::scale(vec![1.0, 2.0, 3.0]);
        assert!(matches!(scale, Operation::Scale { vector, .. } if vector.len() == 3));
    }

    #[test]
    fn test_shift_creation() {
        let shift = Operation::shift(vec![0.1, 0.2, 0.3]);
        assert!(matches!(shift, Operation::Shift { vector, .. } if vector.len() == 3));
    }

    #[test]
    fn test_scale_output_size() {
        let scale = Operation::scale(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(scale.output_size(&[4]), 4);
    }

    #[test]
    fn test_shift_output_size() {
        let shift = Operation::shift(vec![0.1, 0.2, 0.3]);
        assert_eq!(shift.output_size(&[3]), 3);
    }

    #[test]
    fn test_convenience_scale() {
        let input = InputBuffer::new(3);
        let output = ops::scale(vec![1.0, 2.0, 3.0], input.buffer());
        assert_eq!(output.size(), 3);
        assert!(output.producer().is_some());
    }

    #[test]
    fn test_convenience_shift() {
        let input = InputBuffer::new(3);
        let output = ops::shift(vec![0.1, 0.2, 0.3], input.buffer());
        assert_eq!(output.size(), 3);
        assert!(output.producer().is_some());
    }

    #[test]
    #[should_panic(expected = "Scale vector length must match input size")]
    fn test_scale_requires_matching_size() {
        let input = InputBuffer::new(4);
        ops::scale(vec![1.0, 2.0, 3.0], input.buffer());
    }

    #[test]
    #[should_panic(expected = "Shift vector length must match input size")]
    fn test_shift_requires_matching_size() {
        let input = InputBuffer::new(4);
        ops::shift(vec![0.1, 0.2], input.buffer());
    }
}
