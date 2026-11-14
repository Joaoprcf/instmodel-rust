use std::sync::atomic::{AtomicUsize, Ordering};

use crate::layers::Activation;

use super::buffer::DataBuffer;
use super::compile::{CompileContext, InstructionExport};

/// Unique identifier for an operation; applying the same `Operation` twice shares its weights.
pub type OpId = usize;

static OP_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn next_op_id() -> OpId {
    OP_ID_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// A computation operation in the graph. `in_place` ops mutate their input buffer instead of copying it.
#[derive(Debug, Clone)]
pub enum Operation {
    Dense {
        id: OpId,
        output_size: usize,
        activation: Activation,
    },
    Add {
        id: OpId,
    },
    Multiply {
        id: OpId,
    },
    Concat {
        id: OpId,
    },
    Softmax {
        id: OpId,
    },
    /// Element-wise multiply by a fixed vector.
    Scale {
        id: OpId,
        vector: Vec<f32>,
        in_place: bool,
    },
    /// Element-wise add a fixed vector.
    Shift {
        id: OpId,
        vector: Vec<f32>,
        in_place: bool,
    },
    /// `center`: learn beta (else 0); `scale`: learn gamma (else 1).
    BatchNorm {
        id: OpId,
        epsilon: f32,
        center: bool,
        scale: bool,
        in_place: bool,
    },
    StandaloneActivation {
        id: OpId,
        activation: Activation,
        in_place: bool,
    },
}

impl Operation {
    pub fn dense(output_size: usize, activation: Activation) -> Self {
        Self::Dense {
            id: next_op_id(),
            output_size,
            activation,
        }
    }

    pub fn add() -> Self {
        Self::Add { id: next_op_id() }
    }

    pub fn multiply() -> Self {
        Self::Multiply { id: next_op_id() }
    }

    pub fn concat() -> Self {
        Self::Concat { id: next_op_id() }
    }

    pub fn softmax() -> Self {
        Self::Softmax { id: next_op_id() }
    }

    pub fn scale(vector: Vec<f32>) -> Self {
        Self::scale_with_inplace(vector, false)
    }

    pub fn scale_with_inplace(vector: Vec<f32>, in_place: bool) -> Self {
        Self::Scale {
            id: next_op_id(),
            vector,
            in_place,
        }
    }

    pub fn shift(vector: Vec<f32>) -> Self {
        Self::shift_with_inplace(vector, false)
    }

    pub fn shift_with_inplace(vector: Vec<f32>, in_place: bool) -> Self {
        Self::Shift {
            id: next_op_id(),
            vector,
            in_place,
        }
    }

    pub fn batch_norm(epsilon: f32, center: bool, scale: bool) -> Self {
        Self::batch_norm_with_inplace(epsilon, center, scale, false)
    }

    pub fn batch_norm_with_inplace(
        epsilon: f32,
        center: bool,
        scale: bool,
        in_place: bool,
    ) -> Self {
        Self::BatchNorm {
            id: next_op_id(),
            epsilon,
            center,
            scale,
            in_place,
        }
    }

    /// epsilon = 1e-3, center and scale enabled.
    pub fn batch_norm_default() -> Self {
        Self::batch_norm(1e-3, true, true)
    }

    pub fn batch_norm_scale_only(epsilon: f32) -> Self {
        Self::batch_norm(epsilon, false, true)
    }

    pub fn standalone_activation(activation: Activation) -> Self {
        Self::standalone_activation_with_inplace(activation, false)
    }

    pub fn standalone_activation_with_inplace(activation: Activation, in_place: bool) -> Self {
        Self::StandaloneActivation {
            id: next_op_id(),
            activation,
            in_place,
        }
    }

    pub fn id(&self) -> OpId {
        match self {
            Self::Dense { id, .. }
            | Self::Add { id }
            | Self::Multiply { id }
            | Self::Concat { id }
            | Self::Softmax { id }
            | Self::Scale { id, .. }
            | Self::Shift { id, .. }
            | Self::BatchNorm { id, .. }
            | Self::StandaloneActivation { id, .. } => *id,
        }
    }

    pub fn output_size(&self, input_sizes: &[usize]) -> usize {
        match self {
            Self::Dense { output_size, .. } => *output_size,
            Self::Concat { .. } => input_sizes.iter().sum(),
            Self::Scale { vector, .. } | Self::Shift { vector, .. } => vector.len(),
            Self::Add { .. }
            | Self::Multiply { .. }
            | Self::Softmax { .. }
            | Self::BatchNorm { .. }
            | Self::StandaloneActivation { .. } => input_sizes[0],
        }
    }

    pub fn apply(&self, input: DataBuffer) -> DataBuffer {
        self.apply_multi(vec![input])
    }

    pub fn apply_multi(&self, inputs: Vec<DataBuffer>) -> DataBuffer {
        let sizes: Vec<usize> = inputs.iter().map(DataBuffer::size).collect();
        DataBuffer::new(self.output_size(&sizes), Some(self.clone()), inputs)
    }

    /// Emits instructions for this op and returns its output buffer index.
    /// Dense gets a placeholder weight slot; BatchNorm must use [`Self::compile_batch_norm`].
    pub fn compile(&self, input_indices: &[usize], ctx: &mut CompileContext) -> usize {
        let input = input_indices[0];
        match self {
            Self::Dense {
                id,
                output_size,
                activation,
            } => {
                let weights = ctx.get_or_create_weight_index(*id);
                let output = ctx.allocate_buffer(*output_size);
                ctx.add_instruction(InstructionExport::Dot {
                    input,
                    output,
                    weights,
                    activation: activation.to_instruction_name().map(String::from),
                });
                output
            }
            Self::Add { .. } | Self::Multiply { .. } => {
                let output = ctx.allocate_buffer(ctx.buffer_sizes[input]);
                let input = input_indices.to_vec();
                ctx.add_instruction(if matches!(self, Self::Add { .. }) {
                    InstructionExport::AddBuffers { input, output }
                } else {
                    InstructionExport::MultiplyBuffers { input, output }
                });
                output
            }
            Self::Concat { .. } => {
                let total = input_indices.iter().map(|&i| ctx.buffer_sizes[i]).sum();
                let output = ctx.allocate_buffer(total);
                let mut offset = 0;
                for &input in input_indices {
                    ctx.add_instruction(InstructionExport::Copy {
                        input,
                        output,
                        internal_index: offset,
                    });
                    offset += ctx.buffer_sizes[input];
                }
                output
            }
            Self::Softmax { .. } => {
                let target = copy_target(ctx, input, ctx.buffer_sizes[input], false);
                ctx.add_instruction(InstructionExport::Activation {
                    input: target,
                    activation: "SOFTMAX".to_string(),
                });
                target
            }
            Self::Scale {
                id,
                vector,
                in_place,
            }
            | Self::Shift {
                id,
                vector,
                in_place,
            } => {
                let target = copy_target(ctx, input, vector.len(), *in_place);
                let parameters = ctx.get_or_store_parameters(*id, vector);
                ctx.add_instruction(if matches!(self, Self::Scale { .. }) {
                    InstructionExport::MulElementwise {
                        input: target,
                        parameters,
                    }
                } else {
                    InstructionExport::AddElementwise {
                        input: target,
                        parameters,
                    }
                });
                target
            }
            Self::BatchNorm { .. } => {
                panic!(
                    "BatchNorm must be compiled with compile_batch_norm to include trained parameters"
                );
            }
            Self::StandaloneActivation {
                activation,
                in_place,
                ..
            } => {
                let target = copy_target(ctx, input, ctx.buffer_sizes[input], *in_place);
                ctx.add_instruction(InstructionExport::Activation {
                    input: target,
                    activation: activation
                        .to_instruction_name()
                        .unwrap_or("NONE")
                        .to_string(),
                });
                target
            }
        }
    }

    /// Emits BatchNorm as ADD(center) → MUL(std) → ADD(beta), where center = -running_mean,
    /// std = gamma / sqrt(running_var + eps) and beta is the learned shift.
    pub fn compile_batch_norm(
        &self,
        input_indices: &[usize],
        ctx: &mut CompileContext,
        center: &[f32],
        std: &[f32],
        beta: &[f32],
    ) -> usize {
        let Self::BatchNorm { id, in_place, .. } = self else {
            panic!("compile_batch_norm called on non-BatchNorm operation");
        };
        let input = input_indices[0];
        let target = copy_target(ctx, input, ctx.buffer_sizes[input], *in_place);

        // The three vectors are stored contiguously under this op's id, so a shared BatchNorm
        // reuses them and they can never collide with another op's parameters.
        let first = match ctx.visited_param_ops.get(id) {
            Some(&idx) => idx,
            None => {
                let idx = ctx.parameters.len();
                ctx.parameters
                    .extend([center.to_vec(), std.to_vec(), beta.to_vec()]);
                ctx.visited_param_ops.insert(*id, idx);
                idx
            }
        };
        for (offset, mul) in [(0, false), (1, true), (2, false)] {
            let parameters = first + offset;
            ctx.add_instruction(if mul {
                InstructionExport::MulElementwise {
                    input: target,
                    parameters,
                }
            } else {
                InstructionExport::AddElementwise {
                    input: target,
                    parameters,
                }
            });
        }
        target
    }
}

/// Returns the buffer an elementwise op writes to: the input itself, or a fresh COPY of it.
fn copy_target(ctx: &mut CompileContext, input: usize, size: usize, in_place: bool) -> usize {
    if in_place {
        return input;
    }
    let output = ctx.allocate_buffer(size);
    ctx.add_instruction(InstructionExport::Copy {
        input,
        output,
        internal_index: 0,
    });
    output
}

/// One-step create-and-apply helpers mirroring the Python API.
pub mod ops {
    use super::*;

    fn assert_same_sizes(name: &str, inputs: &[DataBuffer]) {
        assert!(inputs.len() >= 2, "{name} requires at least 2 inputs");
        assert!(
            inputs.iter().all(|b| b.size() == inputs[0].size()),
            "All inputs to {name} must have the same size"
        );
    }

    fn assert_vector_len(name: &str, vector: &[f32], input: &DataBuffer) {
        assert_eq!(
            vector.len(),
            input.size(),
            "{name} vector length must match input size"
        );
    }

    pub fn dense(output_size: usize, activation: Activation, input: DataBuffer) -> DataBuffer {
        Operation::dense(output_size, activation).apply(input)
    }

    pub fn add(inputs: Vec<DataBuffer>) -> DataBuffer {
        assert_same_sizes("Add", &inputs);
        Operation::add().apply_multi(inputs)
    }

    pub fn multiply(inputs: Vec<DataBuffer>) -> DataBuffer {
        assert_same_sizes("Multiply", &inputs);
        Operation::multiply().apply_multi(inputs)
    }

    pub fn concat(inputs: Vec<DataBuffer>) -> DataBuffer {
        assert!(!inputs.is_empty(), "Concat requires at least 1 input");
        Operation::concat().apply_multi(inputs)
    }

    pub fn softmax(input: DataBuffer) -> DataBuffer {
        Operation::softmax().apply(input)
    }

    pub fn scale(vector: Vec<f32>, input: DataBuffer) -> DataBuffer {
        assert_vector_len("Scale", &vector, &input);
        Operation::scale(vector).apply(input)
    }

    pub fn shift(vector: Vec<f32>, input: DataBuffer) -> DataBuffer {
        assert_vector_len("Shift", &vector, &input);
        Operation::shift(vector).apply(input)
    }

    /// Overwrites the input buffer at inference; don't reuse the input afterwards.
    pub fn scale_inplace(vector: Vec<f32>, input: DataBuffer) -> DataBuffer {
        assert_vector_len("Scale", &vector, &input);
        Operation::scale_with_inplace(vector, true).apply(input)
    }

    /// Overwrites the input buffer at inference; don't reuse the input afterwards.
    pub fn shift_inplace(vector: Vec<f32>, input: DataBuffer) -> DataBuffer {
        assert_vector_len("Shift", &vector, &input);
        Operation::shift_with_inplace(vector, true).apply(input)
    }

    /// Exported as element-wise ADD/MUL/ADD using the running statistics.
    pub fn batch_norm(input: DataBuffer) -> DataBuffer {
        Operation::batch_norm_default().apply(input)
    }

    pub fn batch_norm_with_epsilon(epsilon: f32, input: DataBuffer) -> DataBuffer {
        Operation::batch_norm(epsilon, true, true).apply(input)
    }

    pub fn batch_norm_scale_only(input: DataBuffer) -> DataBuffer {
        Operation::batch_norm_scale_only(1e-3).apply(input)
    }

    pub fn batch_norm_config(
        epsilon: f32,
        center: bool,
        scale: bool,
        input: DataBuffer,
    ) -> DataBuffer {
        Operation::batch_norm(epsilon, center, scale).apply(input)
    }

    /// Overwrites the input buffer at inference; don't reuse the input afterwards.
    pub fn batch_norm_inplace(input: DataBuffer) -> DataBuffer {
        Operation::batch_norm_with_inplace(1e-3, true, true, true).apply(input)
    }

    /// Overwrites the input buffer at inference; don't reuse the input afterwards.
    pub fn batch_norm_config_inplace(
        epsilon: f32,
        center: bool,
        scale: bool,
        input: DataBuffer,
    ) -> DataBuffer {
        Operation::batch_norm_with_inplace(epsilon, center, scale, true).apply(input)
    }

    pub fn activation(act: Activation, input: DataBuffer) -> DataBuffer {
        Operation::standalone_activation(act).apply(input)
    }

    /// Overwrites the input buffer at inference; don't reuse the input afterwards.
    pub fn activation_inplace(act: Activation, input: DataBuffer) -> DataBuffer {
        Operation::standalone_activation_with_inplace(act, true).apply(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::buffer::InputBuffer;

    #[test]
    fn output_sizes() {
        let input = || InputBuffer::new(4).buffer();
        assert_eq!(ops::dense(8, Activation::Relu, input()).size(), 8);
        assert_eq!(ops::add(vec![input(), input()]).size(), 4);
        assert_eq!(ops::multiply(vec![input(), input()]).size(), 4);
        assert_eq!(
            ops::concat(vec![input(), InputBuffer::new(3).buffer()]).size(),
            7
        );
        assert_eq!(ops::softmax(input()).size(), 4);
        assert_eq!(ops::scale(vec![1.0; 4], input()).size(), 4);
        assert_eq!(ops::shift_inplace(vec![1.0; 4], input()).size(), 4);
        assert_eq!(ops::batch_norm(input()).size(), 4);
        assert_eq!(ops::activation(Activation::Gelu, input()).size(), 4);
        assert_ne!(Operation::add().id(), Operation::add().id());
    }

    #[test]
    #[should_panic(expected = "Add requires at least 2 inputs")]
    fn add_requires_two_inputs() {
        ops::add(vec![InputBuffer::new(4).buffer()]);
    }

    #[test]
    #[should_panic(expected = "All inputs to Multiply must have the same size")]
    fn multiply_requires_same_size() {
        ops::multiply(vec![
            InputBuffer::new(4).buffer(),
            InputBuffer::new(5).buffer(),
        ]);
    }

    #[test]
    #[should_panic(expected = "Concat requires at least 1 input")]
    fn concat_requires_input() {
        ops::concat(vec![]);
    }

    #[test]
    #[should_panic(expected = "Shift vector length must match input size")]
    fn shift_requires_matching_size() {
        ops::shift(vec![0.1, 0.2], InputBuffer::new(4).buffer());
    }

    #[test]
    fn batch_norm_params_do_not_collide_with_other_ops() {
        let bn = Operation::batch_norm_default();
        let mut ctx = CompileContext::new();
        ctx.register_input(0, 2, None);
        // Previously BatchNorm keyed its std/beta vectors on id + 1_000_000 / 2_000_000.
        ctx.get_or_store_parameters(bn.id() + 1_000_000, &[9.0, 9.0]);
        bn.compile_batch_norm(&[0], &mut ctx, &[1.0; 2], &[2.0; 2], &[3.0; 2]);
        bn.compile_batch_norm(&[0], &mut ctx, &[7.0; 2], &[7.0; 2], &[7.0; 2]);
        assert_eq!(
            ctx.parameters,
            vec![vec![9.0; 2], vec![1.0; 2], vec![2.0; 2], vec![3.0; 2]]
        );
        let used: Vec<_> = ctx
            .instructions
            .iter()
            .map(|i| serde_json::to_value(i).unwrap()["parameters"].as_u64())
            .collect();
        let expected = [None, Some(1), Some(2), Some(3)];
        assert_eq!(used, [expected, expected].concat());
    }
}
