//! ModelGraph - the main container for functional neural networks.

use std::collections::HashMap;

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Tensor, backend::Backend};

use crate::errors::ModelError;
use crate::layers::Activation;

use super::buffer::{BufferId, DataBuffer, InputBuffer};
use super::compile::{CompileContext, InstructionModelExport};
use super::operation::{OpId, Operation};

/// Initialized Dense layer with Burn's Linear module.
#[derive(Module, Debug)]
struct DenseLayer<B: Backend> {
    linear: Linear<B>,
    output_size: usize,
    input_size: usize,
    activation_id: u8,
}

impl<B: Backend> DenseLayer<B> {
    fn new(
        input_size: usize,
        output_size: usize,
        activation: Activation,
        device: &B::Device,
    ) -> Self {
        let linear = LinearConfig::new(input_size, output_size).init(device);
        Self {
            linear,
            output_size,
            input_size,
            activation_id: activation.to_id(),
        }
    }

    fn activation(&self) -> Activation {
        Activation::from_id(self.activation_id)
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let output = self.linear.forward(input);
        self.activation().apply(output)
    }

    fn weights_to_vec(&self) -> Vec<Vec<f32>> {
        let weight_tensor = self.linear.weight.val();
        let weights_data = weight_tensor.to_data();
        let flat: Vec<f32> = weights_data.to_vec().unwrap();

        let mut result = vec![vec![0.0f32; self.input_size]; self.output_size];
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                result[i][j] = flat[i + j * self.output_size];
            }
        }
        result
    }

    fn bias_to_vec(&self) -> Vec<f32> {
        match &self.linear.bias {
            Some(bias) => bias.val().to_data().to_vec().unwrap(),
            None => vec![0.0; self.output_size],
        }
    }
}

/// Execution step in the forward pass.
#[derive(Debug, Clone)]
enum Step {
    Dense {
        op_id: OpId,
        input: BufferId,
        output: BufferId,
    },
    Add {
        inputs: Vec<BufferId>,
        output: BufferId,
    },
    Multiply {
        inputs: Vec<BufferId>,
        output: BufferId,
    },
    Concat {
        inputs: Vec<BufferId>,
        output: BufferId,
    },
    Softmax {
        input: BufferId,
        output: BufferId,
    },
    Scale {
        input: BufferId,
        output: BufferId,
        vector: Vec<f32>,
    },
    Shift {
        input: BufferId,
        output: BufferId,
        vector: Vec<f32>,
    },
}

/// ModelGraph holds the computation graph and initialized layers.
#[derive(Debug)]
pub struct ModelGraph<B: Backend> {
    inputs: Vec<InputBuffer>,
    output: DataBuffer,
    dense_layers: HashMap<OpId, DenseLayer<B>>,
    steps: Vec<Step>,
    buffer_index: HashMap<BufferId, usize>,
    features: Vec<String>,
}

impl<B: Backend> ModelGraph<B> {
    /// Creates a new ModelGraph from input buffers and output buffer.
    pub fn new(
        inputs: Vec<InputBuffer>,
        output: DataBuffer,
        device: &B::Device,
    ) -> Result<Self, ModelError> {
        if inputs.is_empty() {
            return Err(ModelError::NoInputBuffer);
        }

        let mut builder = GraphBuilder::new();

        for input in &inputs {
            builder.register_input(input);
        }

        builder.traverse(&output)?;

        let dense_layers = builder.init_dense_layers(device);

        let features: Vec<String> = inputs
            .iter()
            .filter_map(|i| i.features())
            .flatten()
            .cloned()
            .collect();

        Ok(Self {
            inputs,
            output,
            dense_layers,
            steps: builder.steps,
            buffer_index: builder.buffer_index,
            features,
        })
    }

    /// Performs forward pass through the model.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut buffers: HashMap<usize, Tensor<B, 2>> = HashMap::new();

        let input_idx = self.buffer_index[&self.inputs[0].id()];
        buffers.insert(input_idx, input);

        for step in &self.steps {
            match step {
                Step::Dense {
                    op_id,
                    input,
                    output,
                } => {
                    let in_idx = self.buffer_index[input];
                    let out_idx = self.buffer_index[output];
                    let layer = &self.dense_layers[op_id];
                    let result = layer.forward(buffers[&in_idx].clone());
                    buffers.insert(out_idx, result);
                }
                Step::Add { inputs, output } => {
                    let out_idx = self.buffer_index[output];
                    let mut result = buffers[&self.buffer_index[&inputs[0]]].clone();
                    for buf_id in &inputs[1..] {
                        result = result.add(buffers[&self.buffer_index[buf_id]].clone());
                    }
                    buffers.insert(out_idx, result);
                }
                Step::Multiply { inputs, output } => {
                    let out_idx = self.buffer_index[output];
                    let mut result = buffers[&self.buffer_index[&inputs[0]]].clone();
                    for buf_id in &inputs[1..] {
                        result = result.mul(buffers[&self.buffer_index[buf_id]].clone());
                    }
                    buffers.insert(out_idx, result);
                }
                Step::Concat { inputs, output } => {
                    let out_idx = self.buffer_index[output];
                    let tensors: Vec<Tensor<B, 2>> = inputs
                        .iter()
                        .map(|buf_id| buffers[&self.buffer_index[buf_id]].clone())
                        .collect();
                    let result = Tensor::cat(tensors, 1);
                    buffers.insert(out_idx, result);
                }
                Step::Softmax { input, output } => {
                    let in_idx = self.buffer_index[input];
                    let out_idx = self.buffer_index[output];
                    let result = burn::tensor::activation::softmax(buffers[&in_idx].clone(), 1);
                    buffers.insert(out_idx, result);
                }
                Step::Scale {
                    input,
                    output,
                    vector,
                } => {
                    let in_idx = self.buffer_index[input];
                    let out_idx = self.buffer_index[output];
                    let in_tensor = buffers[&in_idx].clone();
                    let device = in_tensor.device();
                    let scale_tensor =
                        Tensor::<B, 1>::from_floats(vector.as_slice(), &device).reshape([1, -1]);
                    let result = in_tensor.mul(scale_tensor);
                    buffers.insert(out_idx, result);
                }
                Step::Shift {
                    input,
                    output,
                    vector,
                } => {
                    let in_idx = self.buffer_index[input];
                    let out_idx = self.buffer_index[output];
                    let in_tensor = buffers[&in_idx].clone();
                    let device = in_tensor.device();
                    let shift_tensor =
                        Tensor::<B, 1>::from_floats(vector.as_slice(), &device).reshape([1, -1]);
                    let result = in_tensor.add(shift_tensor);
                    buffers.insert(out_idx, result);
                }
            }
        }

        let output_idx = self.buffer_index[&self.output.id()];
        buffers.remove(&output_idx).unwrap()
    }

    /// Returns the input feature names.
    pub fn features(&self) -> &[String] {
        &self.features
    }

    /// Returns the number of input features.
    pub fn feature_size(&self) -> usize {
        self.inputs.iter().map(|i| i.size()).sum()
    }

    /// Returns the output size of the model.
    pub fn output_size(&self) -> usize {
        self.output.size()
    }

    /// Exports the model to instruction model JSON format.
    pub fn export_to_instruction_model(&self) -> Result<String, serde_json::Error> {
        let export = self.to_instruction_model_info();
        serde_json::to_string_pretty(&export)
    }

    /// Converts the model to an InstructionModelExport struct.
    pub fn to_instruction_model_info(&self) -> InstructionModelExport {
        let mut ctx = CompileContext::new();

        for input in &self.inputs {
            ctx.register_input(input.id(), input.size(), input.features());
        }

        self.compile_buffer(&self.output, &mut ctx);

        ctx.into_export()
    }

    /// Recursively compiles a buffer and its dependencies.
    fn compile_buffer(&self, buffer: &DataBuffer, ctx: &mut CompileContext) -> usize {
        if let Some(&idx) = ctx.visited_buffers.get(&buffer.id()) {
            return idx;
        }

        let Some(producer) = buffer.producer() else {
            return ctx.visited_buffers[&buffer.id()];
        };

        let input_indices: Vec<usize> = buffer
            .inputs()
            .iter()
            .map(|inp| self.compile_buffer(inp, ctx))
            .collect();

        // Fill weights for Dense operations
        if let Operation::Dense { id, .. } = producer {
            if let Some(layer) = self.dense_layers.get(id) {
                let weight_idx = ctx.get_or_create_weight_index(*id);
                ctx.set_weights(weight_idx, layer.weights_to_vec(), layer.bias_to_vec());
            }
        }

        let output_idx = producer.compile(&input_indices, ctx);
        ctx.visited_buffers.insert(buffer.id(), output_idx);

        output_idx
    }
}

/// Builder for constructing ModelGraph execution order.
struct GraphBuilder {
    buffer_index: HashMap<BufferId, usize>,
    next_idx: usize,
    steps: Vec<Step>,
    dense_info: HashMap<OpId, (usize, usize, Activation)>,
    visited: HashMap<BufferId, bool>,
}

impl GraphBuilder {
    fn new() -> Self {
        Self {
            buffer_index: HashMap::new(),
            next_idx: 0,
            steps: Vec::new(),
            dense_info: HashMap::new(),
            visited: HashMap::new(),
        }
    }

    fn register_input(&mut self, input: &InputBuffer) {
        self.buffer_index.insert(input.id(), self.next_idx);
        self.next_idx += 1;
        self.visited.insert(input.id(), true);
    }

    fn traverse(&mut self, buffer: &DataBuffer) -> Result<(), ModelError> {
        if self.visited.contains_key(&buffer.id()) {
            return Ok(());
        }

        let Some(producer) = buffer.producer() else {
            return Err(ModelError::InvalidGraph(
                "Buffer has no producer and is not a registered input".to_string(),
            ));
        };

        for input in buffer.inputs() {
            self.traverse(input)?;
        }

        self.buffer_index.insert(buffer.id(), self.next_idx);
        self.next_idx += 1;

        let inputs = buffer.inputs();
        let input_size = inputs[0].size();

        match producer {
            Operation::Dense {
                id,
                output_size,
                activation,
            } => {
                if !self.dense_info.contains_key(id) {
                    self.dense_info
                        .insert(*id, (input_size, *output_size, *activation));
                }
                self.steps.push(Step::Dense {
                    op_id: *id,
                    input: inputs[0].id(),
                    output: buffer.id(),
                });
            }
            Operation::Add { .. } => {
                let input_ids: Vec<BufferId> = inputs.iter().map(|b| b.id()).collect();
                self.steps.push(Step::Add {
                    inputs: input_ids,
                    output: buffer.id(),
                });
            }
            Operation::Multiply { .. } => {
                let input_ids: Vec<BufferId> = inputs.iter().map(|b| b.id()).collect();
                self.steps.push(Step::Multiply {
                    inputs: input_ids,
                    output: buffer.id(),
                });
            }
            Operation::Concat { .. } => {
                let input_ids: Vec<BufferId> = inputs.iter().map(|b| b.id()).collect();
                self.steps.push(Step::Concat {
                    inputs: input_ids,
                    output: buffer.id(),
                });
            }
            Operation::Softmax { .. } => {
                self.steps.push(Step::Softmax {
                    input: inputs[0].id(),
                    output: buffer.id(),
                });
            }
            Operation::Scale { vector, .. } => {
                self.steps.push(Step::Scale {
                    input: inputs[0].id(),
                    output: buffer.id(),
                    vector: vector.clone(),
                });
            }
            Operation::Shift { vector, .. } => {
                self.steps.push(Step::Shift {
                    input: inputs[0].id(),
                    output: buffer.id(),
                    vector: vector.clone(),
                });
            }
        }

        self.visited.insert(buffer.id(), true);
        Ok(())
    }

    fn init_dense_layers<B: Backend>(&self, device: &B::Device) -> HashMap<OpId, DenseLayer<B>> {
        self.dense_info
            .iter()
            .map(|(&op_id, &(input_size, output_size, activation))| {
                (
                    op_id,
                    DenseLayer::new(input_size, output_size, activation, device),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::operation::ops;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_simple_model() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let output = ops::dense(1, Activation::Sigmoid, input.buffer());

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        assert_eq!(model.feature_size(), 4);
        assert_eq!(model.output_size(), 1);
    }

    #[test]
    fn test_multilayer_model() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = ops::dense(8, Activation::Relu, input.buffer());
        let output = ops::dense(1, Activation::Sigmoid, x);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        assert_eq!(model.feature_size(), 4);
        assert_eq!(model.output_size(), 1);
    }

    #[test]
    fn test_forward_pass() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(2);
        let output = ops::dense(1, Activation::None, input.buffer());

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        let input_tensor = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0]], &device);
        let output_tensor = model.forward(input_tensor);

        assert_eq!(output_tensor.dims(), [1, 1]);
    }

    #[test]
    fn test_add_operation() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();

        let branch1 = ops::dense(4, Activation::Relu, x.clone());
        let branch2 = ops::dense(4, Activation::Relu, x);

        let added = ops::add(vec![branch1, branch2]);
        let output = ops::dense(1, Activation::Sigmoid, added);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        let input_tensor = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let output_tensor = model.forward(input_tensor);
        assert_eq!(output_tensor.dims(), [1, 1]);
    }

    #[test]
    fn test_multiply_operation() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();

        let branch1 = ops::dense(4, Activation::Relu, x.clone());
        let branch2 = ops::dense(4, Activation::Relu, x);

        let multiplied = ops::multiply(vec![branch1, branch2]);
        let output = ops::dense(1, Activation::Sigmoid, multiplied);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        let input_tensor = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let output_tensor = model.forward(input_tensor);
        assert_eq!(output_tensor.dims(), [1, 1]);
    }

    #[test]
    fn test_export() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();

        let branch1 = ops::dense(4, Activation::Relu, x.clone());
        let branch2 = ops::dense(4, Activation::Relu, x);

        let added = ops::add(vec![branch1, branch2]);
        let output = ops::dense(1, Activation::Sigmoid, added);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        let export = model.to_instruction_model_info();

        assert_eq!(export.instructions.len(), 4);
    }

    #[test]
    fn test_concat_operation() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();

        let branch1 = ops::dense(3, Activation::Relu, x.clone());
        let branch2 = ops::dense(2, Activation::Relu, x);

        let concatenated = ops::concat(vec![branch1, branch2]);
        let output = ops::dense(1, Activation::Sigmoid, concatenated);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        assert_eq!(model.output_size(), 1);

        let input_tensor = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let output_tensor = model.forward(input_tensor);
        assert_eq!(output_tensor.dims(), [1, 1]);
    }

    #[test]
    fn test_softmax_operation() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = ops::dense(3, Activation::None, input.buffer());
        let output = ops::softmax(x);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        assert_eq!(model.output_size(), 3);

        let input_tensor = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let output_tensor = model.forward(input_tensor);
        assert_eq!(output_tensor.dims(), [1, 3]);

        let output_data: Vec<f32> = output_tensor.to_data().to_vec().unwrap();
        let sum: f32 = output_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax output should sum to 1");
    }

    #[test]
    fn test_concat_export() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();

        let branch1 = ops::dense(3, Activation::Relu, x.clone());
        let branch2 = ops::dense(2, Activation::Relu, x);

        let concatenated = ops::concat(vec![branch1, branch2]);
        let output = ops::dense(1, Activation::Sigmoid, concatenated);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        let export = model.to_instruction_model_info();

        assert_eq!(export.buffer_sizes[3], 5);
    }

    #[test]
    fn test_softmax_export() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = ops::dense(3, Activation::None, input.buffer());
        let output = ops::softmax(x);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        let json = model
            .export_to_instruction_model()
            .expect("Serialization failed");
        assert!(json.contains("ACTIVATION"));
        assert!(json.contains("SOFTMAX"));
    }

    #[test]
    fn test_scale_operation() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();
        let scaled = ops::scale(vec![2.0, 3.0, 4.0, 5.0], x);
        let output = ops::dense(1, Activation::Sigmoid, scaled);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        let input_tensor = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let output_tensor = model.forward(input_tensor);
        assert_eq!(output_tensor.dims(), [1, 1]);
    }

    #[test]
    fn test_shift_operation() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();
        let shifted = ops::shift(vec![0.1, 0.2, 0.3, 0.4], x);
        let output = ops::dense(1, Activation::Sigmoid, shifted);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        let input_tensor = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let output_tensor = model.forward(input_tensor);
        assert_eq!(output_tensor.dims(), [1, 1]);
    }

    #[test]
    fn test_scale_shift_combined() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();
        let scaled = ops::scale(vec![2.0, 2.0, 2.0, 2.0], x);
        let shifted = ops::shift(vec![1.0, 1.0, 1.0, 1.0], scaled);
        let output = ops::dense(1, Activation::Sigmoid, shifted);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        let input_tensor = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let output_tensor = model.forward(input_tensor);
        assert_eq!(output_tensor.dims(), [1, 1]);
    }

    #[test]
    fn test_scale_export() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();
        let scaled = ops::scale(vec![2.0, 3.0, 4.0, 5.0], x);
        let output = ops::dense(1, Activation::Sigmoid, scaled);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        let json = model
            .export_to_instruction_model()
            .expect("Serialization failed");
        assert!(json.contains("MUL_ELEMENTWISE"));
        assert!(json.contains("parameters"));
    }

    #[test]
    fn test_shift_export() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();
        let shifted = ops::shift(vec![0.1, 0.2, 0.3, 0.4], x);
        let output = ops::dense(1, Activation::Sigmoid, shifted);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        let json = model
            .export_to_instruction_model()
            .expect("Serialization failed");
        assert!(json.contains("ADD_ELEMENTWISE"));
        assert!(json.contains("parameters"));
    }
}
