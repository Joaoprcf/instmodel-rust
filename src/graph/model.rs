//! ModelGraph - the main container for functional neural networks.

use std::collections::HashMap;

use burn::module::Module;
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig};
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

/// Initialized BatchNorm layer with Burn's BatchNorm module.
#[derive(Module, Debug)]
struct BatchNormLayer<B: Backend> {
    batch_norm: BatchNorm<B, 1>,
    num_features: usize,
    epsilon: f32,
}

impl<B: Backend> BatchNormLayer<B> {
    fn new(num_features: usize, epsilon: f32, device: &B::Device) -> Self {
        let batch_norm = BatchNormConfig::new(num_features)
            .with_epsilon(epsilon as f64)
            .init(device);
        Self {
            batch_norm,
            num_features,
            epsilon,
        }
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch, features] = input.dims();
        let input_3d = input.reshape([batch, features, 1]);
        let output_3d = self.batch_norm.forward(input_3d);
        output_3d.reshape([batch, features])
    }

    fn get_normalization_params(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let gamma: Vec<f32> = self.batch_norm.gamma.val().to_data().to_vec().unwrap();
        let beta: Vec<f32> = self.batch_norm.beta.val().to_data().to_vec().unwrap();
        let running_mean: Vec<f32> = self
            .batch_norm
            .running_mean
            .value()
            .to_data()
            .to_vec()
            .unwrap();
        let running_var: Vec<f32> = self
            .batch_norm
            .running_var
            .value()
            .to_data()
            .to_vec()
            .unwrap();

        let epsilon = self.epsilon;

        // center = -moving_mean
        let center: Vec<f32> = running_mean.iter().map(|m| -m).collect();

        // std = gamma / sqrt(variance + epsilon)
        let std: Vec<f32> = gamma
            .iter()
            .zip(running_var.iter())
            .map(|(g, v)| g / (v + epsilon).sqrt())
            .collect();

        (center, std, beta)
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
    BatchNorm {
        op_id: OpId,
        input: BufferId,
        output: BufferId,
    },
}

/// ModelGraph holds the computation graph and initialized layers.
#[derive(Debug)]
pub struct ModelGraph<B: Backend> {
    inputs: Vec<InputBuffer>,
    output: DataBuffer,
    dense_layers: HashMap<OpId, DenseLayer<B>>,
    batch_norm_layers: HashMap<OpId, BatchNormLayer<B>>,
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
        let batch_norm_layers = builder.init_batch_norm_layers(device);

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
            batch_norm_layers,
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
                Step::BatchNorm {
                    op_id,
                    input,
                    output,
                } => {
                    let in_idx = self.buffer_index[input];
                    let out_idx = self.buffer_index[output];
                    let layer = &self.batch_norm_layers[op_id];
                    let result = layer.forward(buffers[&in_idx].clone());
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

        self.fill_dense_weights(producer, ctx);

        let output_idx = self.compile_operation(producer, &input_indices, ctx);

        ctx.visited_buffers.insert(buffer.id(), output_idx);

        output_idx
    }

    fn fill_dense_weights(&self, producer: &Operation, ctx: &mut CompileContext) {
        let Operation::Dense { id, .. } = producer else {
            return;
        };
        let Some(layer) = self.dense_layers.get(id) else {
            return;
        };
        let weight_idx = ctx.get_or_create_weight_index(*id);
        ctx.set_weights(weight_idx, layer.weights_to_vec(), layer.bias_to_vec());
    }

    fn compile_operation(
        &self,
        producer: &Operation,
        input_indices: &[usize],
        ctx: &mut CompileContext,
    ) -> usize {
        let Operation::BatchNorm { id, .. } = producer else {
            return producer.compile(input_indices, ctx);
        };
        let Some(layer) = self.batch_norm_layers.get(id) else {
            return producer.compile(input_indices, ctx);
        };
        let (center, std, beta) = layer.get_normalization_params();
        producer.compile_batch_norm(input_indices, ctx, &center, &std, &beta)
    }
}

/// Builder for constructing ModelGraph execution order.
struct GraphBuilder {
    buffer_index: HashMap<BufferId, usize>,
    next_idx: usize,
    steps: Vec<Step>,
    dense_info: HashMap<OpId, (usize, usize, Activation)>,
    batch_norm_info: HashMap<OpId, (usize, f32)>,
    visited: HashMap<BufferId, bool>,
}

impl GraphBuilder {
    fn new() -> Self {
        Self {
            buffer_index: HashMap::new(),
            next_idx: 0,
            steps: Vec::new(),
            dense_info: HashMap::new(),
            batch_norm_info: HashMap::new(),
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
            Operation::BatchNorm { id, epsilon, .. } => {
                if !self.batch_norm_info.contains_key(id) {
                    self.batch_norm_info.insert(*id, (input_size, *epsilon));
                }
                self.steps.push(Step::BatchNorm {
                    op_id: *id,
                    input: inputs[0].id(),
                    output: buffer.id(),
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

    fn init_batch_norm_layers<B: Backend>(
        &self,
        device: &B::Device,
    ) -> HashMap<OpId, BatchNormLayer<B>> {
        self.batch_norm_info
            .iter()
            .map(|(&op_id, &(num_features, epsilon))| {
                (op_id, BatchNormLayer::new(num_features, epsilon, device))
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

    #[test]
    fn test_batch_norm_operation() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();
        let normalized = ops::batch_norm(x);
        let output = ops::dense(1, Activation::Sigmoid, normalized);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        let input_tensor = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let output_tensor = model.forward(input_tensor);
        assert_eq!(output_tensor.dims(), [1, 1]);
    }

    #[test]
    fn test_batch_norm_with_dense() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();
        let hidden = ops::dense(8, Activation::Relu, x);
        let normalized = ops::batch_norm(hidden);
        let output = ops::dense(1, Activation::Sigmoid, normalized);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        let input_tensor = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let output_tensor = model.forward(input_tensor);
        assert_eq!(output_tensor.dims(), [1, 1]);
    }

    #[test]
    fn test_batch_norm_export() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();
        let normalized = ops::batch_norm(x);
        let output = ops::dense(1, Activation::Sigmoid, normalized);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        let json = model
            .export_to_instruction_model()
            .expect("Serialization failed");

        // BatchNorm should export as: COPY + ADD_ELEMENTWISE + MUL_ELEMENTWISE + ADD_ELEMENTWISE
        assert!(json.contains("ADD_ELEMENTWISE"));
        assert!(json.contains("MUL_ELEMENTWISE"));
        assert!(json.contains("parameters"));
    }

    #[test]
    fn test_batch_norm_layer_params_computation() {
        use super::BatchNormLayer;

        let device = <TestBackend as Backend>::Device::default();
        let epsilon = 1e-3f32;
        let layer = BatchNormLayer::<TestBackend>::new(4, epsilon, &device);

        let (center, std, beta) = layer.get_normalization_params();

        // Initial state: gamma=1, beta=0, running_mean=0, running_var=1
        // center = -running_mean = -0 = 0
        // std = gamma / sqrt(var + eps) = 1 / sqrt(1 + 0.001) ≈ 0.9995
        // beta = 0

        assert_eq!(center.len(), 4);
        assert_eq!(std.len(), 4);
        assert_eq!(beta.len(), 4);

        for i in 0..4 {
            assert!(
                center[i].abs() < 1e-5,
                "center[{}] should be ~0, got {}",
                i,
                center[i]
            );
            let expected_std = 1.0 / (1.0 + epsilon).sqrt();
            assert!(
                (std[i] - expected_std).abs() < 1e-5,
                "std[{}] should be ~{}, got {}",
                i,
                expected_std,
                std[i]
            );
            assert!(
                beta[i].abs() < 1e-5,
                "beta[{}] should be ~0, got {}",
                i,
                beta[i]
            );
        }
    }

    #[test]
    fn test_batch_norm_forward_initial_state() {
        use super::BatchNormLayer;

        let device = <TestBackend as Backend>::Device::default();
        let epsilon = 1e-3f32;
        let layer = BatchNormLayer::<TestBackend>::new(4, epsilon, &device);

        // With initial state (gamma=1, beta=0, mean=0, var=1):
        // output = gamma * (x - mean) / sqrt(var + eps) + beta
        // output = 1 * (x - 0) / sqrt(1 + 0.001) + 0
        // output ≈ x * 0.9995

        let input = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let output = layer.forward(input);
        let output_data: Vec<f32> = output.to_data().to_vec().unwrap();

        let expected_scale = 1.0 / (1.0 + epsilon).sqrt();
        let expected = vec![
            1.0 * expected_scale,
            2.0 * expected_scale,
            3.0 * expected_scale,
            4.0 * expected_scale,
        ];

        for i in 0..4 {
            assert!(
                (output_data[i] - expected[i]).abs() < 1e-4,
                "output[{}] should be ~{}, got {}",
                i,
                expected[i],
                output_data[i]
            );
        }
    }

    #[test]
    fn test_batch_norm_export_has_three_parameter_vectors() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();
        let normalized = ops::batch_norm(x);
        let output = ops::dense(1, Activation::Sigmoid, normalized);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        let export = model.to_instruction_model_info();

        // BatchNorm should create 3 parameter vectors (center, std, beta)
        assert!(
            export.parameters.len() >= 3,
            "Expected at least 3 parameter vectors for BatchNorm, got {}",
            export.parameters.len()
        );

        // Each parameter vector should have 4 elements (matching input size)
        for (i, params) in export.parameters.iter().take(3).enumerate() {
            assert_eq!(
                params.len(),
                4,
                "Parameter vector {} should have 4 elements, got {}",
                i,
                params.len()
            );
        }
    }

    #[test]
    fn test_batch_norm_export_instructions_order() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();
        let normalized = ops::batch_norm(x);
        let output = ops::dense(1, Activation::Sigmoid, normalized);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        let json = model
            .export_to_instruction_model()
            .expect("Serialization failed");

        // Verify instruction order: COPY, ADD_ELEMENTWISE, MUL_ELEMENTWISE, ADD_ELEMENTWISE, DOT
        let copy_pos = json.find("COPY").expect("Should have COPY");
        let first_add_pos = json
            .find("ADD_ELEMENTWISE")
            .expect("Should have ADD_ELEMENTWISE");
        let mul_pos = json
            .find("MUL_ELEMENTWISE")
            .expect("Should have MUL_ELEMENTWISE");
        let dot_pos = json.find("DOT").expect("Should have DOT");

        assert!(
            copy_pos < first_add_pos,
            "COPY should come before ADD_ELEMENTWISE"
        );
        assert!(
            first_add_pos < mul_pos,
            "First ADD_ELEMENTWISE should come before MUL_ELEMENTWISE"
        );
        assert!(mul_pos < dot_pos, "MUL_ELEMENTWISE should come before DOT");
    }

    #[test]
    fn test_batch_norm_sharing() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();

        // Create a shared BatchNorm operation
        let shared_bn = super::super::operation::Operation::batch_norm_default();

        // Apply same BatchNorm to two branches
        let branch1 = shared_bn.apply(x.clone());
        let branch2 = shared_bn.apply(x);

        // Merge and output
        let merged = ops::add(vec![branch1, branch2]);
        let output = ops::dense(1, Activation::Sigmoid, merged);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        // Forward pass should work
        let input_tensor = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let output_tensor = model.forward(input_tensor);
        assert_eq!(output_tensor.dims(), [1, 1]);

        // Both branches should use the same BatchNorm layer (same op_id)
        // This is verified by the model building without error
    }

    #[test]
    fn test_batch_norm_different_epsilons() {
        let device = <TestBackend as Backend>::Device::default();

        // Test with different epsilon values
        for epsilon in [1e-5, 1e-3, 1e-1] {
            let input = InputBuffer::new(4);
            let x = input.buffer();
            let normalized = ops::batch_norm_with_epsilon(epsilon, x);
            let output = ops::dense(1, Activation::Sigmoid, normalized);

            let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
                .expect("Model creation should succeed");

            let input_tensor =
                Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
            let output_tensor = model.forward(input_tensor);
            assert_eq!(output_tensor.dims(), [1, 1]);
        }
    }

    #[test]
    fn test_batch_norm_preserves_batch_dimension() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();
        let normalized = ops::batch_norm(x);
        let output = ops::dense(2, Activation::None, normalized);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        // Test with batch size > 1
        let input_tensor = Tensor::<TestBackend, 2>::from_floats(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            &device,
        );
        let output_tensor = model.forward(input_tensor);
        assert_eq!(output_tensor.dims(), [3, 2]);
    }

    #[test]
    fn test_batch_norm_with_negative_inputs() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();
        let normalized = ops::batch_norm(x);
        let output = ops::dense(1, Activation::None, normalized);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        // Test with negative values
        let input_tensor =
            Tensor::<TestBackend, 2>::from_floats([[-1.0, -2.0, 3.0, -4.0]], &device);
        let output_tensor = model.forward(input_tensor);
        assert_eq!(output_tensor.dims(), [1, 1]);

        // Output should be a valid number (not NaN or Inf)
        let output_data: Vec<f32> = output_tensor.to_data().to_vec().unwrap();
        assert!(output_data[0].is_finite(), "Output should be finite");
    }

    #[test]
    fn test_batch_norm_with_zero_inputs() {
        let device = <TestBackend as Backend>::Device::default();

        let input = InputBuffer::new(4);
        let x = input.buffer();
        let normalized = ops::batch_norm(x);
        let output = ops::dense(1, Activation::None, normalized);

        let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
            .expect("Model creation should succeed");

        // Test with all zeros
        let input_tensor = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0, 0.0, 0.0]], &device);
        let output_tensor = model.forward(input_tensor);
        assert_eq!(output_tensor.dims(), [1, 1]);

        // Output should be a valid number
        let output_data: Vec<f32> = output_tensor.to_data().to_vec().unwrap();
        assert!(output_data[0].is_finite(), "Output should be finite");
    }
}
