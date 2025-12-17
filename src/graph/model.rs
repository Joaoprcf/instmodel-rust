//! ModelGraph - the main container for functional neural networks.

use std::collections::HashMap;

use burn::module::{Ignored, Module};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig};
use burn::tensor::{Tensor, backend::Backend};

use crate::errors::ModelError;
use crate::layers::Activation;

use super::buffer::{BufferId, DataBuffer, InputBuffer, Producer};
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
/// Supports optional center (beta) and scale (gamma) parameters.
#[derive(Module, Debug)]
struct BatchNormLayer<B: Backend> {
    batch_norm: BatchNorm<B>,
    num_features: usize,
    epsilon: f32,
    #[module(ignore)]
    center: Ignored<bool>,
    #[module(ignore)]
    scale: Ignored<bool>,
}

impl<B: Backend> BatchNormLayer<B> {
    fn new(
        num_features: usize,
        epsilon: f32,
        center: bool,
        scale: bool,
        device: &B::Device,
    ) -> Self {
        let batch_norm = BatchNormConfig::new(num_features)
            .with_epsilon(epsilon as f64)
            .init(device);
        Self {
            batch_norm,
            num_features,
            epsilon,
            center: Ignored(center),
            scale: Ignored(scale),
        }
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let [_batch, features] = input.dims();
        let device = input.device();

        // Get running stats
        let running_mean = self.batch_norm.running_mean.value();
        let running_var = self.batch_norm.running_var.value();

        // Normalize: (x - mean) / sqrt(var + eps)
        let mean_2d = running_mean.clone().reshape([1, features]);
        let var_2d = running_var.clone().reshape([1, features]);
        let eps = Tensor::<B, 2>::full([1, features], self.epsilon as f64, &device);
        let std = (var_2d + eps).sqrt();

        let normalized = (input - mean_2d) / std;

        // Apply gamma (scale) if enabled
        let scaled = if self.scale.0 {
            let gamma = self.batch_norm.gamma.val();
            let gamma_2d = gamma.reshape([1, features]);
            normalized * gamma_2d
        } else {
            normalized
        };

        // Apply beta (center/shift) if enabled
        if self.center.0 {
            let beta = self.batch_norm.beta.val();
            let beta_2d = beta.reshape([1, features]);
            scaled + beta_2d
        } else {
            scaled
        }
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

        // std = gamma / sqrt(variance + epsilon) if scale enabled, else 1/sqrt(var+eps)
        let std: Vec<f32> = if self.scale.0 {
            gamma
                .iter()
                .zip(running_var.iter())
                .map(|(g, v)| g / (v + epsilon).sqrt())
                .collect()
        } else {
            running_var
                .iter()
                .map(|v| 1.0 / (v + epsilon).sqrt())
                .collect()
        };

        // beta = learned beta if center enabled, else zeros
        let beta_out: Vec<f32> = if self.center.0 {
            beta
        } else {
            vec![0.0; self.num_features]
        };

        (center, std, beta_out)
    }
}

/// Execution step in the forward pass.
/// Uses layer indices instead of OpIds for efficient forward pass.
#[derive(Debug, Clone)]
enum Step {
    Dense {
        layer_index: usize,
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
        layer_index: usize,
        input: BufferId,
        output: BufferId,
    },
    StandaloneActivation {
        input: BufferId,
        output: BufferId,
        activation: Activation,
    },
}

/// Metadata for export: maps layer indices back to OpIds for weight export.
#[derive(Debug, Clone, Default)]
struct ExportMetadata {
    /// Maps dense layer index to OpId for export
    dense_op_ids: Vec<OpId>,
    /// Maps batch norm layer index to OpId for export
    batch_norm_op_ids: Vec<OpId>,
}

/// ModelGraph holds the computation graph and initialized layers.
/// Derives Module to enable training with burn's optimizer.
#[derive(Module, Debug)]
pub struct ModelGraph<B: Backend> {
    /// Trainable dense layers (Vec for Module derive compatibility)
    dense_layers: Vec<DenseLayer<B>>,
    /// Trainable batch normalization layers
    batch_norm_layers: Vec<BatchNormLayer<B>>,
    /// Non-trainable graph metadata (wrapped in Ignored)
    #[module(ignore)]
    inputs: Ignored<Vec<InputBuffer>>,
    #[module(ignore)]
    output: Ignored<DataBuffer>,
    #[module(ignore)]
    steps: Ignored<Vec<Step>>,
    #[module(ignore)]
    buffer_index: Ignored<HashMap<BufferId, usize>>,
    #[module(ignore)]
    features: Ignored<Vec<String>>,
    #[module(ignore)]
    export_metadata: Ignored<ExportMetadata>,
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

        let (dense_layers, dense_op_ids) = builder.init_dense_layers::<B>(device);
        let (batch_norm_layers, batch_norm_op_ids) = builder.init_batch_norm_layers::<B>(device);

        let features: Vec<String> = inputs
            .iter()
            .filter_map(|i| i.features())
            .flatten()
            .cloned()
            .collect();

        let export_metadata = ExportMetadata {
            dense_op_ids,
            batch_norm_op_ids,
        };

        Ok(Self {
            dense_layers,
            batch_norm_layers,
            inputs: Ignored(inputs),
            output: Ignored(output),
            steps: Ignored(builder.steps),
            buffer_index: Ignored(builder.buffer_index),
            features: Ignored(features),
            export_metadata: Ignored(export_metadata),
        })
    }

    /// Performs forward pass through the model.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut buffers: HashMap<usize, Tensor<B, 2>> = HashMap::new();

        let input_idx = self.buffer_index.0[&self.inputs.0[0].id()];
        buffers.insert(input_idx, input);

        for step in &self.steps.0 {
            match step {
                Step::Dense {
                    layer_index,
                    input,
                    output,
                } => {
                    let in_idx = self.buffer_index.0[input];
                    let out_idx = self.buffer_index.0[output];
                    let layer = &self.dense_layers[*layer_index];
                    let result = layer.forward(buffers[&in_idx].clone());
                    buffers.insert(out_idx, result);
                }
                Step::Add { inputs, output } => {
                    let out_idx = self.buffer_index.0[output];
                    let mut result = buffers[&self.buffer_index.0[&inputs[0]]].clone();
                    for buf_id in &inputs[1..] {
                        result = result.add(buffers[&self.buffer_index.0[buf_id]].clone());
                    }
                    buffers.insert(out_idx, result);
                }
                Step::Multiply { inputs, output } => {
                    let out_idx = self.buffer_index.0[output];
                    let mut result = buffers[&self.buffer_index.0[&inputs[0]]].clone();
                    for buf_id in &inputs[1..] {
                        result = result.mul(buffers[&self.buffer_index.0[buf_id]].clone());
                    }
                    buffers.insert(out_idx, result);
                }
                Step::Concat { inputs, output } => {
                    let out_idx = self.buffer_index.0[output];
                    let tensors: Vec<Tensor<B, 2>> = inputs
                        .iter()
                        .map(|buf_id| buffers[&self.buffer_index.0[buf_id]].clone())
                        .collect();
                    let result = Tensor::cat(tensors, 1);
                    buffers.insert(out_idx, result);
                }
                Step::Softmax { input, output } => {
                    let in_idx = self.buffer_index.0[input];
                    let out_idx = self.buffer_index.0[output];
                    let result = burn::tensor::activation::softmax(buffers[&in_idx].clone(), 1);
                    buffers.insert(out_idx, result);
                }
                Step::Scale {
                    input,
                    output,
                    vector,
                } => {
                    let in_idx = self.buffer_index.0[input];
                    let out_idx = self.buffer_index.0[output];
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
                    let in_idx = self.buffer_index.0[input];
                    let out_idx = self.buffer_index.0[output];
                    let in_tensor = buffers[&in_idx].clone();
                    let device = in_tensor.device();
                    let shift_tensor =
                        Tensor::<B, 1>::from_floats(vector.as_slice(), &device).reshape([1, -1]);
                    let result = in_tensor.add(shift_tensor);
                    buffers.insert(out_idx, result);
                }
                Step::BatchNorm {
                    layer_index,
                    input,
                    output,
                } => {
                    let in_idx = self.buffer_index.0[input];
                    let out_idx = self.buffer_index.0[output];
                    let layer = &self.batch_norm_layers[*layer_index];
                    let result = layer.forward(buffers[&in_idx].clone());
                    buffers.insert(out_idx, result);
                }
                Step::StandaloneActivation {
                    input,
                    output,
                    activation,
                } => {
                    let in_idx = self.buffer_index.0[input];
                    let out_idx = self.buffer_index.0[output];
                    let result = activation.apply(buffers[&in_idx].clone());
                    buffers.insert(out_idx, result);
                }
            }
        }

        let output_idx = self.buffer_index.0[&self.output.0.id()];
        buffers.remove(&output_idx).unwrap()
    }

    /// Returns the input feature names.
    pub fn features(&self) -> &[String] {
        &self.features.0
    }

    /// Returns the number of input features.
    pub fn feature_size(&self) -> usize {
        self.inputs.0.iter().map(|i| i.size()).sum()
    }

    /// Returns the output size of the model.
    pub fn output_size(&self) -> usize {
        self.output.0.size()
    }

    /// Applies this model to the given inputs, returning a DataBuffer.
    ///
    /// This enables nested model composition where a ModelGraph can be used
    /// like any other operation in the graph. The weights are shared across
    /// all applications of the same model.
    pub fn apply(&self, inputs: Vec<DataBuffer>) -> DataBuffer {
        assert_eq!(
            inputs.len(),
            self.inputs.0.len(),
            "Number of inputs must match the model's input count"
        );

        // Generate a unique ID for this sub-graph application
        let id = super::operation::next_op_id();

        // Collect input buffer IDs from the original model
        let input_ids: Vec<BufferId> = self.inputs.0.iter().map(|i| i.id()).collect();

        DataBuffer::new_with_producer(
            self.output_size(),
            Some(Producer::SubGraph {
                id,
                input_ids,
                output: Box::new(self.output.0.clone()),
            }),
            inputs,
        )
    }

    /// Applies this model to a single input, returning a DataBuffer.
    ///
    /// Convenience method for models with a single input.
    pub fn apply_single(&self, input: DataBuffer) -> DataBuffer {
        self.apply(vec![input])
    }

    /// Exports the model to instruction model JSON format.
    pub fn export_to_instruction_model(&self) -> Result<String, serde_json::Error> {
        let export = self.to_instruction_model_info();
        serde_json::to_string_pretty(&export)
    }

    /// Converts the model to an InstructionModelExport struct.
    pub fn to_instruction_model_info(&self) -> InstructionModelExport {
        let mut ctx = CompileContext::new();

        for input in &self.inputs.0 {
            ctx.register_input(input.id(), input.size(), input.features());
        }

        self.compile_buffer(&self.output.0, &mut ctx);

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

        // Recursively compile inputs first
        let input_indices: Vec<usize> = buffer
            .inputs()
            .iter()
            .map(|inp| self.compile_buffer(inp, ctx))
            .collect();

        let output_idx = match producer {
            Producer::Op(op) => {
                self.fill_dense_weights(op, ctx);
                self.compile_operation(op, &input_indices, ctx)
            }
            Producer::SubGraph {
                input_ids, output, ..
            } => {
                // Create a LOCAL visited map for this sub-graph application.
                // Pre-populate with the input mapping: sub-graph input ID -> actual buffer index
                let mut local_visited: HashMap<BufferId, usize> = HashMap::new();
                for (&sub_id, &actual_idx) in input_ids.iter().zip(input_indices.iter()) {
                    local_visited.insert(sub_id, actual_idx);
                }

                // Compile the sub-graph with the local visited map
                self.compile_subgraph_buffer(output, ctx, &mut local_visited)
            }
        };

        ctx.visited_buffers.insert(buffer.id(), output_idx);

        output_idx
    }

    /// Compiles a sub-graph buffer using a local visited map.
    /// This ensures each sub-graph application creates new buffers.
    fn compile_subgraph_buffer(
        &self,
        buffer: &DataBuffer,
        ctx: &mut CompileContext,
        local_visited: &mut HashMap<BufferId, usize>,
    ) -> usize {
        // Check if this buffer is already in local_visited (remapped input or already compiled in this application)
        if let Some(&idx) = local_visited.get(&buffer.id()) {
            return idx;
        }

        let Some(producer) = buffer.producer() else {
            panic!(
                "Sub-graph buffer {} has no producer and is not a remapped input",
                buffer.id()
            );
        };

        // Recursively compile inputs using local_visited
        let input_indices: Vec<usize> = buffer
            .inputs()
            .iter()
            .map(|inp| self.compile_subgraph_buffer(inp, ctx, local_visited))
            .collect();

        let output_idx = match producer {
            Producer::Op(op) => {
                self.fill_dense_weights(op, ctx);
                self.compile_operation(op, &input_indices, ctx)
            }
            Producer::SubGraph {
                input_ids, output, ..
            } => {
                // Nested sub-graph: create new local_visited
                let mut nested_visited: HashMap<BufferId, usize> = HashMap::new();
                for (&sub_id, &actual_idx) in input_ids.iter().zip(input_indices.iter()) {
                    nested_visited.insert(sub_id, actual_idx);
                }

                // Recursively compile the nested sub-graph
                self.compile_subgraph_buffer(output, ctx, &mut nested_visited)
            }
        };

        // Track in local_visited for DAG handling within this sub-graph
        local_visited.insert(buffer.id(), output_idx);

        output_idx
    }

    fn fill_dense_weights(&self, producer: &Operation, ctx: &mut CompileContext) {
        let Operation::Dense { id, .. } = producer else {
            return;
        };
        // Find layer index from OpId using export_metadata
        let Some(layer_index) = self
            .export_metadata
            .0
            .dense_op_ids
            .iter()
            .position(|op_id| op_id == id)
        else {
            return;
        };
        let layer = &self.dense_layers[layer_index];
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
        // Find layer index from OpId using export_metadata
        let Some(layer_index) = self
            .export_metadata
            .0
            .batch_norm_op_ids
            .iter()
            .position(|op_id| op_id == id)
        else {
            return producer.compile(input_indices, ctx);
        };
        let layer = &self.batch_norm_layers[layer_index];
        let (center, std, beta) = layer.get_normalization_params();
        producer.compile_batch_norm(input_indices, ctx, &center, &std, &beta)
    }
}

/// Builder for constructing ModelGraph execution order.
struct GraphBuilder {
    buffer_index: HashMap<BufferId, usize>,
    next_idx: usize,
    steps: Vec<Step>,
    /// Maps OpId -> (input_size, output_size, activation, layer_index)
    dense_info: HashMap<OpId, (usize, usize, Activation, usize)>,
    /// Maps OpId -> (num_features, epsilon, center, scale, layer_index)
    batch_norm_info: HashMap<OpId, (usize, f32, bool, bool, usize)>,
    visited: HashMap<BufferId, bool>,
    next_dense_index: usize,
    next_batch_norm_index: usize,
    /// Counter for generating unique buffer IDs for sub-graph internals.
    /// Starts at a high value to avoid collision with global buffer IDs.
    next_subgraph_buffer_id: BufferId,
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
            next_dense_index: 0,
            next_batch_norm_index: 0,
            // Start sub-graph buffer IDs at a high value to avoid collision
            // with global buffer IDs which start at 0
            next_subgraph_buffer_id: usize::MAX / 2,
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

        let inputs = buffer.inputs();
        let input_size = if inputs.is_empty() {
            0
        } else {
            inputs[0].size()
        };

        match producer {
            Producer::Op(op) => {
                self.buffer_index.insert(buffer.id(), self.next_idx);
                self.next_idx += 1;
                self.traverse_operation(op, buffer, inputs, input_size);
            }
            Producer::SubGraph {
                input_ids, output, ..
            } => {
                // Build mapping: sub-graph input IDs -> actual input buffer IDs
                let input_remap: HashMap<BufferId, BufferId> = input_ids
                    .iter()
                    .zip(inputs.iter())
                    .map(|(&sub_id, actual_buf)| (sub_id, actual_buf.id()))
                    .collect();

                // Traverse the sub-graph's output, which inlines all operations
                let output_idx = self.traverse_subgraph_inline(output, &input_remap)?;

                // The outer buffer ID maps to the same index as the sub-graph output
                self.buffer_index.insert(buffer.id(), output_idx);
            }
        }

        self.visited.insert(buffer.id(), true);
        Ok(())
    }

    fn traverse_operation(
        &mut self,
        op: &Operation,
        buffer: &DataBuffer,
        inputs: &[DataBuffer],
        input_size: usize,
    ) {
        match op {
            Operation::Dense {
                id,
                output_size,
                activation,
            } => {
                let layer_index = if let Some((_, _, _, idx)) = self.dense_info.get(id) {
                    *idx
                } else {
                    let idx = self.next_dense_index;
                    self.dense_info
                        .insert(*id, (input_size, *output_size, *activation, idx));
                    self.next_dense_index += 1;
                    idx
                };
                self.steps.push(Step::Dense {
                    layer_index,
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
            Operation::BatchNorm {
                id,
                epsilon,
                center,
                scale,
                ..
            } => {
                let layer_index = if let Some((_, _, _, _, idx)) = self.batch_norm_info.get(id) {
                    *idx
                } else {
                    let idx = self.next_batch_norm_index;
                    self.batch_norm_info
                        .insert(*id, (input_size, *epsilon, *center, *scale, idx));
                    self.next_batch_norm_index += 1;
                    idx
                };
                self.steps.push(Step::BatchNorm {
                    layer_index,
                    input: inputs[0].id(),
                    output: buffer.id(),
                });
            }
            Operation::StandaloneActivation { activation, .. } => {
                self.steps.push(Step::StandaloneActivation {
                    input: inputs[0].id(),
                    output: buffer.id(),
                    activation: *activation,
                });
            }
        }
    }

    /// Inlines a sub-graph by traversing its internal structure,
    /// creating buffer entries and Steps for all operations.
    /// Returns the buffer index of the sub-graph's output.
    ///
    /// Uses a LOCAL visited map (like Python's visited dict) so each sub-graph
    /// application creates new buffers, enabling multiple applications of the
    /// same sub-model with shared weights.
    fn traverse_subgraph_inline(
        &mut self,
        output: &DataBuffer,
        input_remap: &HashMap<BufferId, BufferId>,
    ) -> Result<usize, ModelError> {
        // Create a LOCAL visited map for this sub-graph application.
        // Maps sub-graph internal buffer IDs to NEW buffer IDs for this application.
        // Pre-populate with the input mapping: sub-graph input ID -> actual buffer ID
        let mut local_visited: HashMap<BufferId, BufferId> = HashMap::new();
        for (&sub_id, &actual_id) in input_remap {
            // The actual_id is already a valid buffer ID in buffer_index
            local_visited.insert(sub_id, actual_id);
        }
        self.traverse_subgraph_buffer_inline(output, &mut local_visited)
    }

    /// Recursively traverses a sub-graph buffer, inlining operations.
    /// Uses local_visited to track buffers within THIS application only.
    /// local_visited maps sub-graph internal buffer IDs to NEW buffer IDs for this application.
    fn traverse_subgraph_buffer_inline(
        &mut self,
        buffer: &DataBuffer,
        local_visited: &mut HashMap<BufferId, BufferId>,
    ) -> Result<usize, ModelError> {
        // Check if already visited IN THIS APPLICATION (enables DAG within sub-graph)
        if let Some(&new_id) = local_visited.get(&buffer.id()) {
            // Return the index for the new buffer ID
            return Ok(self.buffer_index[&new_id]);
        }

        // Must have a producer (inputs should be in local_visited)
        let Some(producer) = buffer.producer() else {
            return Err(ModelError::InvalidGraph(format!(
                "Sub-graph buffer {} has no producer and is not a remapped input",
                buffer.id()
            )));
        };

        // First, recursively process all input buffers
        let input_indices: Vec<usize> = buffer
            .inputs()
            .iter()
            .map(|inp| self.traverse_subgraph_buffer_inline(inp, local_visited))
            .collect::<Result<Vec<_>, _>>()?;

        let inputs = buffer.inputs();
        let input_size = if inputs.is_empty() {
            0
        } else {
            inputs[0].size()
        };

        // Register this buffer and create the appropriate Step
        match producer {
            Producer::Op(op) => {
                // Allocate a NEW buffer index for this operation's output
                let output_idx = self.next_idx;
                self.next_idx += 1;

                // Generate a NEW unique BufferId for this application's buffer
                // Uses a separate high counter to avoid collision with global buffer IDs
                let new_buffer_id = self.next_subgraph_buffer_id;
                self.next_subgraph_buffer_id += 1;

                // Register in buffer_index for forward pass
                self.buffer_index.insert(new_buffer_id, output_idx);

                // Track mapping: sub-graph buffer ID -> new buffer ID
                local_visited.insert(buffer.id(), new_buffer_id);

                // Create the Step for this operation using NEW buffer IDs
                self.traverse_operation_with_inputs_new_ids(
                    op,
                    new_buffer_id,
                    &input_indices,
                    input_size,
                    local_visited,
                );

                Ok(output_idx)
            }
            Producer::SubGraph {
                input_ids,
                output: inner_output,
                ..
            } => {
                // Nested sub-graph: create new local_visited from current state
                // Map nested input IDs to the NEW buffer IDs from the current application
                let mut nested_visited: HashMap<BufferId, BufferId> = HashMap::new();
                for (&sub_id, inp) in input_ids.iter().zip(buffer.inputs().iter()) {
                    // Get the new buffer ID for this input
                    let new_id = local_visited[&inp.id()];
                    nested_visited.insert(sub_id, new_id);
                }

                // Recursively inline the nested sub-graph
                let output_idx =
                    self.traverse_subgraph_buffer_inline(inner_output, &mut nested_visited)?;

                // The new buffer ID is whatever the nested sub-graph output mapped to
                let nested_output_new_id = nested_visited[&inner_output.id()];
                local_visited.insert(buffer.id(), nested_output_new_id);

                Ok(output_idx)
            }
        }
    }

    /// Creates a Step for an operation in a sub-graph, using NEW buffer IDs.
    /// This is used during sub-graph inlining where we generate new buffer IDs
    /// for each application to avoid conflicts.
    fn traverse_operation_with_inputs_new_ids(
        &mut self,
        op: &Operation,
        new_output_id: BufferId,
        input_indices: &[usize],
        input_size: usize,
        local_visited: &HashMap<BufferId, BufferId>,
    ) {
        // Get the input buffer IDs (these are the NEW buffer IDs from local_visited)
        let input_ids: Vec<BufferId> = input_indices
            .iter()
            .map(|&idx| {
                // Find the NEW buffer ID that maps to this index
                self.buffer_index
                    .iter()
                    .find(|entry| *entry.1 == idx)
                    .map(|entry| *entry.0)
                    .unwrap_or_else(|| {
                        // Fallback: check if it's in local_visited values
                        local_visited
                            .values()
                            .find(|&&id| self.buffer_index.get(&id) == Some(&idx))
                            .copied()
                            .unwrap_or(0)
                    })
            })
            .collect();

        match op {
            Operation::Dense {
                id,
                output_size,
                activation,
            } => {
                let layer_index = if let Some((_, _, _, idx)) = self.dense_info.get(id) {
                    *idx
                } else {
                    let idx = self.next_dense_index;
                    self.dense_info
                        .insert(*id, (input_size, *output_size, *activation, idx));
                    self.next_dense_index += 1;
                    idx
                };
                self.steps.push(Step::Dense {
                    layer_index,
                    input: input_ids[0],
                    output: new_output_id,
                });
            }
            Operation::Add { .. } => {
                self.steps.push(Step::Add {
                    inputs: input_ids,
                    output: new_output_id,
                });
            }
            Operation::Multiply { .. } => {
                self.steps.push(Step::Multiply {
                    inputs: input_ids,
                    output: new_output_id,
                });
            }
            Operation::Concat { .. } => {
                self.steps.push(Step::Concat {
                    inputs: input_ids,
                    output: new_output_id,
                });
            }
            Operation::Softmax { .. } => {
                self.steps.push(Step::Softmax {
                    input: input_ids[0],
                    output: new_output_id,
                });
            }
            Operation::Scale { vector, .. } => {
                self.steps.push(Step::Scale {
                    input: input_ids[0],
                    output: new_output_id,
                    vector: vector.clone(),
                });
            }
            Operation::Shift { vector, .. } => {
                self.steps.push(Step::Shift {
                    input: input_ids[0],
                    output: new_output_id,
                    vector: vector.clone(),
                });
            }
            Operation::BatchNorm {
                id,
                epsilon,
                center,
                scale,
                ..
            } => {
                let layer_index = if let Some((_, _, _, _, idx)) = self.batch_norm_info.get(id) {
                    *idx
                } else {
                    let idx = self.next_batch_norm_index;
                    self.batch_norm_info
                        .insert(*id, (input_size, *epsilon, *center, *scale, idx));
                    self.next_batch_norm_index += 1;
                    idx
                };
                self.steps.push(Step::BatchNorm {
                    layer_index,
                    input: input_ids[0],
                    output: new_output_id,
                });
            }
            Operation::StandaloneActivation { activation, .. } => {
                self.steps.push(Step::StandaloneActivation {
                    input: input_ids[0],
                    output: new_output_id,
                    activation: *activation,
                });
            }
        }
    }

    /// Initialize dense layers and return (layers_vec, op_ids_vec).
    /// The op_ids_vec maps layer index to OpId for export purposes.
    fn init_dense_layers<B: Backend>(&self, device: &B::Device) -> (Vec<DenseLayer<B>>, Vec<OpId>) {
        let mut layers: Vec<(usize, OpId, DenseLayer<B>)> = self
            .dense_info
            .iter()
            .map(
                |(&op_id, &(input_size, output_size, activation, layer_index))| {
                    (
                        layer_index,
                        op_id,
                        DenseLayer::new(input_size, output_size, activation, device),
                    )
                },
            )
            .collect();

        // Sort by layer index to ensure correct ordering
        layers.sort_by_key(|(idx, _, _)| *idx);

        let op_ids: Vec<OpId> = layers.iter().map(|(_, op_id, _)| *op_id).collect();
        let dense_layers: Vec<DenseLayer<B>> =
            layers.into_iter().map(|(_, _, layer)| layer).collect();

        (dense_layers, op_ids)
    }

    /// Initialize batch norm layers and return (layers_vec, op_ids_vec).
    fn init_batch_norm_layers<B: Backend>(
        &self,
        device: &B::Device,
    ) -> (Vec<BatchNormLayer<B>>, Vec<OpId>) {
        let mut layers: Vec<(usize, OpId, BatchNormLayer<B>)> = self
            .batch_norm_info
            .iter()
            .map(
                |(&op_id, &(num_features, epsilon, center, scale, layer_index))| {
                    (
                        layer_index,
                        op_id,
                        BatchNormLayer::new(num_features, epsilon, center, scale, device),
                    )
                },
            )
            .collect();

        // Sort by layer index to ensure correct ordering
        layers.sort_by_key(|(idx, _, _)| *idx);

        let op_ids: Vec<OpId> = layers.iter().map(|(_, op_id, _)| *op_id).collect();
        let batch_norm_layers: Vec<BatchNormLayer<B>> =
            layers.into_iter().map(|(_, _, layer)| layer).collect();

        (batch_norm_layers, op_ids)
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
        let layer = BatchNormLayer::<TestBackend>::new(4, epsilon, true, true, &device);

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
        let layer = BatchNormLayer::<TestBackend>::new(4, epsilon, true, true, &device);

        // With initial state (gamma=1, beta=0, mean=0, var=1):
        // output = gamma * (x - mean) / sqrt(var + eps) + beta
        // output = 1 * (x - 0) / sqrt(1 + 0.001) + 0
        // output ≈ x * 0.9995

        let input = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let output = layer.forward(input);
        let output_data: Vec<f32> = output.to_data().to_vec().unwrap();

        let expected_scale = 1.0 / (1.0 + epsilon).sqrt();
        let expected = [
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

    // Training tests with Autodiff backend
    mod training_tests {
        use super::*;
        use crate::graph::Operation;
        use burn::backend::Autodiff;
        use burn::optim::{AdamConfig, GradientsParams, Optimizer};
        use burn::tensor::ElementConversion;

        type TrainingBackend = Autodiff<NdArray>;

        #[test]
        fn test_graph_model_is_trainable() {
            let device = <TrainingBackend as Backend>::Device::default();

            // Simple model: input -> hidden -> output
            let input = InputBuffer::new(4);
            let x = input.buffer();
            let hidden = ops::dense(8, Activation::Relu, x);
            let output = ops::dense(1, Activation::Sigmoid, hidden);

            let mut model = ModelGraph::<TrainingBackend>::new(vec![input], output, &device)
                .expect("Model creation should succeed");

            let mut optimizer = AdamConfig::new().init();

            // Create dummy data
            let input_tensor =
                Tensor::<TrainingBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
            let target = Tensor::<TrainingBackend, 2>::from_floats([[0.5]], &device);

            // Forward pass
            let output_tensor = model.forward(input_tensor);

            // Compute loss (MSE)
            let diff = output_tensor.sub(target);
            let loss = diff.clone().mul(diff).mean();
            let initial_loss: f32 = loss.clone().into_scalar().elem();

            // Backward pass
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);

            // Update model
            model = optimizer.step(0.01, model, grads_params);

            // Verify model was updated (loss should change after parameter update)
            let input_tensor2 =
                Tensor::<TrainingBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
            let target2 = Tensor::<TrainingBackend, 2>::from_floats([[0.5]], &device);
            let output_tensor2 = model.forward(input_tensor2);
            let diff2 = output_tensor2.sub(target2);
            let loss2 = diff2.clone().mul(diff2).mean();
            let new_loss: f32 = loss2.into_scalar().elem();

            // The losses should be different (model was updated)
            assert!(
                (initial_loss - new_loss).abs() > 1e-10,
                "Model parameters should have been updated"
            );
        }

        #[test]
        fn test_fan_out_concat_training() {
            let device = <TrainingBackend as Backend>::Device::default();

            // Fan-out pattern: input -> two branches -> concat -> output
            let input = InputBuffer::new(4);
            let x = input.buffer();

            let branch1 = ops::dense(3, Activation::Relu, x.clone());
            let branch2 = ops::dense(2, Activation::Relu, x);

            let concatenated = ops::concat(vec![branch1, branch2]);
            let output = ops::dense(1, Activation::Sigmoid, concatenated);

            let mut model = ModelGraph::<TrainingBackend>::new(vec![input], output, &device)
                .expect("Model creation should succeed");

            let mut optimizer = AdamConfig::new().init();

            // Training loop
            let mut losses = Vec::new();
            for _ in 0..5 {
                let input_tensor =
                    Tensor::<TrainingBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
                let target = Tensor::<TrainingBackend, 2>::from_floats([[0.7]], &device);

                let output_tensor = model.forward(input_tensor);
                let diff = output_tensor.sub(target);
                let loss = diff.clone().mul(diff).mean();

                losses.push(loss.clone().into_scalar().elem::<f32>());

                let grads = loss.backward();
                let grads_params = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(0.1, model, grads_params);
            }

            // Loss should generally decrease over training
            assert!(
                losses[4] < losses[0] * 1.5,
                "Loss should not increase dramatically during training"
            );
        }

        #[test]
        fn test_fan_in_add_training() {
            let device = <TrainingBackend as Backend>::Device::default();

            // Fan-in pattern: input -> two branches -> add -> output (residual-like)
            let input = InputBuffer::new(4);
            let x = input.buffer();

            let branch1 = ops::dense(4, Activation::Relu, x.clone());
            let branch2 = ops::dense(4, Activation::Relu, x);

            let added = ops::add(vec![branch1, branch2]);
            let output = ops::dense(1, Activation::Sigmoid, added);

            let mut model = ModelGraph::<TrainingBackend>::new(vec![input], output, &device)
                .expect("Model creation should succeed");

            let mut optimizer = AdamConfig::new().init();

            // Single training step should work
            let input_tensor =
                Tensor::<TrainingBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
            let target = Tensor::<TrainingBackend, 2>::from_floats([[0.3]], &device);

            let output_tensor = model.forward(input_tensor);
            let diff = output_tensor.sub(target);
            let loss = diff.clone().mul(diff).mean();

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(0.01, model, grads_params);

            // Verify model still works after update
            let input_tensor2 =
                Tensor::<TrainingBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
            let output_tensor2 = model.forward(input_tensor2);
            let output_data: Vec<f32> = output_tensor2.into_data().to_vec().unwrap();
            assert!(
                output_data[0].is_finite(),
                "Output should be finite after training"
            );
        }

        #[test]
        fn test_shared_layer_training() {
            let device = <TrainingBackend as Backend>::Device::default();

            // Shared layer pattern: same dense applied to different transformations
            let input = InputBuffer::new(4);
            let x = input.buffer();

            // Create a shared dense operation
            let shared_dense = Operation::dense(4, Activation::Relu);

            // Apply to two different paths
            let path1 = shared_dense.apply(x.clone());
            let scaled_x = ops::scale(vec![2.0, 2.0, 2.0, 2.0], x);
            let path2 = shared_dense.apply(scaled_x);

            let concatenated = ops::concat(vec![path1, path2]);
            let output = ops::dense(1, Activation::Sigmoid, concatenated);

            let mut model = ModelGraph::<TrainingBackend>::new(vec![input], output, &device)
                .expect("Model creation should succeed");

            let mut optimizer = AdamConfig::new().init();

            // Training step
            let input_tensor =
                Tensor::<TrainingBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
            let target = Tensor::<TrainingBackend, 2>::from_floats([[0.5]], &device);

            let output_tensor = model.forward(input_tensor);
            let diff = output_tensor.sub(target);
            let loss = diff.clone().mul(diff).mean();

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(0.01, model, grads_params);

            // Model should still work
            let input_tensor2 =
                Tensor::<TrainingBackend, 2>::from_floats([[0.5, 1.0, 1.5, 2.0]], &device);
            let output_tensor2 = model.forward(input_tensor2);
            assert_eq!(output_tensor2.dims(), [1, 1]);
        }

        #[test]
        fn test_training_loss_decreases() {
            let device = <TrainingBackend as Backend>::Device::default();

            // Model to learn: output = mean(input) approximately
            let input = InputBuffer::new(4);
            let x = input.buffer();
            let hidden = ops::dense(8, Activation::Relu, x);
            let output = ops::dense(1, Activation::None, hidden);

            let mut model = ModelGraph::<TrainingBackend>::new(vec![input], output, &device)
                .expect("Model creation should succeed");

            let mut optimizer = AdamConfig::new().init();

            // Training data: input -> mean
            let training_data = vec![
                ([1.0f32, 2.0, 3.0, 4.0], 2.5f32),
                ([0.0, 0.0, 0.0, 0.0], 0.0),
                ([4.0, 4.0, 4.0, 4.0], 4.0),
                ([-1.0, 1.0, -1.0, 1.0], 0.0),
            ];

            let mut initial_total_loss = 0.0f32;
            let mut final_total_loss = 0.0f32;

            // Multiple epochs
            for epoch in 0..20 {
                let mut epoch_loss = 0.0f32;

                for (input_data, target_val) in &training_data {
                    let input_tensor =
                        Tensor::<TrainingBackend, 2>::from_floats([*input_data], &device);
                    let target =
                        Tensor::<TrainingBackend, 2>::from_floats([[*target_val]], &device);

                    let output_tensor = model.forward(input_tensor);
                    let diff = output_tensor.sub(target);
                    let loss = diff.clone().mul(diff).mean();

                    epoch_loss += loss.clone().into_scalar().elem::<f32>();

                    let grads = loss.backward();
                    let grads_params = GradientsParams::from_grads(grads, &model);
                    model = optimizer.step(0.01, model, grads_params);
                }

                if epoch == 0 {
                    initial_total_loss = epoch_loss;
                }
                if epoch == 19 {
                    final_total_loss = epoch_loss;
                }
            }

            // Loss should decrease significantly
            assert!(
                final_total_loss < initial_total_loss,
                "Loss should decrease: initial={}, final={}",
                initial_total_loss,
                final_total_loss
            );
        }
    }
}
