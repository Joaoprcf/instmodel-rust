use std::collections::HashMap;

use burn::module::{Ignored, Module};
use burn::nn::{BatchNorm, BatchNormConfig};
use burn::tensor::{Tensor, backend::Backend};

use crate::errors::ModelError;
use crate::layers::{Activation, Dense, DenseConfig};

use super::buffer::{BufferId, DataBuffer, InputBuffer, Producer};
use super::compile::{CompileContext, InstructionModelExport};
use super::core::{GraphId, ModelGraph};
use super::operation::{OpId, Operation};

/// BatchNorm that always normalises with the running statistics, matching the exported model.
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
        Self {
            batch_norm: BatchNormConfig::new(num_features)
                .with_epsilon(epsilon as f64)
                .init(device),
            num_features,
            epsilon,
            center: Ignored(center),
            scale: Ignored(scale),
        }
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let [_, features] = input.dims();
        let bn = &self.batch_norm;
        let mean = bn.running_mean.value().reshape([1, features]);
        let std = (bn.running_var.value().reshape([1, features]) + self.epsilon).sqrt();
        let mut x = (input - mean) / std;
        if self.scale.0 {
            x = x * bn.gamma.val().reshape([1, features]);
        }
        if self.center.0 {
            x = x + bn.beta.val().reshape([1, features]);
        }
        x
    }

    /// Returns `(center, std, beta)` as consumed by [`Operation::compile_batch_norm`].
    fn normalization_params(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let to_vec = |t: Tensor<B, 1>| -> Vec<f32> { t.to_data().to_vec().unwrap() };
        let bn = &self.batch_norm;
        let center = to_vec(bn.running_mean.value()).iter().map(|m| -m).collect();
        let inv_std = to_vec(bn.running_var.value())
            .into_iter()
            .map(|v| 1.0 / (v + self.epsilon).sqrt());
        let std = if self.scale.0 {
            to_vec(bn.gamma.val())
                .into_iter()
                .zip(inv_std)
                .map(|(g, s)| g * s)
                .collect()
        } else {
            inv_std.collect()
        };
        let beta = if self.center.0 {
            to_vec(bn.beta.val())
        } else {
            vec![0.0; self.num_features]
        };
        (center, std, beta)
    }
}

#[derive(Debug, Clone)]
enum StepKind {
    Dense(usize),
    BatchNorm(usize),
    Add,
    Multiply,
    Concat,
    Scale(Vec<f32>),
    Shift(Vec<f32>),
    Activation(Activation),
}

/// One forward-pass step; `inputs`/`output` and layer indices are positions, not ids.
#[derive(Debug, Clone)]
struct Step {
    kind: StepKind,
    inputs: Vec<usize>,
    output: usize,
}

fn run_steps<B: Backend>(
    steps: &[Step],
    dense: &[Dense<B>],
    batch_norms: &[BatchNormLayer<B>],
    (input_idx, input): (usize, Tensor<B, 2>),
    output_idx: usize,
) -> Tensor<B, 2> {
    let mut buffers = HashMap::from([(input_idx, input)]);
    for step in steps {
        let all = || step.inputs.iter().map(|i| buffers[i].clone());
        let first = || buffers[&step.inputs[0]].clone();
        let row = |v: &[f32], x: &Tensor<B, 2>| {
            Tensor::<B, 1>::from_floats(v, &x.device()).reshape([1, -1])
        };
        let out = match &step.kind {
            StepKind::Dense(l) => dense[*l].forward(first()),
            StepKind::BatchNorm(l) => batch_norms[*l].forward(first()),
            StepKind::Add => all().reduce(|a, b| a + b).unwrap(),
            StepKind::Multiply => all().reduce(|a, b| a * b).unwrap(),
            StepKind::Concat => Tensor::cat(all().collect(), 1),
            StepKind::Scale(v) => {
                let x = first();
                let p = row(v, &x);
                x * p
            }
            StepKind::Shift(v) => {
                let x = first();
                let p = row(v, &x);
                x + p
            }
            StepKind::Activation(a) => a.apply(first()),
        };
        buffers.insert(step.output, out);
    }
    buffers.remove(&output_idx).unwrap()
}

/// A sub-graph's slice of the compiled model, with layer indices local to the sub-model.
#[derive(Debug, Clone)]
struct SubgraphMapping {
    dense_layer_indices: Vec<usize>,
    batch_norm_layer_indices: Vec<usize>,
    steps: Vec<Step>,
    input: usize,
    output: usize,
}

impl SubgraphMapping {
    fn new(steps: &[Step], input: usize, output: usize) -> Self {
        fn local(layers: &mut Vec<usize>, global: usize) -> usize {
            layers.iter().position(|&l| l == global).unwrap_or_else(|| {
                layers.push(global);
                layers.len() - 1
            })
        }
        let (mut dense, mut batch_norms) = (Vec::new(), Vec::new());
        let steps = steps
            .iter()
            .cloned()
            .map(|mut step| {
                match &mut step.kind {
                    StepKind::Dense(l) => *l = local(&mut dense, *l),
                    StepKind::BatchNorm(l) => *l = local(&mut batch_norms, *l),
                    _ => {}
                }
                step
            })
            .collect();
        Self {
            dense_layer_indices: dense,
            batch_norm_layer_indices: batch_norms,
            steps,
            input,
            output,
        }
    }
}

/// Maps ops to their layer index, for weight export.
#[derive(Debug, Clone, Default)]
struct ExportMetadata {
    dense: HashMap<OpId, usize>,
    batch_norms: HashMap<OpId, usize>,
}

/// A compiled graph with trainable weights (a Burn `Module`). Forward feeds the first input.
#[derive(Module, Debug)]
pub struct CompiledModel<B: Backend> {
    dense_layers: Vec<Dense<B>>,
    batch_norm_layers: Vec<BatchNormLayer<B>>,
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
    #[module(ignore)]
    subgraph_mappings: Ignored<HashMap<GraphId, SubgraphMapping>>,
}

impl<B: Backend> CompiledModel<B> {
    pub fn new(
        inputs: Vec<InputBuffer>,
        output: DataBuffer,
        device: &B::Device,
    ) -> Result<Self, ModelError> {
        if inputs.is_empty() {
            return Err(ModelError::NoInputBuffer);
        }
        let mut builder = GraphBuilder::default();
        let mut visited: HashMap<BufferId, usize> = inputs
            .iter()
            .enumerate()
            .map(|(idx, input)| (input.id(), idx))
            .collect();
        builder.num_buffers = inputs.len();
        builder.visit(&output, &mut visited, false)?;

        let dense_layers = builder
            .dense
            .iter()
            .map(|&(_, input, output, act)| {
                DenseConfig::new(input, output)
                    .with_activation(act)
                    .init(device)
            })
            .collect();
        let batch_norm_layers = builder
            .batch_norms
            .iter()
            .map(|&(_, n, eps, center, scale)| BatchNormLayer::new(n, eps, center, scale, device))
            .collect();
        let export_metadata = ExportMetadata {
            dense: builder
                .dense
                .iter()
                .enumerate()
                .map(|(i, d)| (d.0, i))
                .collect(),
            batch_norms: (builder.batch_norms.iter().enumerate())
                .map(|(i, b)| (b.0, i))
                .collect(),
        };
        let features = inputs
            .iter()
            .filter_map(InputBuffer::features)
            .flatten()
            .cloned()
            .collect();

        Ok(Self {
            dense_layers,
            batch_norm_layers,
            inputs: Ignored(inputs),
            output: Ignored(output),
            steps: Ignored(builder.steps),
            buffer_index: Ignored(visited),
            features: Ignored(features),
            export_metadata: Ignored(export_metadata),
            subgraph_mappings: Ignored(builder.subgraphs),
        })
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let index = &self.buffer_index.0;
        run_steps(
            &self.steps,
            &self.dense_layers,
            &self.batch_norm_layers,
            (index[&self.inputs[0].id()], input),
            index[&self.output.id()],
        )
    }

    pub fn features(&self) -> &[String] {
        &self.features
    }

    pub fn feature_size(&self) -> usize {
        self.inputs.iter().map(InputBuffer::size).sum()
    }

    pub fn output_size(&self) -> usize {
        self.output.size()
    }

    pub fn export_to_instruction_model(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.to_instruction_model_info())
    }

    pub fn to_instruction_model_info(&self) -> InstructionModelExport {
        let mut ctx = CompileContext::new();
        for input in self.inputs.iter() {
            ctx.register_input(input.id(), input.size(), input.features());
        }
        let mut visited = std::mem::take(&mut ctx.visited_buffers);
        self.compile_buffer(&self.output, &mut ctx, &mut visited);
        ctx.visited_buffers = visited;
        ctx.into_export()
    }

    /// `visited` is the top-level map, or a fresh per-application map inside a sub-graph so that
    /// every application gets its own buffers while sharing weights.
    fn compile_buffer(
        &self,
        buffer: &DataBuffer,
        ctx: &mut CompileContext,
        visited: &mut HashMap<BufferId, usize>,
    ) -> usize {
        if let Some(&idx) = visited.get(&buffer.id()) {
            return idx;
        }
        let producer = buffer
            .producer()
            .expect("graph structure is validated in CompiledModel::new");
        let inputs: Vec<usize> = buffer
            .inputs()
            .iter()
            .map(|b| self.compile_buffer(b, ctx, visited))
            .collect();
        let idx = match producer {
            Producer::Op(op) => self.compile_operation(op, &inputs, ctx),
            Producer::SubGraph {
                input_ids, output, ..
            } => {
                let mut local = input_ids.iter().copied().zip(inputs).collect();
                self.compile_buffer(output, ctx, &mut local)
            }
        };
        visited.insert(buffer.id(), idx);
        idx
    }

    fn compile_operation(
        &self,
        op: &Operation,
        inputs: &[usize],
        ctx: &mut CompileContext,
    ) -> usize {
        let meta = &self.export_metadata.0;
        match op {
            Operation::Dense { id, .. } if ctx.get_weight_index(*id).is_none() => {
                let layer = &self.dense_layers[meta.dense[id]];
                ctx.store_weights(*id, layer.weights_to_vec(), layer.bias_to_vec());
            }
            Operation::BatchNorm { id, .. } => {
                let (center, std, beta) =
                    self.batch_norm_layers[meta.batch_norms[id]].normalization_params();
                return op.compile_batch_norm(inputs, ctx, &center, &std, &beta);
            }
            _ => {}
        }
        op.compile(inputs, ctx)
    }

    /// Extracts the sub-model for `graph` (applied somewhere in this model) with current weights.
    pub fn submodel(&self, graph: &ModelGraph) -> SubModel<B> {
        let mapping = self
            .subgraph_mappings
            .get(&graph.id())
            .expect("Sub-graph not found in compiled model");
        SubModel {
            dense_layers: mapping
                .dense_layer_indices
                .iter()
                .map(|&i| self.dense_layers[i].clone())
                .collect(),
            batch_norm_layers: mapping
                .batch_norm_layer_indices
                .iter()
                .map(|&i| self.batch_norm_layers[i].clone())
                .collect(),
            steps: mapping.steps.clone(),
            input: mapping.input,
            output: mapping.output,
        }
    }
}

/// A sub-model extracted from a [`CompiledModel`] with cloned weights.
#[derive(Debug, Clone)]
pub struct SubModel<B: Backend> {
    dense_layers: Vec<Dense<B>>,
    batch_norm_layers: Vec<BatchNormLayer<B>>,
    steps: Vec<Step>,
    input: usize,
    output: usize,
}

impl<B: Backend> SubModel<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        run_steps(
            &self.steps,
            &self.dense_layers,
            &self.batch_norm_layers,
            (self.input, input),
            self.output,
        )
    }
}

/// Flattens the graph (inlining sub-graphs) into steps and layer specs.
#[derive(Default)]
struct GraphBuilder {
    num_buffers: usize,
    steps: Vec<Step>,
    /// Per layer index: (op, input size, output size, activation).
    dense: Vec<(OpId, usize, usize, Activation)>,
    /// Per layer index: (op, features, epsilon, center, scale).
    batch_norms: Vec<(OpId, usize, f32, bool, bool)>,
    subgraphs: HashMap<GraphId, SubgraphMapping>,
}

impl GraphBuilder {
    fn visit(
        &mut self,
        buffer: &DataBuffer,
        visited: &mut HashMap<BufferId, usize>,
        in_subgraph: bool,
    ) -> Result<usize, ModelError> {
        if let Some(&idx) = visited.get(&buffer.id()) {
            return Ok(idx);
        }
        let Some(producer) = buffer.producer() else {
            return Err(ModelError::InvalidGraph(if in_subgraph {
                format!(
                    "Sub-graph buffer {} has no producer and is not a remapped input",
                    buffer.id()
                )
            } else {
                "Buffer has no producer and is not a registered input".to_string()
            }));
        };
        let inputs = buffer
            .inputs()
            .iter()
            .map(|b| self.visit(b, visited, in_subgraph))
            .collect::<Result<Vec<_>, _>>()?;

        let idx = match producer {
            Producer::Op(op) => {
                let input_size = buffer.inputs().first().map_or(0, DataBuffer::size);
                let kind = self.step_kind(op, input_size);
                let output = self.num_buffers;
                self.num_buffers += 1;
                self.steps.push(Step {
                    kind,
                    inputs,
                    output,
                });
                output
            }
            Producer::SubGraph {
                input_ids,
                output,
                graph_id,
                ..
            } => {
                let start = self.steps.len();
                let mut local = input_ids
                    .iter()
                    .copied()
                    .zip(inputs.iter().copied())
                    .collect();
                let out = self.visit(output, &mut local, true)?;
                if let Some(&input) = inputs.first() {
                    self.subgraphs
                        .entry(*graph_id)
                        .or_insert_with(|| SubgraphMapping::new(&self.steps[start..], input, out));
                }
                out
            }
        };
        visited.insert(buffer.id(), idx);
        Ok(idx)
    }

    fn step_kind(&mut self, op: &Operation, input_size: usize) -> StepKind {
        fn layer<T>(layers: &mut Vec<T>, id: OpId, op_id: impl Fn(&T) -> OpId, new: T) -> usize {
            layers
                .iter()
                .position(|l| op_id(l) == id)
                .unwrap_or_else(|| {
                    layers.push(new);
                    layers.len() - 1
                })
        }
        match op {
            Operation::Dense {
                id,
                output_size,
                activation,
            } => StepKind::Dense(layer(
                &mut self.dense,
                *id,
                |d| d.0,
                (*id, input_size, *output_size, *activation),
            )),
            Operation::BatchNorm {
                id,
                epsilon,
                center,
                scale,
                ..
            } => StepKind::BatchNorm(layer(
                &mut self.batch_norms,
                *id,
                |b| b.0,
                (*id, input_size, *epsilon, *center, *scale),
            )),
            Operation::Add { .. } => StepKind::Add,
            Operation::Multiply { .. } => StepKind::Multiply,
            Operation::Concat { .. } => StepKind::Concat,
            Operation::Softmax { .. } => StepKind::Activation(Activation::Softmax),
            Operation::Scale { vector, .. } => StepKind::Scale(vector.clone()),
            Operation::Shift { vector, .. } => StepKind::Shift(vector.clone()),
            Operation::StandaloneActivation { activation, .. } => StepKind::Activation(*activation),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::operation::ops;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    fn compile(input: InputBuffer, output: DataBuffer) -> CompiledModel<TestBackend> {
        CompiledModel::new(vec![input], output, &Default::default()).unwrap()
    }

    fn run(model: &CompiledModel<TestBackend>, rows: &[[f32; 4]]) -> Tensor<TestBackend, 2> {
        let data: Vec<f32> = rows.concat();
        let x = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &Default::default());
        model.forward(x.reshape([rows.len(), 4]))
    }

    #[test]
    fn softmax_forward_and_export() {
        let input = InputBuffer::new(4);
        let output = ops::softmax(ops::dense(3, Activation::None, input.buffer()));
        let model = compile(input, output);
        let out: Vec<f32> = run(&model, &[[1.0, 2.0, 3.0, 4.0]])
            .to_data()
            .to_vec()
            .unwrap();
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let json = model.export_to_instruction_model().unwrap();
        assert!(json.contains("\"activation\": \"SOFTMAX\""));
        assert_eq!(
            model.to_instruction_model_info().buffer_sizes,
            vec![4, 3, 3]
        );
    }

    #[test]
    fn batch_norm_initial_state() {
        let device = Default::default();
        let eps = 1e-3f32;
        let layer = BatchNormLayer::<TestBackend>::new(4, eps, true, true, &device);
        let (center, std, beta) = layer.normalization_params();
        let s = 1.0 / (1.0 + eps).sqrt();
        assert_eq!((center, beta), (vec![0.0; 4], vec![0.0; 4]));
        assert!(std.iter().all(|v| (v - s).abs() < 1e-6));

        let x = Tensor::<TestBackend, 2>::from_floats([[1.0, -2.0, 0.0, 4.0]], &device);
        let out: Vec<f32> = layer.forward(x).to_data().to_vec().unwrap();
        for (o, e) in out.iter().zip([1.0, -2.0, 0.0, 4.0]) {
            assert!((o - e * s).abs() < 1e-5);
        }
    }

    #[test]
    fn shared_batch_norm_exports_one_parameter_set() {
        let input = InputBuffer::new(4);
        let bn = Operation::batch_norm_default();
        let x = input.buffer();
        let merged = ops::add(vec![bn.apply(x.clone()), bn.apply(x)]);
        let model = compile(input, ops::dense(2, Activation::None, merged));
        assert_eq!(model.batch_norm_layers.len(), 1);
        assert_eq!(run(&model, &[[1.0; 4], [2.0; 4], [3.0; 4]]).dims(), [3, 2]);
        assert_eq!(model.to_instruction_model_info().parameters.len(), 3);
    }

    #[test]
    fn unregistered_input_is_an_error() {
        let input = InputBuffer::new(2);
        let stray = InputBuffer::new(2);
        let output = ops::add(vec![input.buffer(), stray.buffer()]);
        let err = CompiledModel::<TestBackend>::new(vec![input], output, &Default::default());
        assert!(matches!(err, Err(ModelError::InvalidGraph(_))));
    }

    #[test]
    fn submodel_of_graph_applied_twice() {
        let sub_input = InputBuffer::new(4);
        let sub_out = ops::dense(4, Activation::Tanh, ops::batch_norm(sub_input.buffer()));
        let sub = ModelGraph::new(vec![sub_input], sub_out);

        let input = InputBuffer::new(4);
        let once = sub.apply_single(input.buffer());
        let twice = sub.apply_single(once.clone());
        let model = compile(input, ops::concat(vec![once, twice]));
        let submodel = model.submodel(&sub);

        let x = [[0.5, -1.0, 2.0, 0.0]];
        let full: Vec<f32> = run(&model, &x).to_data().to_vec().unwrap();
        let first: Vec<f32> = submodel
            .forward(Tensor::from_floats(x, &Default::default()))
            .to_data()
            .to_vec()
            .unwrap();
        assert_eq!(&full[..4], first.as_slice());
    }

    mod training {
        use super::*;
        use burn::backend::Autodiff;
        use burn::optim::{AdamConfig, GradientsParams, Optimizer};
        use burn::tensor::ElementConversion;

        type TrainingBackend = Autodiff<NdArray>;

        #[test]
        fn shared_dense_model_learns() {
            let device = Default::default();
            let input = InputBuffer::new(4);
            let x = input.buffer();
            let shared = Operation::dense(4, Activation::Relu);
            let scaled = ops::scale(vec![2.0; 4], x.clone());
            let joined = ops::concat(vec![shared.apply(x), shared.apply(scaled)]);
            let output = ops::dense(1, Activation::None, joined);
            let mut model =
                CompiledModel::<TrainingBackend>::new(vec![input], output, &device).unwrap();
            assert_eq!(model.dense_layers.len(), 2);

            let mut optimizer = AdamConfig::new().init();
            let data = [
                ([1.0f32, 2.0, 3.0, 4.0], 2.5f32),
                ([0.0; 4], 0.0),
                ([4.0; 4], 4.0),
                ([-1.0, 1.0, -1.0, 1.0], 0.0),
            ];
            let mut epoch_losses = Vec::new();
            for _ in 0..20 {
                let mut total = 0.0;
                for (x, y) in &data {
                    let pred = model.forward(Tensor::from_floats([*x], &device));
                    let diff = pred - Tensor::from_floats([[*y]], &device);
                    let loss = (diff.clone() * diff).mean();
                    total += loss.clone().into_scalar().elem::<f32>();
                    let grads = GradientsParams::from_grads(loss.backward(), &model);
                    model = optimizer.step(0.01, model, grads);
                }
                epoch_losses.push(total);
            }
            assert!(epoch_losses[19] < epoch_losses[0], "{epoch_losses:?}");
        }
    }
}
