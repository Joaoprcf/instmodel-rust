use std::collections::HashMap;

use super::buffer::BufferId;
use super::operation::OpId;

/// One instruction of the exported JSON model.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum InstructionExport {
    #[serde(rename = "DOT")]
    Dot {
        input: usize,
        output: usize,
        weights: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        activation: Option<String>,
    },
    #[serde(rename = "COPY_MASKED")]
    CopyMasked {
        input: usize,
        output: usize,
        indexes: Vec<usize>,
    },
    /// Copies the whole input into `output` starting at `internal_index`.
    #[serde(rename = "COPY")]
    Copy {
        input: usize,
        output: usize,
        internal_index: usize,
    },
    #[serde(rename = "ADD_ELEMENTWISE_BUFFERS")]
    AddBuffers { input: Vec<usize>, output: usize },
    #[serde(rename = "MULTIPLY_ELEMENTWISE_BUFFERS")]
    MultiplyBuffers { input: Vec<usize>, output: usize },
    /// In-place.
    #[serde(rename = "ACTIVATION")]
    Activation { input: usize, activation: String },
    /// In-place multiply by `parameters[parameters]`.
    #[serde(rename = "MUL_ELEMENTWISE")]
    MulElementwise { input: usize, parameters: usize },
    /// In-place add of `parameters[parameters]`.
    #[serde(rename = "ADD_ELEMENTWISE")]
    AddElementwise { input: usize, parameters: usize },
}

/// Accumulates buffers, instructions and deduplicated weights/parameters during export.
#[derive(Default)]
pub struct CompileContext {
    pub buffer_sizes: Vec<usize>,
    pub instructions: Vec<InstructionExport>,
    /// `[output][input]` per Dense layer.
    pub weights: Vec<Vec<Vec<f32>>>,
    pub bias: Vec<Vec<f32>>,
    pub parameters: Vec<Vec<f32>>,
    pub visited_buffers: HashMap<BufferId, usize>,
    /// Dense op → weight index (shared ops reuse it).
    pub visited_ops: HashMap<OpId, usize>,
    /// Scale/Shift/BatchNorm op → first parameter index.
    pub visited_param_ops: HashMap<OpId, usize>,
    pub features: Vec<String>,
}

impl CompileContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_input(
        &mut self,
        buffer_id: BufferId,
        size: usize,
        features: Option<&[String]>,
    ) -> usize {
        let idx = self.allocate_buffer(size);
        self.visited_buffers.insert(buffer_id, idx);
        if let Some(feats) = features {
            self.features.extend_from_slice(feats);
        }
        idx
    }

    pub fn allocate_buffer(&mut self, size: usize) -> usize {
        self.buffer_sizes.push(size);
        self.buffer_sizes.len() - 1
    }

    pub fn get_weight_index(&self, op_id: OpId) -> Option<usize> {
        self.visited_ops.get(&op_id).copied()
    }

    /// Returns the op's weight index, reserving an empty slot if it has none yet.
    pub fn get_or_create_weight_index(&mut self, op_id: OpId) -> usize {
        match self.get_weight_index(op_id) {
            Some(idx) => idx,
            None => self.store_weights(op_id, vec![], vec![]),
        }
    }

    pub fn store_weights(&mut self, op_id: OpId, weights: Vec<Vec<f32>>, bias: Vec<f32>) -> usize {
        let idx = self.weights.len();
        self.weights.push(weights);
        self.bias.push(bias);
        self.visited_ops.insert(op_id, idx);
        idx
    }

    pub fn set_weights(&mut self, index: usize, weights: Vec<Vec<f32>>, bias: Vec<f32>) {
        self.weights[index] = weights;
        self.bias[index] = bias;
    }

    pub fn get_or_store_parameters(&mut self, op_id: OpId, params: &[f32]) -> usize {
        *self.visited_param_ops.entry(op_id).or_insert_with(|| {
            self.parameters.push(params.to_vec());
            self.parameters.len() - 1
        })
    }

    pub fn add_instruction(&mut self, instruction: InstructionExport) {
        self.instructions.push(instruction);
    }

    pub fn into_export(self) -> InstructionModelExport {
        InstructionModelExport {
            features: (!self.features.is_empty()).then_some(self.features),
            feature_size: Some(self.buffer_sizes.first().copied().unwrap_or(0)),
            buffer_sizes: self.buffer_sizes,
            instructions: self.instructions,
            weights: self.weights,
            bias: self.bias,
            parameters: self.parameters,
        }
    }
}

/// The complete exported model (serialises to the shared instruction-model JSON).
#[derive(Debug, Clone, serde::Serialize)]
pub struct InstructionModelExport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_size: Option<usize>,
    pub buffer_sizes: Vec<usize>,
    pub instructions: Vec<InstructionExport>,
    pub weights: Vec<Vec<Vec<f32>>>,
    pub bias: Vec<Vec<f32>>,
    pub parameters: Vec<Vec<f32>>,
}
