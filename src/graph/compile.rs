//! Compilation context for converting the graph to instruction model format.

use std::collections::HashMap;

use super::buffer::BufferId;
use super::operation::OpId;

/// Export format for instructions.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum InstructionExport {
    /// Dense/Dot product instruction.
    #[serde(rename = "DOT")]
    Dot {
        input: usize,
        output: usize,
        weights: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        activation: Option<String>,
    },
    /// Copy with column selection.
    #[serde(rename = "COPY_MASKED")]
    CopyMasked {
        input: usize,
        output: usize,
        indexes: Vec<usize>,
    },
    /// Copy entire buffer.
    #[serde(rename = "COPY")]
    Copy {
        input: usize,
        output: usize,
        internal_index: usize,
    },
    /// Element-wise add of multiple buffers.
    #[serde(rename = "ADD_ELEMENTWISE_BUFFERS")]
    AddBuffers { input: Vec<usize>, output: usize },
    /// Element-wise multiply of multiple buffers.
    #[serde(rename = "MULTIPLY_ELEMENTWISE_BUFFERS")]
    MultiplyBuffers { input: Vec<usize>, output: usize },
    /// Activation function applied to a buffer (in-place).
    #[serde(rename = "ACTIVATION")]
    Activation { input: usize, activation: String },
    /// Element-wise multiply by a parameter vector (in-place).
    #[serde(rename = "MUL_ELEMENTWISE")]
    MulElementwise { input: usize, parameters: usize },
    /// Element-wise add a parameter vector (in-place).
    #[serde(rename = "ADD_ELEMENTWISE")]
    AddElementwise { input: usize, parameters: usize },
}

/// Context for compiling the graph to instruction model format.
///
/// This is similar to Python's `model_structure` + `weights_visited` combined.
pub struct CompileContext {
    /// Sizes of each buffer in the computation.
    pub buffer_sizes: Vec<usize>,
    /// List of instructions to execute.
    pub instructions: Vec<InstructionExport>,
    /// Weight matrices (shape: [output, input] per layer).
    pub weights: Vec<Vec<Vec<f32>>>,
    /// Bias vectors.
    pub bias: Vec<Vec<f32>>,
    /// Parameter vectors (for Scale/Shift operations).
    pub parameters: Vec<Vec<f32>>,
    /// Maps buffer IDs to their index in buffer_sizes.
    pub visited_buffers: HashMap<BufferId, usize>,
    /// Maps op IDs to their weight index (for weight sharing).
    pub visited_ops: HashMap<OpId, usize>,
    /// Maps op IDs to their parameter index (for Scale/Shift).
    pub visited_param_ops: HashMap<OpId, usize>,
    /// Feature names for input buffers.
    pub features: Vec<String>,
}

impl CompileContext {
    /// Creates a new empty compile context.
    pub fn new() -> Self {
        Self {
            buffer_sizes: Vec::new(),
            instructions: Vec::new(),
            weights: Vec::new(),
            bias: Vec::new(),
            parameters: Vec::new(),
            visited_buffers: HashMap::new(),
            visited_ops: HashMap::new(),
            visited_param_ops: HashMap::new(),
            features: Vec::new(),
        }
    }

    /// Registers an input buffer, returning its buffer index.
    pub fn register_input(
        &mut self,
        buffer_id: BufferId,
        size: usize,
        features: Option<&[String]>,
    ) -> usize {
        let idx = self.buffer_sizes.len();
        self.buffer_sizes.push(size);
        self.visited_buffers.insert(buffer_id, idx);

        if let Some(feats) = features {
            self.features.extend(feats.iter().cloned());
        }

        idx
    }

    /// Allocates a new output buffer, returning its index.
    pub fn allocate_buffer(&mut self, size: usize) -> usize {
        let idx = self.buffer_sizes.len();
        self.buffer_sizes.push(size);
        idx
    }

    /// Checks if an operation's weights have already been stored.
    /// Returns the weight index if so.
    pub fn get_weight_index(&self, op_id: OpId) -> Option<usize> {
        self.visited_ops.get(&op_id).copied()
    }

    /// Gets existing weight index or creates a placeholder for later filling.
    /// Used during compilation when weights will be filled by ModelGraph.
    pub fn get_or_create_weight_index(&mut self, op_id: OpId) -> usize {
        if let Some(idx) = self.visited_ops.get(&op_id) {
            return *idx;
        }
        // Create placeholder - will be filled by ModelGraph
        let idx = self.weights.len();
        self.weights.push(vec![]);
        self.bias.push(vec![]);
        self.visited_ops.insert(op_id, idx);
        idx
    }

    /// Stores weights for an operation and returns the weight index.
    pub fn store_weights(&mut self, op_id: OpId, weights: Vec<Vec<f32>>, bias: Vec<f32>) -> usize {
        let idx = self.weights.len();
        self.weights.push(weights);
        self.bias.push(bias);
        self.visited_ops.insert(op_id, idx);
        idx
    }

    /// Updates weights at a given index (for filling placeholders).
    pub fn set_weights(&mut self, index: usize, weights: Vec<Vec<f32>>, bias: Vec<f32>) {
        self.weights[index] = weights;
        self.bias[index] = bias;
    }

    /// Gets existing parameter index or stores new parameters.
    pub fn get_or_store_parameters(&mut self, op_id: OpId, params: &[f32]) -> usize {
        if let Some(&idx) = self.visited_param_ops.get(&op_id) {
            return idx;
        }
        let idx = self.parameters.len();
        self.parameters.push(params.to_vec());
        self.visited_param_ops.insert(op_id, idx);
        idx
    }

    /// Adds an instruction to the list.
    pub fn add_instruction(&mut self, instruction: InstructionExport) {
        self.instructions.push(instruction);
    }

    /// Converts the context into the export format.
    pub fn into_export(self) -> InstructionModelExport {
        InstructionModelExport {
            features: if self.features.is_empty() {
                None
            } else {
                Some(self.features)
            },
            feature_size: Some(self.buffer_sizes.first().copied().unwrap_or(0)),
            buffer_sizes: self.buffer_sizes,
            instructions: self.instructions,
            weights: self.weights,
            bias: self.bias,
            parameters: self.parameters,
        }
    }
}

impl Default for CompileContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Export format for the complete instruction model.
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
