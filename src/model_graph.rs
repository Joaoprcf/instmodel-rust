//! ModelGraph - the main container for building and training neural networks.
//!
//! This module provides the `ModelGraph` struct which mirrors the Python
//! `instmodel.ModelGraph` class, allowing users to define neural network
//! architectures, train them, and export to instruction model format.

use crate::errors::ModelError;
use crate::layers::{Activation, Dense, DenseConfig};
use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
};
use serde::Serialize;

/// Configuration for building a ModelGraph.
#[derive(Debug, Clone)]
pub struct ModelGraphConfig {
    /// Names of input features.
    pub features: Vec<String>,
    /// Layer configurations.
    pub layer_configs: Vec<DenseConfig>,
}

impl ModelGraphConfig {
    /// Creates a new ModelGraphConfig with the specified features.
    pub fn new(features: Vec<String>) -> Self {
        Self {
            features,
            layer_configs: Vec::new(),
        }
    }

    /// Creates a ModelGraphConfig with numbered features.
    pub fn with_feature_size(size: usize) -> Self {
        let features = (0..size).map(|i| format!("feature_{}", i)).collect();
        Self {
            features,
            layer_configs: Vec::new(),
        }
    }

    /// Adds a dense layer to the configuration.
    pub fn dense(mut self, output_size: usize, activation: Activation) -> Self {
        let input_size = if self.layer_configs.is_empty() {
            self.features.len()
        } else {
            self.layer_configs.last().unwrap().output_size
        };

        self.layer_configs
            .push(DenseConfig::new(input_size, output_size).with_activation(activation));
        self
    }

    /// Builds the ModelGraph with the given device.
    pub fn build<B: Backend>(&self, device: &B::Device) -> Result<ModelGraph<B>, ModelError> {
        if self.features.is_empty() {
            return Err(ModelError::NoInputBuffer);
        }

        if self.layer_configs.is_empty() {
            return Err(ModelError::NoLayers);
        }

        let layers: Vec<Dense<B>> = self
            .layer_configs
            .iter()
            .map(|config| config.init(device))
            .collect();

        Ok(ModelGraph {
            features: self.features.clone(),
            layers,
        })
    }
}

/// The main model container for building and training neural networks.
///
/// Similar to Python's `instmodel.ModelGraph`, this struct allows you to:
/// - Define sequential neural network architectures
/// - Perform forward passes
/// - Export to instruction model JSON format
#[derive(Module, Debug)]
pub struct ModelGraph<B: Backend> {
    /// Names of input features (stored as constant).
    features: Vec<String>,
    /// The dense layers in sequence.
    layers: Vec<Dense<B>>,
}

impl<B: Backend> ModelGraph<B> {
    /// Creates a new configuration builder.
    pub fn new_config(features: Vec<String>) -> ModelGraphConfig {
        ModelGraphConfig::new(features)
    }

    /// Creates a configuration with a specified feature size.
    pub fn config_with_size(size: usize) -> ModelGraphConfig {
        ModelGraphConfig::with_feature_size(size)
    }

    /// Performs a forward pass through all layers.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }

    /// Returns the input feature names.
    pub fn features(&self) -> &[String] {
        &self.features
    }

    /// Returns the number of input features.
    pub fn feature_size(&self) -> usize {
        self.features.len()
    }

    /// Returns the output size of the model.
    pub fn output_size(&self) -> usize {
        self.layers.last().map(|l| l.output_size()).unwrap_or(0)
    }

    /// Returns the number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Exports the model to instruction model JSON format.
    ///
    /// This format is compatible with `instmodel-rust-inference`.
    pub fn export_to_instruction_model(&self) -> Result<String, serde_json::Error> {
        let export = self.to_instruction_model_info();
        serde_json::to_string_pretty(&export)
    }

    /// Converts the model to an InstructionModelExport struct.
    pub fn to_instruction_model_info(&self) -> InstructionModelExport {
        let mut buffer_sizes = vec![self.feature_size()];
        let mut instructions = Vec::new();
        let mut weights = Vec::new();
        let mut bias = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            buffer_sizes.push(layer.output_size());
            weights.push(layer.weights_to_vec());
            bias.push(layer.bias_to_vec());

            instructions.push(InstructionExport::Dot {
                r#type: "DOT".to_string(),
                input: i,
                output: i + 1,
                weights: i,
                activation: layer.activation().to_instruction_name().map(String::from),
            });
        }

        InstructionModelExport {
            features: Some(self.features.clone()),
            feature_size: Some(self.feature_size()),
            buffer_sizes,
            instructions,
            weights,
            bias,
            parameters: Vec::new(),
            maps: Vec::new(),
            validation_data: None,
        }
    }

    /// Generates validation data by running inference on sample inputs.
    pub fn generate_validation_data(
        &self,
        inputs: &[Vec<f32>],
        device: &B::Device,
    ) -> ValidationDataExport {
        let mut expected_outputs = Vec::new();

        for input in inputs {
            let data: Vec<f32> = input.clone();
            let input_tensor =
                Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([1, input.len()]);

            let output = self.forward(input_tensor);
            let output_data: Vec<f32> = output.to_data().to_vec().unwrap();
            expected_outputs.push(output_data);
        }

        ValidationDataExport {
            inputs: inputs.to_vec(),
            expected_outputs,
        }
    }
}

/// Export format for instruction model JSON.
#[derive(Debug, Clone, Serialize)]
pub struct InstructionModelExport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_size: Option<usize>,
    pub buffer_sizes: Vec<usize>,
    pub instructions: Vec<InstructionExport>,
    pub weights: Vec<Vec<Vec<f32>>>,
    pub bias: Vec<Vec<f32>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub parameters: Vec<Vec<f32>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub maps: Vec<std::collections::HashMap<String, Vec<f32>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_data: Option<ValidationDataExport>,
}

/// Export format for instructions.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum InstructionExport {
    Dot {
        r#type: String,
        input: usize,
        output: usize,
        weights: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        activation: Option<String>,
    },
}

/// Export format for validation data.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationDataExport {
    pub inputs: Vec<Vec<f32>>,
    pub expected_outputs: Vec<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_model_graph_config_creation() {
        let config = ModelGraphConfig::with_feature_size(4)
            .dense(8, Activation::Relu)
            .dense(1, Activation::Sigmoid);

        assert_eq!(config.features.len(), 4);
        assert_eq!(config.layer_configs.len(), 2);
    }

    #[test]
    fn test_model_graph_build() {
        let device = <TestBackend as Backend>::Device::default();
        let model: ModelGraph<TestBackend> = ModelGraphConfig::with_feature_size(4)
            .dense(8, Activation::Relu)
            .dense(1, Activation::Sigmoid)
            .build(&device)
            .expect("Failed to build model");

        assert_eq!(model.feature_size(), 4);
        assert_eq!(model.output_size(), 1);
        assert_eq!(model.num_layers(), 2);
    }

    #[test]
    fn test_model_graph_forward() {
        let device = <TestBackend as Backend>::Device::default();
        let model: ModelGraph<TestBackend> = ModelGraphConfig::with_feature_size(4)
            .dense(8, Activation::Relu)
            .dense(2, Activation::None)
            .build(&device)
            .expect("Failed to build model");

        let input = Tensor::<TestBackend, 2>::zeros([3, 4], &device);
        let output = model.forward(input);

        assert_eq!(output.dims(), [3, 2]);
    }

    #[test]
    fn test_model_graph_export() {
        let device = <TestBackend as Backend>::Device::default();
        let model: ModelGraph<TestBackend> = ModelGraphConfig::with_feature_size(2)
            .dense(3, Activation::Relu)
            .dense(1, Activation::Sigmoid)
            .build(&device)
            .expect("Failed to build model");

        let export = model.to_instruction_model_info();

        assert_eq!(export.buffer_sizes, vec![2, 3, 1]);
        assert_eq!(export.instructions.len(), 2);
        assert_eq!(export.weights.len(), 2);
        assert_eq!(export.bias.len(), 2);
    }

    #[test]
    fn test_model_graph_no_layers_error() {
        let device = <TestBackend as Backend>::Device::default();
        let result: Result<ModelGraph<TestBackend>, _> =
            ModelGraphConfig::with_feature_size(4).build(&device);

        assert!(matches!(result, Err(ModelError::NoLayers)));
    }

    #[test]
    fn test_model_graph_no_input_error() {
        let device = <TestBackend as Backend>::Device::default();
        let result: Result<ModelGraph<TestBackend>, _> = ModelGraphConfig::new(vec![])
            .dense(4, Activation::Relu)
            .build(&device);

        assert!(matches!(result, Err(ModelError::NoInputBuffer)));
    }
}
