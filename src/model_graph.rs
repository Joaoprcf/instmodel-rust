//! Sequential Dense-only model, mirroring Python `instmodel.ModelGraph`.

use crate::errors::ModelError;
use crate::layers::{Activation, Dense, DenseConfig};
use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
};
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct ModelGraphConfig {
    pub features: Vec<String>,
    pub layer_configs: Vec<DenseConfig>,
}

impl ModelGraphConfig {
    pub fn new(features: Vec<String>) -> Self {
        Self {
            features,
            layer_configs: Vec::new(),
        }
    }

    /// Features named `feature_0..feature_{size-1}`.
    pub fn with_feature_size(size: usize) -> Self {
        let features = (0..size).map(|i| format!("feature_{}", i)).collect();
        Self {
            features,
            layer_configs: Vec::new(),
        }
    }

    pub fn dense(mut self, output_size: usize, activation: Activation) -> Self {
        let input_size = self
            .layer_configs
            .last()
            .map_or(self.features.len(), |c| c.output_size);
        self.layer_configs
            .push(DenseConfig::new(input_size, output_size).with_activation(activation));
        self
    }

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

/// A sequential stack of Dense layers.
#[derive(Module, Debug)]
pub struct ModelGraph<B: Backend> {
    features: Vec<String>,
    layers: Vec<Dense<B>>,
}

impl<B: Backend> ModelGraph<B> {
    pub fn new_config(features: Vec<String>) -> ModelGraphConfig {
        ModelGraphConfig::new(features)
    }

    pub fn config_with_size(size: usize) -> ModelGraphConfig {
        ModelGraphConfig::with_feature_size(size)
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }

    pub fn features(&self) -> &[String] {
        &self.features
    }

    pub fn feature_size(&self) -> usize {
        self.features.len()
    }

    pub fn output_size(&self) -> usize {
        self.layers.last().map(|l| l.output_size()).unwrap_or(0)
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn export_to_instruction_model(&self) -> Result<String, serde_json::Error> {
        let export = self.to_instruction_model_info();
        serde_json::to_string_pretty(&export)
    }

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

    /// Runs each input through the model to build `validation_data`.
    pub fn generate_validation_data(
        &self,
        inputs: &[Vec<f32>],
        device: &B::Device,
    ) -> ValidationDataExport {
        let mut expected_outputs = Vec::new();

        for input in inputs {
            let input_tensor =
                Tensor::<B, 1>::from_floats(input.as_slice(), device).reshape([1, input.len()]);

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

/// Export format for the sequential model (DOT instructions only).
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
    fn build_forward_export() {
        let device = Default::default();
        let model: ModelGraph<TestBackend> = ModelGraphConfig::with_feature_size(2)
            .dense(3, Activation::Relu)
            .dense(1, Activation::Sigmoid)
            .build(&device)
            .unwrap();
        assert_eq!(
            (
                model.feature_size(),
                model.output_size(),
                model.num_layers()
            ),
            (2, 1, 2)
        );
        assert_eq!(model.forward(Tensor::zeros([5, 2], &device)).dims(), [5, 1]);

        let export = model.to_instruction_model_info();
        assert_eq!(export.buffer_sizes, vec![2, 3, 1]);
        assert_eq!(export.features.as_deref(), Some(model.features()));
        let json: serde_json::Value =
            serde_json::from_str(&model.export_to_instruction_model().unwrap()).unwrap();
        assert_eq!(
            json["instructions"][1],
            serde_json::json!({"type": "DOT", "input": 1, "output": 2, "weights": 1, "activation": "SIGMOID"})
        );
        assert!(json.get("parameters").is_none());

        let validation = model.generate_validation_data(&[vec![0.0, 1.0]], &device);
        assert_eq!(validation.expected_outputs[0].len(), 1);
    }

    #[test]
    fn build_errors() {
        let device = Default::default();
        let no_layers = ModelGraphConfig::with_feature_size(4).build::<TestBackend>(&device);
        assert!(matches!(no_layers, Err(ModelError::NoLayers)));
        let no_input = ModelGraphConfig::new(vec![])
            .dense(4, Activation::Relu)
            .build::<TestBackend>(&device);
        assert!(matches!(no_input, Err(ModelError::NoInputBuffer)));
    }
}
