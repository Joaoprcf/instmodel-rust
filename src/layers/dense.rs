//! Dense (fully connected) layer implementation.

use crate::layers::Activation;
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Tensor, backend::Backend},
};

/// Configuration for a Dense layer.
#[derive(Debug, Clone)]
pub struct DenseConfig {
    /// Number of input features.
    pub input_size: usize,
    /// Number of output features.
    pub output_size: usize,
    /// Activation function to apply after the linear transformation.
    pub activation: Activation,
}

impl DenseConfig {
    /// Creates a new DenseConfig.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            output_size,
            activation: Activation::None,
        }
    }

    /// Sets the activation function.
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Initializes the Dense layer with the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Dense<B> {
        let linear = LinearConfig::new(self.input_size, self.output_size).init(device);

        Dense {
            linear,
            input_size: self.input_size,
            output_size: self.output_size,
            activation_id: self.activation.to_id(),
        }
    }
}

/// A dense (fully connected) layer with optional activation.
///
/// This corresponds to the `Dense` ComputationOp in the Python `instmodel` package.
/// It performs: output = activation(input @ weights.T + bias)
#[derive(Module, Debug)]
pub struct Dense<B: Backend> {
    /// The underlying linear transformation.
    linear: Linear<B>,
    /// Input size (constant metadata).
    input_size: usize,
    /// Output size (constant metadata).
    output_size: usize,
    /// Activation function ID (0=None, 1=Relu, 2=Sigmoid, 3=Tanh, 4=Softmax).
    activation_id: u8,
}

impl<B: Backend> Dense<B> {
    /// Performs the forward pass.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let output = self.linear.forward(input);
        Activation::from_id(self.activation_id).apply(output)
    }

    /// Returns the input size of this layer.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the output size of this layer.
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Returns the activation function.
    pub fn activation(&self) -> Activation {
        Activation::from_id(self.activation_id)
    }

    /// Extracts the weights as a 2D vector (shape: [output_size, input_size]).
    ///
    /// This format matches the instruction model JSON format.
    pub fn weights_to_vec(&self) -> Vec<Vec<f32>> {
        let weight_tensor = self.linear.weight.val();
        let weights_data = weight_tensor.to_data();
        let flat: Vec<f32> = weights_data.to_vec().unwrap();

        // Burn stores tensors in column-major order, so we convert to row-major
        let mut result = vec![vec![0.0f32; self.input_size]; self.output_size];
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                result[i][j] = flat[i + j * self.output_size];
            }
        }
        result
    }

    /// Extracts the bias as a 1D vector.
    pub fn bias_to_vec(&self) -> Vec<f32> {
        match &self.linear.bias {
            Some(bias) => bias.val().to_data().to_vec().unwrap(),
            None => vec![0.0; self.output_size],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_dense_config_creation() {
        let config = DenseConfig::new(10, 5).with_activation(Activation::Relu);

        assert_eq!(config.input_size, 10);
        assert_eq!(config.output_size, 5);
        assert_eq!(config.activation, Activation::Relu);
    }

    #[test]
    fn test_dense_layer_creation() {
        let device = <TestBackend as Backend>::Device::default();
        let dense: Dense<TestBackend> = DenseConfig::new(4, 2)
            .with_activation(Activation::Sigmoid)
            .init(&device);

        assert_eq!(dense.input_size(), 4);
        assert_eq!(dense.output_size(), 2);
        assert_eq!(dense.activation(), Activation::Sigmoid);
    }

    #[test]
    fn test_dense_forward_shape() {
        let device = <TestBackend as Backend>::Device::default();
        let dense: Dense<TestBackend> = DenseConfig::new(4, 2).init(&device);

        let input = Tensor::<TestBackend, 2>::zeros([3, 4], &device);
        let output = dense.forward(input);

        assert_eq!(output.dims(), [3, 2]);
    }

    #[test]
    fn test_dense_weights_extraction() {
        let device = <TestBackend as Backend>::Device::default();
        let dense: Dense<TestBackend> = DenseConfig::new(3, 2).init(&device);

        let weights = dense.weights_to_vec();
        let bias = dense.bias_to_vec();

        assert_eq!(weights.len(), 2); // output_size rows
        assert_eq!(weights[0].len(), 3); // input_size columns
        assert_eq!(bias.len(), 2); // output_size
    }
}
