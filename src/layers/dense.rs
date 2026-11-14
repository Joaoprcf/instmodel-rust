use crate::layers::Activation;
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Tensor, backend::Backend},
};

#[derive(Debug, Clone)]
pub struct DenseConfig {
    pub input_size: usize,
    pub output_size: usize,
    pub activation: Activation,
}

impl DenseConfig {
    /// No activation by default.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            output_size,
            activation: Activation::None,
        }
    }

    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Dense<B> {
        Dense {
            linear: LinearConfig::new(self.input_size, self.output_size).init(device),
            input_size: self.input_size,
            output_size: self.output_size,
            activation_id: self.activation.to_id(),
        }
    }
}

/// Fully connected layer: `activation(input @ W + b)`.
#[derive(Module, Debug)]
pub struct Dense<B: Backend> {
    linear: Linear<B>,
    input_size: usize,
    output_size: usize,
    /// [`Activation::to_id`]; stored as a primitive so it is a Module constant.
    activation_id: u8,
}

impl<B: Backend> Dense<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.activation().apply(self.linear.forward(input))
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub fn activation(&self) -> Activation {
        Activation::from_id(self.activation_id)
    }

    /// Weights as `[output][input]`, the instruction-model layout.
    pub fn weights_to_vec(&self) -> Vec<Vec<f32>> {
        // Burn's Linear stores the weight row-major as [input, output].
        let flat: Vec<f32> = self.linear.weight.val().to_data().to_vec().unwrap();
        (0..self.output_size)
            .map(|o| {
                (0..self.input_size)
                    .map(|i| flat[i * self.output_size + o])
                    .collect()
            })
            .collect()
    }

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

    #[test]
    fn exported_weights_reproduce_forward() {
        let device = Default::default();
        let dense: Dense<NdArray> = DenseConfig::new(3, 2)
            .with_activation(Activation::Sigmoid)
            .init(&device);
        assert_eq!((dense.input_size(), dense.output_size()), (3, 2));
        assert_eq!(dense.activation(), Activation::Sigmoid);

        let x = [0.5f32, -1.0, 2.0];
        let out: Vec<f32> = dense
            .forward(Tensor::from_floats([x], &device))
            .to_data()
            .to_vec()
            .unwrap();
        let (w, b) = (dense.weights_to_vec(), dense.bias_to_vec());
        for o in 0..2 {
            let z: f32 = b[o] + (0..3).map(|i| w[o][i] * x[i]).sum::<f32>();
            assert!((out[o] - 1.0 / (1.0 + (-z).exp())).abs() < 1e-6);
        }
    }
}
