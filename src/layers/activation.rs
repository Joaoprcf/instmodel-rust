//! Activation functions for neural network layers.

use burn::tensor::{Tensor, backend::Backend};
use serde::{Deserialize, Serialize};

/// Supported activation functions.
///
/// These mirror the activations available in the Python `instmodel` package
/// and the `instmodel-rust-inference` library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "UPPERCASE")]
pub enum Activation {
    /// No activation (identity function).
    #[default]
    None,
    /// Rectified Linear Unit: f(x) = max(0, x)
    Relu,
    /// Sigmoid: f(x) = 1 / (1 + exp(-x))
    Sigmoid,
    /// Hyperbolic tangent: f(x) = tanh(x)
    Tanh,
    /// Softmax normalization (across last dimension)
    Softmax,
    /// Square root: f(x) = sqrt(x) for x > 0, else 0
    Sqrt,
    /// Natural logarithm: f(x) = ln(x + 1) for x > 0, else 0
    Log,
    /// Base-10 logarithm: f(x) = log10(x + 1) for x > 0, else 0
    Log10,
    /// Complement: f(x) = 1 - x
    Inverse,
    /// Gaussian Error Linear Unit: f(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    Gelu,
}

impl Activation {
    /// Applies the activation function to a tensor.
    pub fn apply<B: Backend, const D: usize>(&self, tensor: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Activation::None => tensor,
            Activation::Relu => burn::tensor::activation::relu(tensor),
            Activation::Sigmoid => burn::tensor::activation::sigmoid(tensor),
            Activation::Tanh => burn::tensor::activation::tanh(tensor),
            Activation::Softmax => burn::tensor::activation::softmax(tensor, D - 1),
            Activation::Sqrt => {
                // sqrt(x) for x > 0, else 0
                let zeros = tensor.zeros_like();
                let mask = tensor.clone().greater_elem(0.0);
                // mask_where: where mask is true, use second arg; else keep self
                zeros.mask_where(mask, tensor.sqrt())
            }
            Activation::Log => {
                // ln(x + 1) for x > 0, else 0
                let zeros = tensor.zeros_like();
                let mask = tensor.clone().greater_elem(0.0);
                let result = (tensor + 1.0).log();
                zeros.mask_where(mask, result)
            }
            Activation::Log10 => {
                // log10(x + 1) for x > 0, else 0
                let zeros = tensor.zeros_like();
                let mask = tensor.clone().greater_elem(0.0);
                let ln_10 = 10.0_f64.ln() as f32;
                let result = (tensor + 1.0).log() / ln_10;
                zeros.mask_where(mask, result)
            }
            Activation::Inverse => {
                // 1 - x
                tensor.neg() + 1.0
            }
            Activation::Gelu => burn::tensor::activation::gelu(tensor),
        }
    }

    /// Returns the string name for export to instruction model format.
    pub fn to_instruction_name(&self) -> Option<&'static str> {
        match self {
            Activation::None => None,
            Activation::Relu => Some("RELU"),
            Activation::Sigmoid => Some("SIGMOID"),
            Activation::Tanh => Some("TANH"),
            Activation::Softmax => Some("SOFTMAX"),
            Activation::Sqrt => Some("SQRT"),
            Activation::Log => Some("LOG"),
            Activation::Log10 => Some("LOG10"),
            Activation::Inverse => Some("INVERSE"),
            Activation::Gelu => Some("GELU"),
        }
    }

    /// Creates an Activation from a string name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_uppercase().as_str() {
            "NONE" => Some(Activation::None),
            "RELU" => Some(Activation::Relu),
            "SIGMOID" => Some(Activation::Sigmoid),
            "TANH" => Some(Activation::Tanh),
            "SOFTMAX" => Some(Activation::Softmax),
            "SQRT" => Some(Activation::Sqrt),
            "LOG" => Some(Activation::Log),
            "LOG10" => Some(Activation::Log10),
            "INVERSE" => Some(Activation::Inverse),
            "GELU" => Some(Activation::Gelu),
            _ => None,
        }
    }

    /// Converts activation to a numeric ID for storage in Module.
    pub fn to_id(&self) -> u8 {
        match self {
            Activation::None => 0,
            Activation::Relu => 1,
            Activation::Sigmoid => 2,
            Activation::Tanh => 3,
            Activation::Softmax => 4,
            Activation::Sqrt => 5,
            Activation::Log => 6,
            Activation::Log10 => 7,
            Activation::Inverse => 8,
            Activation::Gelu => 9,
        }
    }

    /// Creates an Activation from a numeric ID.
    pub fn from_id(id: u8) -> Self {
        match id {
            0 => Activation::None,
            1 => Activation::Relu,
            2 => Activation::Sigmoid,
            3 => Activation::Tanh,
            4 => Activation::Softmax,
            5 => Activation::Sqrt,
            6 => Activation::Log,
            7 => Activation::Log10,
            8 => Activation::Inverse,
            9 => Activation::Gelu,
            _ => Activation::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_activation_to_instruction_name() {
        assert_eq!(Activation::None.to_instruction_name(), None);
        assert_eq!(Activation::Relu.to_instruction_name(), Some("RELU"));
        assert_eq!(Activation::Sigmoid.to_instruction_name(), Some("SIGMOID"));
        assert_eq!(Activation::Tanh.to_instruction_name(), Some("TANH"));
        assert_eq!(Activation::Softmax.to_instruction_name(), Some("SOFTMAX"));
        assert_eq!(Activation::Sqrt.to_instruction_name(), Some("SQRT"));
        assert_eq!(Activation::Log.to_instruction_name(), Some("LOG"));
        assert_eq!(Activation::Log10.to_instruction_name(), Some("LOG10"));
        assert_eq!(Activation::Inverse.to_instruction_name(), Some("INVERSE"));
        assert_eq!(Activation::Gelu.to_instruction_name(), Some("GELU"));
    }

    #[test]
    fn test_activation_from_name() {
        assert_eq!(Activation::from_name("relu"), Some(Activation::Relu));
        assert_eq!(Activation::from_name("SIGMOID"), Some(Activation::Sigmoid));
        assert_eq!(Activation::from_name("sqrt"), Some(Activation::Sqrt));
        assert_eq!(Activation::from_name("LOG"), Some(Activation::Log));
        assert_eq!(Activation::from_name("log10"), Some(Activation::Log10));
        assert_eq!(Activation::from_name("INVERSE"), Some(Activation::Inverse));
        assert_eq!(Activation::from_name("gelu"), Some(Activation::Gelu));
        assert_eq!(Activation::from_name("GELU"), Some(Activation::Gelu));
        assert_eq!(Activation::from_name("invalid"), None);
    }

    #[test]
    fn test_activation_id_roundtrip() {
        let activations = [
            Activation::None,
            Activation::Relu,
            Activation::Sigmoid,
            Activation::Tanh,
            Activation::Softmax,
            Activation::Sqrt,
            Activation::Log,
            Activation::Log10,
            Activation::Inverse,
            Activation::Gelu,
        ];
        for act in activations {
            assert_eq!(Activation::from_id(act.to_id()), act);
        }
    }

    #[test]
    fn test_sqrt_activation() {
        use burn::tensor::backend::Backend;
        let device = <TestBackend as Backend>::Device::default();
        let input = Tensor::<TestBackend, 1>::from_floats([4.0, 9.0, -1.0, 0.0, 16.0], &device);
        let output = Activation::Sqrt.apply(input);
        let result: Vec<f32> = output.to_data().to_vec().unwrap();
        // sqrt(4)=2, sqrt(9)=3, sqrt(-1)=0, sqrt(0)=0, sqrt(16)=4
        assert!((result[0] - 2.0).abs() < 1e-5);
        assert!((result[1] - 3.0).abs() < 1e-5);
        assert!((result[2] - 0.0).abs() < 1e-5); // negative -> 0
        assert!((result[3] - 0.0).abs() < 1e-5); // zero -> 0
        assert!((result[4] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_activation() {
        use burn::tensor::backend::Backend;
        let device = <TestBackend as Backend>::Device::default();
        // ln(x+1) for x > 0, else 0
        let input = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, -1.0, 0.0], &device);
        let output = Activation::Log.apply(input);
        let result: Vec<f32> = output.to_data().to_vec().unwrap();
        // ln(1+1)=ln(2), ln(2+1)=ln(3), ln(-1+1)=0 (masked), ln(0+1)=0 (masked)
        assert!((result[0] - 2.0_f32.ln()).abs() < 1e-5);
        assert!((result[1] - 3.0_f32.ln()).abs() < 1e-5);
        assert!((result[2] - 0.0).abs() < 1e-5); // negative -> 0
        assert!((result[3] - 0.0).abs() < 1e-5); // zero -> 0
    }

    #[test]
    fn test_log10_activation() {
        use burn::tensor::backend::Backend;
        let device = <TestBackend as Backend>::Device::default();
        // log10(x+1) for x > 0, else 0
        let input = Tensor::<TestBackend, 1>::from_floats([9.0, 99.0, -1.0, 0.0], &device);
        let output = Activation::Log10.apply(input);
        let result: Vec<f32> = output.to_data().to_vec().unwrap();
        // log10(9+1)=1, log10(99+1)=2, masked, masked
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 2.0).abs() < 1e-5);
        assert!((result[2] - 0.0).abs() < 1e-5); // negative -> 0
        assert!((result[3] - 0.0).abs() < 1e-5); // zero -> 0
    }

    #[test]
    fn test_inverse_activation() {
        use burn::tensor::backend::Backend;
        let device = <TestBackend as Backend>::Device::default();
        // 1 - x
        let input = Tensor::<TestBackend, 1>::from_floats([0.0, 0.5, 1.0, -0.5, 2.0], &device);
        let output = Activation::Inverse.apply(input);
        let result: Vec<f32> = output.to_data().to_vec().unwrap();
        // 1-0=1, 1-0.5=0.5, 1-1=0, 1-(-0.5)=1.5, 1-2=-1
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 0.5).abs() < 1e-5);
        assert!((result[2] - 0.0).abs() < 1e-5);
        assert!((result[3] - 1.5).abs() < 1e-5);
        assert!((result[4] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_gelu_activation() {
        use burn::tensor::backend::Backend;
        let device = <TestBackend as Backend>::Device::default();
        // GeLU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        // Expected values computed from TensorFlow/NumPy
        let input = Tensor::<TestBackend, 1>::from_floats([-2.0, -1.0, 0.0, 1.0, 2.0], &device);
        let output = Activation::Gelu.apply(input);
        let result: Vec<f32> = output.to_data().to_vec().unwrap();
        // GeLU(-2) ≈ -0.0454, GeLU(-1) ≈ -0.1587, GeLU(0) = 0, GeLU(1) ≈ 0.8413, GeLU(2) ≈ 1.9545
        assert!((result[0] - (-0.0454)).abs() < 1e-3);
        assert!((result[1] - (-0.1587)).abs() < 1e-3);
        assert!((result[2] - 0.0).abs() < 1e-5);
        assert!((result[3] - 0.8413).abs() < 1e-3);
        assert!((result[4] - 1.9545).abs() < 1e-3);
    }
}
