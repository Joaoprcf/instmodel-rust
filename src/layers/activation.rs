use burn::tensor::{Tensor, backend::Backend};
use serde::{Deserialize, Serialize};

/// Activation functions shared with the instmodel inference engines.
/// Sqrt/Log/Log10 return 0 for x <= 0; Log is ln(x + 1), Log10 is log10(x + 1); Inverse is 1 - x.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "UPPERCASE")]
pub enum Activation {
    #[default]
    None,
    Relu,
    Sigmoid,
    Tanh,
    /// Across the last dimension.
    Softmax,
    Sqrt,
    Log,
    Log10,
    Inverse,
    /// Exact (erf-based) GeLU.
    Gelu,
}

/// Indexed by [`Activation::to_id`].
const NAMES: [(Activation, &str); 10] = [
    (Activation::None, "NONE"),
    (Activation::Relu, "RELU"),
    (Activation::Sigmoid, "SIGMOID"),
    (Activation::Tanh, "TANH"),
    (Activation::Softmax, "SOFTMAX"),
    (Activation::Sqrt, "SQRT"),
    (Activation::Log, "LOG"),
    (Activation::Log10, "LOG10"),
    (Activation::Inverse, "INVERSE"),
    (Activation::Gelu, "GELU"),
];

fn positive_only<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    f: impl FnOnce(Tensor<B, D>) -> Tensor<B, D>,
) -> Tensor<B, D> {
    let mask = x.clone().greater_elem(0.0);
    x.zeros_like().mask_where(mask, f(x))
}

impl Activation {
    pub fn apply<B: Backend, const D: usize>(&self, tensor: Tensor<B, D>) -> Tensor<B, D> {
        use burn::tensor::activation as act;
        match self {
            Activation::None => tensor,
            Activation::Relu => act::relu(tensor),
            Activation::Sigmoid => act::sigmoid(tensor),
            Activation::Tanh => act::tanh(tensor),
            Activation::Softmax => act::softmax(tensor, D - 1),
            Activation::Sqrt => positive_only(tensor, Tensor::sqrt),
            Activation::Log => positive_only(tensor, |x| (x + 1.0).log()),
            Activation::Log10 => {
                positive_only(tensor, |x| (x + 1.0).log() / (10.0_f64.ln() as f32))
            }
            Activation::Inverse => tensor.neg() + 1.0,
            Activation::Gelu => act::gelu(tensor),
        }
    }

    /// JSON instruction name; `None` for the identity.
    pub fn to_instruction_name(&self) -> Option<&'static str> {
        (*self != Activation::None).then(|| NAMES[self.to_id() as usize].1)
    }

    /// Case-insensitive inverse of the instruction name ("NONE" included).
    pub fn from_name(name: &str) -> Option<Self> {
        let name = name.to_uppercase();
        NAMES.iter().find(|(_, n)| *n == name).map(|(a, _)| *a)
    }

    /// Stable numeric id used to store the activation inside Burn modules.
    pub fn to_id(&self) -> u8 {
        *self as u8
    }

    /// Unknown ids map to `None`.
    pub fn from_id(id: u8) -> Self {
        NAMES.get(id as usize).map_or(Activation::None, |(a, _)| *a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn names_and_ids_roundtrip() {
        for (i, (act, name)) in NAMES.iter().enumerate() {
            assert_eq!(act.to_id() as usize, i);
            assert_eq!(Activation::from_id(i as u8), *act);
            assert_eq!(Activation::from_name(&name.to_lowercase()), Some(*act));
            let expected = (*act != Activation::None).then_some(*name);
            assert_eq!(act.to_instruction_name(), expected);
        }
        assert_eq!(Activation::from_name("invalid"), None);
        assert_eq!(Activation::from_id(200), Activation::None);
    }

    #[test]
    fn apply_values() {
        let cases: [(Activation, [f32; 5], [f32; 5]); 6] = [
            (
                Activation::Sqrt,
                [4.0, 9.0, -1.0, 0.0, 16.0],
                [2.0, 3.0, 0.0, 0.0, 4.0],
            ),
            (
                Activation::Log,
                [1.0, 2.0, -1.0, 0.0, 0.0],
                [2f32.ln(), 3f32.ln(), 0.0, 0.0, 0.0],
            ),
            (
                Activation::Log10,
                [9.0, 99.0, -1.0, 0.0, 0.0],
                [1.0, 2.0, 0.0, 0.0, 0.0],
            ),
            (
                Activation::Inverse,
                [0.0, 0.5, 1.0, -0.5, 2.0],
                [1.0, 0.5, 0.0, 1.5, -1.0],
            ),
            (
                Activation::Relu,
                [-1.0, 0.0, 1.0, 2.0, -3.0],
                [0.0, 0.0, 1.0, 2.0, 0.0],
            ),
            (
                Activation::Gelu,
                [-2.0, -1.0, 0.0, 1.0, 2.0],
                [-0.0455, -0.1587, 0.0, 0.8413, 1.9545],
            ),
        ];
        for (act, input, expected) in cases {
            let x = Tensor::<NdArray, 1>::from_floats(input, &Default::default());
            let out: Vec<f32> = act.apply(x).to_data().to_vec().unwrap();
            let tol = if act == Activation::Gelu { 1e-3 } else { 1e-5 };
            for (o, e) in out.iter().zip(expected) {
                assert!((o - e).abs() < tol, "{act:?}: {out:?} vs {expected:?}");
            }
        }
    }
}
