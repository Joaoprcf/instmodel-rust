use burn::tensor::{Tensor, backend::Backend};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    Mse,
    BinaryCrossEntropy,
}

impl Loss {
    pub fn compute<B: Backend>(
        &self,
        predictions: Tensor<B, 2>,
        targets: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        match self {
            Loss::Mse => {
                let diff = predictions - targets;
                (diff.clone() * diff).mean()
            }
            Loss::BinaryCrossEntropy => {
                // Clamp keeps log() finite for saturated predictions.
                let epsilon = 1e-7;
                let ones = Tensor::ones_like(&predictions);
                let p = predictions.clamp(epsilon, 1.0 - epsilon);
                let bce =
                    targets.clone() * p.clone().log() + (ones.clone() - targets) * (ones - p).log();
                bce.neg().mean()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    fn loss(kind: Loss, pred: [[f32; 1]; 2], target: [[f32; 1]; 2]) -> f32 {
        let device = Default::default();
        kind.compute(
            Tensor::<NdArray, 2>::from_floats(pred, &device),
            Tensor::from_floats(target, &device),
        )
        .into_scalar()
    }

    #[test]
    fn loss_values() {
        assert!(loss(Loss::Mse, [[1.0], [2.0]], [[1.0], [2.0]]).abs() < 1e-6);
        assert!((loss(Loss::Mse, [[1.0], [2.0]], [[2.0], [2.0]]) - 0.5).abs() < 1e-6);
        let bce = loss(Loss::BinaryCrossEntropy, [[0.99], [0.01]], [[1.0], [0.0]]);
        assert!((bce - -(0.99f32.ln())).abs() < 1e-4);
        assert!(loss(Loss::BinaryCrossEntropy, [[1.0], [0.0]], [[0.0], [1.0]]).is_finite());
    }
}
