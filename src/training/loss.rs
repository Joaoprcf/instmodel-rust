//! Loss functions for training.

use burn::tensor::{Tensor, backend::Backend};

/// Supported loss functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    /// Mean Squared Error loss.
    Mse,
    /// Binary Cross Entropy loss.
    BinaryCrossEntropy,
}

impl Loss {
    /// Computes the loss between predictions and targets.
    pub fn compute<B: Backend>(
        &self,
        predictions: Tensor<B, 2>,
        targets: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        match self {
            Loss::Mse => {
                let diff = predictions - targets;
                let squared = diff.clone() * diff;
                squared.mean()
            }
            Loss::BinaryCrossEntropy => {
                // BCE = -mean(y * log(p) + (1-y) * log(1-p))
                // Add small epsilon for numerical stability
                let epsilon = 1e-7;
                let ones = Tensor::ones_like(&predictions);
                let p_clipped = predictions.clone().clamp(epsilon, 1.0 - epsilon);
                let log_p = p_clipped.clone().log();
                let log_1_minus_p = (ones.clone() - p_clipped).log();
                let bce = targets.clone() * log_p + (ones - targets) * log_1_minus_p;
                bce.neg().mean()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_mse_loss_zero() {
        let device = <TestBackend as Backend>::Device::default();
        let predictions = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        let targets = predictions.clone();

        let loss = Loss::Mse.compute(predictions, targets);
        let loss_value: f32 = loss.into_scalar();

        assert!(
            loss_value.abs() < 1e-6,
            "MSE of identical tensors should be 0"
        );
    }

    #[test]
    fn test_mse_loss_nonzero() {
        let device = <TestBackend as Backend>::Device::default();
        let predictions = Tensor::<TestBackend, 2>::from_floats([[1.0], [2.0]], &device);
        let targets = Tensor::<TestBackend, 2>::from_floats([[2.0], [2.0]], &device);

        let loss = Loss::Mse.compute(predictions, targets);
        let loss_value: f32 = loss.into_scalar();

        // MSE = mean((1-2)^2 + (2-2)^2) = mean(1 + 0) = 0.5
        assert!((loss_value - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bce_loss_perfect_prediction() {
        let device = <TestBackend as Backend>::Device::default();
        let predictions = Tensor::<TestBackend, 2>::from_floats([[0.99], [0.01]], &device);
        let targets = Tensor::<TestBackend, 2>::from_floats([[1.0], [0.0]], &device);

        let loss = Loss::BinaryCrossEntropy.compute(predictions, targets);
        let loss_value: f32 = loss.into_scalar();

        // Loss should be very small for near-perfect predictions
        assert!(loss_value < 0.1);
    }
}
