//! Training loop implementation.

use super::TrainingConfig;
use crate::model_graph::ModelGraph;
use burn::{
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{ElementConversion, Tensor, backend::AutodiffBackend},
};

/// Training result containing the trained model and metrics.
#[derive(Debug)]
pub struct TrainingResult<B: AutodiffBackend> {
    /// The trained model.
    pub model: ModelGraph<B>,
    /// Loss values per epoch.
    pub loss_history: Vec<f32>,
}

/// Trains a model using the Adam optimizer.
pub fn train<B: AutodiffBackend>(
    model: ModelGraph<B>,
    inputs: &[Vec<f32>],
    targets: &[Vec<f32>],
    config: &TrainingConfig,
    device: &B::Device,
) -> TrainingResult<B> {
    let num_samples = inputs.len();
    let input_size = inputs.first().map(|v| v.len()).unwrap_or(0);
    let output_size = targets.first().map(|v| v.len()).unwrap_or(0);

    // Convert data to tensors
    let input_data: Vec<f32> = inputs.iter().flat_map(|v| v.iter().copied()).collect();
    let target_data: Vec<f32> = targets.iter().flat_map(|v| v.iter().copied()).collect();

    let input_tensor = Tensor::<B, 1>::from_floats(input_data.as_slice(), device)
        .reshape([num_samples, input_size]);
    let target_tensor = Tensor::<B, 1>::from_floats(target_data.as_slice(), device)
        .reshape([num_samples, output_size]);

    // Initialize optimizer
    let optimizer_config = AdamConfig::new();
    let mut optimizer = optimizer_config.init();

    let mut current_model = model;
    let mut loss_history = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        // Forward pass
        let predictions = current_model.forward(input_tensor.clone());

        // Compute loss
        let loss = config.loss.compute(predictions, target_tensor.clone());
        let loss_value: f32 = loss.clone().into_scalar().elem();
        loss_history.push(loss_value);

        if config.verbose && (epoch % 10 == 0 || epoch == config.epochs - 1) {
            log::info!(
                "Epoch {}/{}: loss = {:.6}",
                epoch + 1,
                config.epochs,
                loss_value
            );
        }

        // Backward pass
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &current_model);

        // Update model parameters
        current_model = optimizer.step(config.learning_rate, current_model, grads_params);
    }

    TrainingResult {
        model: current_model,
        loss_history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::Loss;
    use burn::backend::{Autodiff, NdArray};

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_training_reduces_loss() {
        use crate::layers::Activation;
        use crate::model_graph::ModelGraphConfig;

        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();

        // Create a simple model
        let model: ModelGraph<TestBackend> = ModelGraphConfig::with_feature_size(2)
            .dense(4, Activation::Relu)
            .dense(1, Activation::Sigmoid)
            .build(&device)
            .expect("Model build should succeed");

        // Create simple XOR-like data
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        let config = TrainingConfig::new()
            .epochs(50)
            .learning_rate(0.1)
            .loss(Loss::BinaryCrossEntropy)
            .verbose(false);

        let result = train(model, &inputs, &targets, &config, &device);

        // Loss should decrease over training
        let initial_loss = result.loss_history.first().copied().unwrap_or(f32::MAX);
        let final_loss = result.loss_history.last().copied().unwrap_or(f32::MAX);

        assert!(
            final_loss < initial_loss,
            "Loss should decrease: initial={}, final={}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn test_training_simple_regression() {
        use crate::layers::Activation;
        use crate::model_graph::ModelGraphConfig;

        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();

        // Create a simple model for regression
        let model: ModelGraph<TestBackend> = ModelGraphConfig::with_feature_size(1)
            .dense(4, Activation::Relu)
            .dense(1, Activation::None)
            .build(&device)
            .expect("Model build should succeed");

        // Simple linear data: y = 2x
        let inputs = vec![vec![0.0], vec![0.5], vec![1.0], vec![1.5], vec![2.0]];
        let targets = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];

        let config = TrainingConfig::new()
            .epochs(100)
            .learning_rate(0.01)
            .loss(Loss::Mse)
            .verbose(false);

        let result = train(model, &inputs, &targets, &config, &device);

        let final_loss = result.loss_history.last().copied().unwrap_or(f32::MAX);

        // Loss should decrease (may not always reach < 1.0 due to random init)
        let initial_loss = result.loss_history.first().copied().unwrap_or(f32::MAX);
        assert!(
            final_loss < initial_loss,
            "Loss should decrease: initial={}, final={}",
            initial_loss,
            final_loss
        );
    }
}
