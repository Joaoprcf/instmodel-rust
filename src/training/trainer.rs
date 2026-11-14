use super::TrainingConfig;
use crate::model_graph::ModelGraph;
use burn::{
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{ElementConversion, Tensor, backend::AutodiffBackend},
};

#[derive(Debug)]
pub struct TrainingResult<B: AutodiffBackend> {
    pub model: ModelGraph<B>,
    /// One entry per epoch.
    pub loss_history: Vec<f32>,
}

/// Full-batch Adam training: every epoch is one step over all `inputs`.
pub fn train<B: AutodiffBackend>(
    model: ModelGraph<B>,
    inputs: &[Vec<f32>],
    targets: &[Vec<f32>],
    config: &TrainingConfig,
    device: &B::Device,
) -> TrainingResult<B> {
    let to_tensor = |rows: &[Vec<f32>]| {
        let width = rows.first().map_or(0, Vec::len);
        let flat: Vec<f32> = rows.concat();
        Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([rows.len(), width])
    };
    let (x, y) = (to_tensor(inputs), to_tensor(targets));

    let mut optimizer = AdamConfig::new().init();
    let mut model = model;
    let mut loss_history = Vec::with_capacity(config.epochs);
    for epoch in 0..config.epochs {
        let loss = config.loss.compute(model.forward(x.clone()), y.clone());
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
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(config.learning_rate, model, grads);
    }
    TrainingResult {
        model,
        loss_history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Activation;
    use crate::model_graph::ModelGraphConfig;
    use crate::training::Loss;
    use burn::backend::{Autodiff, NdArray};

    #[test]
    fn training_reduces_loss() {
        let device = Default::default();
        let xor = (
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]],
        );
        let line = (
            vec![vec![0.0], vec![0.5], vec![1.0], vec![1.5], vec![2.0]],
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
        );
        let cases = [
            (xor, Activation::Sigmoid, Loss::BinaryCrossEntropy, 50, 0.1),
            (line, Activation::None, Loss::Mse, 100, 0.01),
        ];
        for ((inputs, targets), out_act, loss, epochs, lr) in cases {
            let model: ModelGraph<Autodiff<NdArray>> =
                ModelGraphConfig::with_feature_size(inputs[0].len())
                    .dense(4, Activation::Relu)
                    .dense(1, out_act)
                    .build(&device)
                    .unwrap();
            let config = TrainingConfig::new()
                .epochs(epochs)
                .learning_rate(lr)
                .loss(loss)
                .verbose(false);
            let h = train(model, &inputs, &targets, &config, &device).loss_history;
            assert_eq!(h.len(), epochs);
            assert!(h[epochs - 1] < h[0], "{loss:?}: {h:?}");
        }
    }
}
