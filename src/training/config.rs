use super::Loss;

/// Defaults: 100 epochs, lr 0.001, batch 32, MSE, verbose.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    /// Not used by [`train`](super::train), which is full-batch.
    pub batch_size: usize,
    pub loss: Loss,
    /// Logs progress via `log::info!` every 10 epochs.
    pub verbose: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            learning_rate: 0.001,
            batch_size: 32,
            loss: Loss::Mse,
            verbose: true,
        }
    }
}

impl TrainingConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    pub fn loss(mut self, loss: Loss) -> Self {
        self.loss = loss;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}
