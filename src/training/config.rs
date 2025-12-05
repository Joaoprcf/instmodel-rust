//! Training configuration.

use super::Loss;

/// Configuration for model training.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate for the optimizer.
    pub learning_rate: f64,
    /// Batch size for training.
    pub batch_size: usize,
    /// Loss function to use.
    pub loss: Loss,
    /// Whether to print progress during training.
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
    /// Creates a new TrainingConfig with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of epochs.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Sets the learning rate.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Sets the batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Sets the loss function.
    pub fn loss(mut self, loss: Loss) -> Self {
        self.loss = loss;
        self
    }

    /// Sets whether to print progress.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 100);
        assert!((config.learning_rate - 0.001).abs() < 1e-10);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_config_builder() {
        let config = TrainingConfig::new()
            .epochs(50)
            .learning_rate(0.01)
            .batch_size(64)
            .loss(Loss::BinaryCrossEntropy);

        assert_eq!(config.epochs, 50);
        assert!((config.learning_rate - 0.01).abs() < 1e-10);
        assert_eq!(config.batch_size, 64);
        assert!(matches!(config.loss, Loss::BinaryCrossEntropy));
    }
}
