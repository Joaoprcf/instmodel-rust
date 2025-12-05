//! Training utilities for neural network models.
//!
//! This module provides training functionality including:
//! - Loss functions (MSE, Binary Cross Entropy)
//! - Training configuration
//! - Training loop with Adam optimizer

mod config;
mod loss;
mod trainer;

pub use config::TrainingConfig;
pub use loss::Loss;
pub use trainer::train;
