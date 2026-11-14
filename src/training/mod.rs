//! Full-batch Adam training for the sequential [`ModelGraph`](crate::ModelGraph).

mod config;
mod loss;
mod trainer;

pub use config::TrainingConfig;
pub use loss::Loss;
pub use trainer::{TrainingResult, train};
