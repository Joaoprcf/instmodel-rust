//! DataBuffer and InputBuffer - core graph node types.
//!
//! These structures represent data flowing through the computation graph,
//! similar to Python instmodel's DataBuffer and InputBuffer classes.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::operation::Operation;

/// Global counter for unique buffer IDs.
static BUFFER_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Unique identifier for a DataBuffer.
pub type BufferId = usize;

/// DataBuffer represents a node in the computation graph.
///
/// It tracks:
/// - The size of the data (output dimension)
/// - The operation that produced it (if any)
/// - The input buffers used by that operation
#[derive(Clone, Debug)]
pub struct DataBuffer {
    id: BufferId,
    size: usize,
    producer: Option<Operation>,
    inputs: Vec<DataBuffer>,
}

impl DataBuffer {
    /// Creates a new DataBuffer with the given size and producer.
    pub(crate) fn new(size: usize, producer: Option<Operation>, inputs: Vec<DataBuffer>) -> Self {
        Self {
            id: BUFFER_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            size,
            producer,
            inputs,
        }
    }

    /// Returns the unique ID of this buffer.
    pub fn id(&self) -> BufferId {
        self.id
    }

    /// Returns the output size of this buffer.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns true if this is an input buffer (no producer).
    pub fn is_input(&self) -> bool {
        self.producer.is_none()
    }

    /// Returns the producer operation, if any.
    pub fn producer(&self) -> Option<&Operation> {
        self.producer.as_ref()
    }

    /// Returns the input buffers.
    pub fn inputs(&self) -> &[DataBuffer] {
        &self.inputs
    }
}

/// InputBuffer represents an input to the computation graph.
///
/// This is the entry point for data into the model.
#[derive(Clone, Debug)]
pub struct InputBuffer {
    buffer: DataBuffer,
    features: Option<Vec<String>>,
}

impl InputBuffer {
    /// Creates a new InputBuffer with the given size.
    pub fn new(size: usize) -> Self {
        Self {
            buffer: DataBuffer::new(size, None, vec![]),
            features: None,
        }
    }

    /// Creates a new InputBuffer with named features.
    pub fn with_features(features: Vec<String>) -> Self {
        let size = features.len();
        Self {
            buffer: DataBuffer::new(size, None, vec![]),
            features: Some(features),
        }
    }

    /// Returns the DataBuffer for use in graph building.
    pub fn buffer(&self) -> DataBuffer {
        self.buffer.clone()
    }

    /// Returns the size of this input.
    pub fn size(&self) -> usize {
        self.buffer.size
    }

    /// Returns the feature names, if set.
    pub fn features(&self) -> Option<&[String]> {
        self.features.as_deref()
    }

    /// Returns the buffer ID.
    pub fn id(&self) -> BufferId {
        self.buffer.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_buffer_creation() {
        let input = InputBuffer::new(4);
        assert_eq!(input.size(), 4);
        assert!(input.features().is_none());
        assert!(input.buffer().is_input());
    }

    #[test]
    fn test_input_buffer_with_features() {
        let input =
            InputBuffer::with_features(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        assert_eq!(input.size(), 3);
        assert_eq!(input.features().unwrap().len(), 3);
    }

    #[test]
    fn test_buffer_ids_are_unique() {
        let input1 = InputBuffer::new(4);
        let input2 = InputBuffer::new(4);
        assert_ne!(input1.id(), input2.id());
    }

    #[test]
    fn test_data_buffer_clone() {
        let input = InputBuffer::new(4);
        let buf1 = input.buffer();
        let buf2 = input.buffer();
        assert_eq!(buf1.id(), buf2.id());
    }
}
