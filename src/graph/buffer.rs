use std::sync::atomic::{AtomicUsize, Ordering};

use super::operation::{OpId, Operation};

static BUFFER_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Unique identifier for a DataBuffer.
pub type BufferId = usize;

/// What produced a DataBuffer.
#[derive(Clone, Debug)]
pub enum Producer {
    Op(Operation),
    /// A ModelGraph applied to new inputs; inlined at compile time by remapping `input_ids`
    /// (the sub-graph's own InputBuffer ids, in order) to the actual inputs.
    SubGraph {
        id: OpId,
        input_ids: Vec<BufferId>,
        output: Box<DataBuffer>,
        graph_id: usize,
    },
}

/// A node in the computation graph: its size, producer and producer inputs.
#[derive(Clone, Debug)]
pub struct DataBuffer {
    id: BufferId,
    size: usize,
    producer: Option<Producer>,
    inputs: Vec<DataBuffer>,
}

impl DataBuffer {
    pub(crate) fn new(size: usize, producer: Option<Operation>, inputs: Vec<DataBuffer>) -> Self {
        Self::new_with_producer(size, producer.map(Producer::Op), inputs)
    }

    pub(crate) fn new_with_producer(
        size: usize,
        producer: Option<Producer>,
        inputs: Vec<DataBuffer>,
    ) -> Self {
        Self {
            id: BUFFER_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            size,
            producer,
            inputs,
        }
    }

    pub fn id(&self) -> BufferId {
        self.id
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// True for graph inputs (no producer).
    pub fn is_input(&self) -> bool {
        self.producer.is_none()
    }

    pub fn producer(&self) -> Option<&Producer> {
        self.producer.as_ref()
    }

    pub fn inputs(&self) -> &[DataBuffer] {
        &self.inputs
    }
}

/// Graph input, optionally with feature names.
#[derive(Clone, Debug)]
pub struct InputBuffer {
    buffer: DataBuffer,
    features: Option<Vec<String>>,
}

impl InputBuffer {
    pub fn new(size: usize) -> Self {
        Self {
            buffer: DataBuffer::new(size, None, vec![]),
            features: None,
        }
    }

    pub fn with_features(features: Vec<String>) -> Self {
        Self {
            buffer: DataBuffer::new(features.len(), None, vec![]),
            features: Some(features),
        }
    }

    pub fn buffer(&self) -> DataBuffer {
        self.buffer.clone()
    }

    pub fn size(&self) -> usize {
        self.buffer.size
    }

    pub fn features(&self) -> Option<&[String]> {
        self.features.as_deref()
    }

    pub fn id(&self) -> BufferId {
        self.buffer.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_buffers() {
        let a = InputBuffer::new(4);
        let b = InputBuffer::with_features(vec!["x".into(), "y".into(), "z".into()]);
        assert_eq!((a.size(), b.size()), (4, 3));
        assert!(a.features().is_none() && a.buffer().is_input());
        assert_eq!(b.features().unwrap()[2], "z");
        assert_ne!(a.id(), b.id());
        assert_eq!(a.buffer().id(), a.id());
    }
}
