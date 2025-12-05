# instmodel

A Rust library for building and training neural networks using the [Burn](https://burn.dev/) framework, with export to a lightweight instruction-based format for fast inference.

## Installation

```bash
cargo add instmodel
```

For inference, also add the companion crate:

```bash
cargo add instmodel_inference
```

## Features

- **Graph-based model construction** - Build neural networks using a functional API
- **Weight sharing** - Reuse layers across different parts of the network
- **Training support** - Train models with MSE or BCE loss
- **Export to instruction format** - Export trained models to JSON for use with [instmodel-rust-inference](https://github.com/Joaoprcf/instmodel-rust-inference)

### Supported Operations

| Operation | Description |
|-----------|-------------|
| Dense | Fully connected layer with optional activation |
| Add | Element-wise addition of buffers |
| Multiply | Element-wise multiplication of buffers |
| Concat | Concatenate buffers along feature dimension |
| Softmax | Softmax normalization |
| Scale | Element-wise multiply by fixed vector |
| Shift | Element-wise add fixed vector |

### Supported Activations

`None`, `Relu`, `Sigmoid`, `Tanh`, `Softmax`, `Sqrt`, `Log`, `Log10`, `Inverse`

## Usage

### Basic Model (Sequential)

```rust
use instmodel::{ModelGraph, ModelGraphConfig};
use instmodel::layers::Activation;
use burn::backend::NdArray;

type Backend = NdArray;

fn main() {
    let device = Default::default();

    // Build a simple feedforward network
    let model: ModelGraph<Backend> = ModelGraphConfig::with_feature_size(4)
        .dense(8, Activation::Relu)
        .dense(4, Activation::Relu)
        .dense(1, Activation::Sigmoid)
        .build(&device)
        .expect("Failed to build model");

    // Export to JSON for inference
    let json = model.export_to_instruction_model().unwrap();
    println!("{}", json);
}
```

### Graph API with Shared Layers

The Graph API allows building complex architectures with branching, merging, and weight sharing.

```rust
use instmodel::graph::{InputBuffer, ModelGraph, ops};
use instmodel::layers::Activation;
use burn::backend::NdArray;
use burn::tensor::{Tensor, backend::Backend};

type TestBackend = NdArray;

fn main() {
    let device = <TestBackend as Backend>::Device::default();

    // Create input buffer
    let input = InputBuffer::new(4);
    let x = input.buffer();

    // Create a shared dense layer (will be reused)
    let shared_layer = instmodel::graph::Operation::dense(4, Activation::Relu);

    // Apply the same layer to two different branches
    let branch1 = shared_layer.apply(x.clone());
    let branch2 = shared_layer.apply(x.clone());  // Same weights as branch1!

    // Merge branches with element-wise addition
    let merged = ops::add(vec![branch1, branch2]);

    // Final output layer (not shared)
    let output = ops::dense(1, Activation::Sigmoid, merged);

    // Build the model
    let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
        .expect("Model creation should succeed");

    // Run inference
    let input_tensor = Tensor::<TestBackend, 2>::from_floats(
        [[1.0, 2.0, 3.0, 4.0]],
        &device
    );
    let result = model.forward(input_tensor);
    println!("Output: {:?}", result.to_data());

    // Export for use with instmodel_inference
    let json = model.export_to_instruction_model().unwrap();
    println!("{}", json);
}
```

### Non-Shared Layers (Default)

When using `ops::dense()`, each call creates a new layer with independent weights:

```rust
use instmodel::graph::{InputBuffer, ModelGraph, ops};
use instmodel::layers::Activation;
use burn::backend::NdArray;
use burn::tensor::backend::Backend;

type TestBackend = NdArray;

fn main() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();

    // These are separate layers with different weights
    let branch1 = ops::dense(4, Activation::Relu, x.clone());
    let branch2 = ops::dense(4, Activation::Relu, x.clone());  // Different weights!

    // Merge and output
    let merged = ops::add(vec![branch1, branch2]);
    let output = ops::dense(1, Activation::Sigmoid, merged);

    let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
        .expect("Model creation should succeed");

    let json = model.export_to_instruction_model().unwrap();
    println!("{}", json);
}
```

### Using Scale and Shift (Feature Normalization)

```rust
use instmodel::graph::{InputBuffer, ModelGraph, ops};
use instmodel::layers::Activation;
use burn::backend::NdArray;
use burn::tensor::backend::Backend;

type TestBackend = NdArray;

fn main() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();

    // Normalize input: (x * scale) + shift
    let scale_vector = vec![0.5, 0.5, 0.5, 0.5];
    let shift_vector = vec![-0.5, -0.5, -0.5, -0.5];

    let scaled = ops::scale(scale_vector, x);
    let normalized = ops::shift(shift_vector, scaled);

    // Continue with dense layers
    let hidden = ops::dense(8, Activation::Relu, normalized);
    let output = ops::dense(1, Activation::Sigmoid, hidden);

    let model = ModelGraph::<TestBackend>::new(vec![input], output, &device)
        .expect("Model creation should succeed");

    let json = model.export_to_instruction_model().unwrap();
    println!("{}", json);
}
```

### Inference with instmodel_inference

After exporting a model, use [instmodel_inference](https://github.com/Joaoprcf/instmodel-rust-inference) for fast, dependency-light inference:

```rust
use instmodel_inference::{InstructionModel, InstructionModelInfo};

fn main() {
    // Load the exported JSON
    let json = r#"{
        "buffer_sizes": [4, 1],
        "instructions": [{"type": "DOT", "input": 0, "output": 1, "weights": 0, "activation": "SIGMOID"}],
        "weights": [[[0.1, 0.2, 0.3, 0.4]]],
        "bias": [[0.0]],
        "parameters": []
    }"#;

    let model_info: InstructionModelInfo = serde_json::from_str(json).unwrap();
    let model = InstructionModel::new(model_info).unwrap();

    // Run inference
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = model.predict(&input).unwrap();

    println!("Prediction: {:?}", output);
}
```

### Training Example

```rust
use instmodel::{ModelGraph, ModelGraphConfig};
use instmodel::layers::Activation;
use instmodel::training::{TrainingConfig, Loss, train};
use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type TrainingBackend = Autodiff<NdArray>;

fn main() {
    let device = Default::default();

    // Build model
    let model: ModelGraph<TrainingBackend> = ModelGraphConfig::with_feature_size(2)
        .dense(4, Activation::Relu)
        .dense(1, Activation::Sigmoid)
        .build(&device)
        .expect("Failed to build model");

    // Training data (XOR problem)
    let x_train = Tensor::<TrainingBackend, 2>::from_floats(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        &device,
    );
    let y_train = Tensor::<TrainingBackend, 2>::from_floats(
        [[0.0], [1.0], [1.0], [0.0]],
        &device,
    );

    // Configure and run training
    let config = TrainingConfig::builder()
        .epochs(1000)
        .learning_rate(0.1)
        .loss(Loss::Bce)
        .build();

    let trained_model = train(model, x_train, y_train, config, &device);

    // Export trained model
    let json = trained_model.export_to_instruction_model().unwrap();
    println!("{}", json);
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [instmodel-rust-inference](https://github.com/Joaoprcf/instmodel-rust-inference) - Lightweight inference library for exported models
- [Burn](https://burn.dev/) - The deep learning framework powering this library