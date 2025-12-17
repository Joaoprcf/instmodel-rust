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
- **Deferred compilation** - Build graph structure first, compile with weights later
- **Nested model composition** - Compose sub-models and extract them after training
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
| BatchNorm | Batch normalization |
| Activation | Standalone activation function |

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

### Graph API (Deferred Compilation)

The Graph API separates graph construction from weight initialization. Build your model structure first, then compile it to create weights on a specific device.

```rust
use instmodel::graph::{InputBuffer, ModelGraph, ops};
use instmodel::layers::Activation;
use burn::backend::NdArray;
use burn::tensor::{Tensor, backend::Backend};

type TestBackend = NdArray;

fn main() {
    let device = <TestBackend as Backend>::Device::default();

    // Phase 1: Build graph structure (no device, no weights)
    let input = InputBuffer::new(4);
    let x = input.buffer();
    let hidden = ops::dense(8, Activation::Relu, x);
    let output = ops::dense(1, Activation::Sigmoid, hidden);
    let graph = ModelGraph::new(vec![input], output);

    // Phase 2: Compile with device (weights initialized here)
    let model = graph.compile::<TestBackend>(&device)
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

### Nested Models with Sub-Model Extraction

Build complex architectures by composing sub-models. After training the combined model, extract sub-models with the trained weights.

```rust
use instmodel::graph::{InputBuffer, ModelGraph, ops};
use instmodel::layers::Activation;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Tensor, backend::Backend};

type TestBackend = NdArray;
type TrainingBackend = Autodiff<NdArray>;

fn main() {
    let device = <TrainingBackend as Backend>::Device::default();

    // Create sub-graph for "buy" signal
    let buy_input = InputBuffer::new(4);
    let buy_hidden = ops::dense(8, Activation::Relu, buy_input.buffer());
    let buy_out = ops::dense(1, Activation::Sigmoid, buy_hidden);
    let buy_graph = ModelGraph::new(vec![buy_input], buy_out);

    // Create sub-graph for "sell" signal
    let sell_input = InputBuffer::new(4);
    let sell_hidden = ops::dense(8, Activation::Relu, sell_input.buffer());
    let sell_out = ops::dense(1, Activation::Sigmoid, sell_hidden);
    let sell_graph = ModelGraph::new(vec![sell_input], sell_out);

    // Create combined graph using both sub-graphs
    let main_input = InputBuffer::new(4);
    let x = main_input.buffer();
    let buy_result = buy_graph.apply_single(x.clone());
    let sell_result = sell_graph.apply_single(x);
    let combined = ops::concat(vec![buy_result, sell_result]);
    let combined_graph = ModelGraph::new(vec![main_input], combined);

    // Compile the combined model (creates all weights)
    let mut model = combined_graph.compile::<TrainingBackend>(&device)
        .expect("Model creation should succeed");

    // ... train the combined model here ...

    // After training, extract sub-models with trained weights
    let buy_model = model.submodel(&buy_graph);
    let sell_model = model.submodel(&sell_graph);

    // Sub-models produce the same output as corresponding parts of combined model
    let test_input = Tensor::<TrainingBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
    let combined_result = model.forward(test_input.clone());
    let buy_result = buy_model.forward(test_input.clone());
    let sell_result = sell_model.forward(test_input);

    // combined_result[0] == buy_result[0]
    // combined_result[1] == sell_result[0]
}
```

### Branching and Merging

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

    // Two separate branches with different weights
    let branch1 = ops::dense(4, Activation::Relu, x.clone());
    let branch2 = ops::dense(4, Activation::Relu, x.clone());

    // Merge branches with element-wise addition
    let merged = ops::add(vec![branch1, branch2]);

    // Final output layer
    let output = ops::dense(1, Activation::Sigmoid, merged);

    // Build graph structure, then compile
    let graph = ModelGraph::new(vec![input], output);
    let model = graph.compile::<TestBackend>(&device)
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

    let graph = ModelGraph::new(vec![input], output);
    let model = graph.compile::<TestBackend>(&device)
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

## API Overview

### Graph Construction (No Backend Required)

- `InputBuffer::new(size)` - Create an input node
- `ModelGraph::new(inputs, output)` - Create a graph from inputs and output
- `graph.apply(inputs)` / `graph.apply_single(input)` - Use a graph as a sub-model
- `ops::dense(size, activation, input)` - Dense layer
- `ops::add(buffers)` / `ops::multiply(buffers)` - Element-wise operations
- `ops::concat(buffers)` - Concatenation
- `ops::batch_norm(input)` - Batch normalization
- `ops::activation(activation, input)` - Standalone activation

### Compilation (Backend Required)

- `graph.compile::<Backend>(&device)` - Compile graph to `CompiledModel<B>`

### CompiledModel Methods

- `model.forward(input)` - Run inference
- `model.submodel(&graph)` - Extract sub-model with current weights
- `model.export_to_instruction_model()` - Export to JSON
- `model.feature_size()` / `model.output_size()` - Get dimensions

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [instmodel-rust-inference](https://github.com/Joaoprcf/instmodel-rust-inference) - Lightweight inference library for exported models
- [Burn](https://burn.dev/) - The deep learning framework powering this library