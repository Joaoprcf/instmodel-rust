# instmodel

This repository is meant to be used through coding agents, not read manually.

Define, train and export neural networks in Rust with [Burn](https://burn.dev/): a functional
graph API (branches, shared layers, nested sub-models, BatchNorm, scale/shift, standalone
activations) that exports to the compact instmodel JSON instruction format, runnable with
[`instmodel_inference`](https://github.com/Joaoprcf/instmodel-rust-inference) and the other
instmodel engines. Trained sub-models can be extracted with their weights.

```bash
cargo add instmodel
cargo add instmodel_inference  # to run exported models
```

```rust
use burn::backend::NdArray;
use instmodel::graph::{InputBuffer, ModelGraph, ops};
use instmodel::layers::Activation;

let input = InputBuffer::new(4);
let hidden = ops::dense(8, Activation::Relu, input.buffer());
let output = ops::dense(1, Activation::Sigmoid, hidden);
let model = ModelGraph::new(vec![input], output)
    .compile::<NdArray>(&Default::default())
    .unwrap();
let json = model.export_to_instruction_model().unwrap();
```

MIT licensed.
