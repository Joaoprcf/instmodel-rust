//! Integration tests verifying that models created with instmodel-rust
//! can be loaded and executed by instmodel-rust-inference.

use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Tensor, backend::Backend};
use instmodel::ModelGraph;
use instmodel::graph::{InputBuffer, ModelGraph as GraphModel, Operation, ops};
use instmodel::layers::Activation;
use instmodel::model_graph::ModelGraphConfig;
use instmodel::training::{Loss, TrainingConfig, train};
use instmodel_inference::{InstructionModel, InstructionModelInfo};

type TestBackend = NdArray;
type TrainingBackend = Autodiff<NdArray>;
type ExpectedInstruction<'a> = (
    &'a str,
    Option<usize>,
    Option<usize>,
    Option<usize>,
    Option<usize>,
);

const TOLERANCE: f32 = 1e-6;

fn floats_close(a: f32, b: f32, tolerance: f32) -> bool {
    (a - b).abs() < tolerance
}

#[test]
fn test_simple_model_inference_equivalence() {
    let device = <TestBackend as Backend>::Device::default();

    let model: ModelGraph<TestBackend> = ModelGraphConfig::with_feature_size(2)
        .dense(1, Activation::None)
        .build(&device)
        .expect("Model build should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");

    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0f32];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 2]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), inference_result.len());
    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "Mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

#[test]
fn test_multilayer_model_with_relu() {
    let device = <TestBackend as Backend>::Device::default();

    let model: ModelGraph<TestBackend> = ModelGraphConfig::with_feature_size(3)
        .dense(4, Activation::Relu)
        .dense(2, Activation::None)
        .build(&device)
        .expect("Model build should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");
    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let test_inputs = vec![
        vec![1.0f32, 0.0, -1.0],
        vec![0.5, 0.5, 0.5],
        vec![-1.0, 1.0, 0.0],
    ];

    for inputs in test_inputs {
        let input_tensor = Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device)
            .reshape([1, inputs.len()]);

        let burn_output = model.forward(input_tensor);
        let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

        let inference_result = inference_model
            .predict(&inputs)
            .expect("Inference should succeed");

        assert_eq!(burn_result.len(), inference_result.len());
        for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
            assert!(
                floats_close(*burn_val, *inference_val, TOLERANCE),
                "Mismatch for input {:?}: burn={}, inference={}",
                inputs,
                burn_val,
                inference_val
            );
        }
    }
}

#[test]
fn test_model_with_sigmoid_output() {
    let device = <TestBackend as Backend>::Device::default();

    let model: ModelGraph<TestBackend> = ModelGraphConfig::with_feature_size(4)
        .dense(8, Activation::Relu)
        .dense(1, Activation::Sigmoid)
        .build(&device)
        .expect("Model build should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");
    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![0.5f32, -0.5, 1.0, 0.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert!(inference_result[0] >= 0.0 && inference_result[0] <= 1.0);

    assert!(
        floats_close(burn_result[0], inference_result[0], TOLERANCE),
        "Mismatch: burn={}, inference={}",
        burn_result[0],
        inference_result[0]
    );
}

#[test]
fn test_deep_network() {
    let device = <TestBackend as Backend>::Device::default();

    let model: ModelGraph<TestBackend> = ModelGraphConfig::with_feature_size(5)
        .dense(10, Activation::Relu)
        .dense(8, Activation::Relu)
        .dense(4, Activation::Relu)
        .dense(1, Activation::Sigmoid)
        .build(&device)
        .expect("Model build should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");
    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, -1.0, 0.5, 0.0, -0.5];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 5]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), 1);
    assert_eq!(inference_result.len(), 1);
    assert!(
        floats_close(burn_result[0], inference_result[0], TOLERANCE),
        "Mismatch: burn={}, inference={}",
        burn_result[0],
        inference_result[0]
    );
}

#[test]
fn test_export_json_format() {
    let device = <TestBackend as Backend>::Device::default();

    let model: ModelGraph<TestBackend> =
        ModelGraphConfig::new(vec!["feature_a".to_string(), "feature_b".to_string()])
            .dense(3, Activation::Relu)
            .dense(1, Activation::Sigmoid)
            .build(&device)
            .expect("Model build should succeed");

    let export = model.to_instruction_model_info();

    assert_eq!(
        export.features,
        Some(vec!["feature_a".to_string(), "feature_b".to_string()])
    );
    assert_eq!(export.feature_size, Some(2));
    assert_eq!(export.buffer_sizes, vec![2, 3, 1]);
    assert_eq!(export.instructions.len(), 2);
    assert_eq!(export.weights.len(), 2);
    assert_eq!(export.bias.len(), 2);

    assert_eq!(export.weights[0].len(), 3);
    assert_eq!(export.weights[0][0].len(), 2);
    assert_eq!(export.weights[1].len(), 1);
    assert_eq!(export.weights[1][0].len(), 3);
}

#[test]
fn test_trained_model_inference_equivalence() {
    let device = <TrainingBackend as Backend>::Device::default();

    let model: ModelGraph<TrainingBackend> = ModelGraphConfig::with_feature_size(2)
        .dense(8, Activation::Relu)
        .dense(1, Activation::Sigmoid)
        .build(&device)
        .expect("Model build should succeed");

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![0.0], vec![0.0], vec![1.0]];

    let config = TrainingConfig::new()
        .epochs(200)
        .learning_rate(0.1)
        .loss(Loss::BinaryCrossEntropy)
        .verbose(false);

    let result = train(model, &inputs, &targets, &config, &device);
    let trained_model = result.model;

    let initial_loss = result.loss_history.first().copied().unwrap_or(f32::MAX);
    let final_loss = result.loss_history.last().copied().unwrap_or(f32::MAX);
    assert!(final_loss < initial_loss, "Training should reduce loss");

    let json = trained_model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    for (input, _target) in inputs.iter().zip(targets.iter()) {
        let input_tensor = Tensor::<TrainingBackend, 1>::from_floats(input.as_slice(), &device)
            .reshape([1, input.len()]);

        let burn_output = trained_model.forward(input_tensor);
        let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

        let inference_result = inference_model
            .predict(input)
            .expect("Inference should succeed");

        assert_eq!(burn_result.len(), inference_result.len());
        for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
            assert!(
                floats_close(*burn_val, *inference_val, TOLERANCE),
                "Mismatch for input {:?}: burn={}, inference={}",
                input,
                burn_val,
                inference_val
            );
        }
    }

    let prediction_11 = inference_model.predict(&[1.0, 1.0]).unwrap()[0];
    let prediction_00 = inference_model.predict(&[0.0, 0.0]).unwrap()[0];
    assert!(
        prediction_11 > prediction_00,
        "Model should predict higher for (1,1) than (0,0)"
    );
}

// ==================== Graph API Tests ====================

#[test]
fn test_graph_api_simple_inference() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(3);
    let x = input.buffer();
    let x = ops::dense(4, Activation::Relu, x);
    let output = ops::dense(1, Activation::Sigmoid, x);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, -1.0, 0.5];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 3]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), inference_result.len());
    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "Mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

#[test]
fn test_graph_api_weight_sharing() {
    let device = <TestBackend as Backend>::Device::default();

    // Create a reusable Dense operation for weight sharing
    let shared = Operation::dense(4, Activation::Relu);

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let y = shared.apply(x); // First use
    let z = shared.apply(y); // Second use - same weights!

    let output = ops::dense(1, Activation::Sigmoid, z);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let export = model.to_instruction_model_info();

    // With weight sharing, only 2 unique weight matrices
    assert_eq!(
        export.weights.len(),
        2,
        "Should have 2 unique weight sets due to sharing"
    );
    assert_eq!(export.buffer_sizes, vec![4, 4, 4, 1]);
}

#[test]
fn test_graph_api_multilayer_deep() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(5);
    let x = input.buffer();
    let x = ops::dense(8, Activation::Relu, x);
    let x = ops::dense(6, Activation::Relu, x);
    let x = ops::dense(4, Activation::Relu, x);
    let output = ops::dense(1, Activation::Sigmoid, x);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");
    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let test_inputs = vec![
        vec![1.0f32, 0.0, -1.0, 0.5, -0.5],
        vec![0.0, 0.0, 0.0, 0.0, 0.0],
        vec![-1.0, -1.0, -1.0, -1.0, -1.0],
    ];

    for inputs in test_inputs {
        let input_tensor =
            Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 5]);

        let burn_output = model.forward(input_tensor);
        let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

        let inference_result = inference_model
            .predict(&inputs)
            .expect("Inference should succeed");

        for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
            assert!(
                floats_close(*burn_val, *inference_val, TOLERANCE),
                "Mismatch for input {:?}: burn={}, inference={}",
                inputs,
                burn_val,
                inference_val
            );
        }
    }
}

#[test]
fn test_graph_api_add_operation() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();

    let branch1 = ops::dense(4, Activation::Relu, x.clone());
    let branch2 = ops::dense(4, Activation::Relu, x);

    let added = ops::add(vec![branch1, branch2]);
    let output = ops::dense(1, Activation::Sigmoid, added);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let test_inputs = vec![
        vec![1.0f32, 0.0, -1.0, 0.5],
        vec![0.5, 0.5, 0.5, 0.5],
        vec![-1.0, 1.0, 0.0, -0.5],
    ];

    for inputs in test_inputs {
        let input_tensor =
            Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

        let burn_output = model.forward(input_tensor);
        let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

        let inference_result = inference_model
            .predict(&inputs)
            .expect("Inference should succeed");

        assert_eq!(burn_result.len(), inference_result.len());
        for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
            assert!(
                floats_close(*burn_val, *inference_val, TOLERANCE),
                "Add op mismatch for input {:?}: burn={}, inference={}",
                inputs,
                burn_val,
                inference_val
            );
        }
    }
}

#[test]
fn test_graph_api_multiply_operation() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();

    let branch1 = ops::dense(4, Activation::Relu, x.clone());
    let branch2 = ops::dense(4, Activation::Relu, x);

    let multiplied = ops::multiply(vec![branch1, branch2]);
    let output = ops::dense(1, Activation::Sigmoid, multiplied);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let test_inputs = vec![
        vec![1.0f32, 0.0, -1.0, 0.5],
        vec![0.5, 0.5, 0.5, 0.5],
        vec![-1.0, 1.0, 0.0, -0.5],
    ];

    for inputs in test_inputs {
        let input_tensor =
            Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

        let burn_output = model.forward(input_tensor);
        let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

        let inference_result = inference_model
            .predict(&inputs)
            .expect("Inference should succeed");

        assert_eq!(burn_result.len(), inference_result.len());
        for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
            assert!(
                floats_close(*burn_val, *inference_val, TOLERANCE),
                "Multiply op mismatch for input {:?}: burn={}, inference={}",
                inputs,
                burn_val,
                inference_val
            );
        }
    }
}

#[test]
fn test_graph_api_residual_connection() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();

    // Residual: hidden + input
    let hidden = ops::dense(4, Activation::Relu, x.clone());
    let residual = ops::add(vec![hidden, x]);

    let output = ops::dense(1, Activation::Sigmoid, residual);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, -1.0, 0.5, 0.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), inference_result.len());
    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "Residual mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

// ==================== New Activation Tests ====================

#[test]
fn test_sqrt_activation() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let x = ops::dense(4, Activation::Sqrt, x);
    let output = ops::dense(1, Activation::Sigmoid, x);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    // Use positive inputs to test sqrt behavior
    let inputs = vec![1.0f32, 4.0, 9.0, 16.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), inference_result.len());
    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "SQRT activation mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

#[test]
fn test_log_activation() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let x = ops::dense(4, Activation::Log, x);
    let output = ops::dense(1, Activation::Sigmoid, x);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    // Use positive inputs to test log behavior
    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), inference_result.len());
    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "LOG activation mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

#[test]
fn test_log10_activation() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let x = ops::dense(4, Activation::Log10, x);
    let output = ops::dense(1, Activation::Sigmoid, x);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    // Use positive inputs to test log10 behavior
    let inputs = vec![9.0f32, 99.0, 999.0, 9999.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), inference_result.len());
    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "LOG10 activation mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

#[test]
fn test_inverse_activation() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let x = ops::dense(4, Activation::Inverse, x);
    let output = ops::dense(1, Activation::Sigmoid, x);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    // Test with various inputs
    let inputs = vec![0.0f32, 0.5, 1.0, -0.5];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), inference_result.len());
    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "INVERSE activation mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

#[test]
fn test_tanh_activation() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let x = ops::dense(4, Activation::Tanh, x);
    let output = ops::dense(1, Activation::Sigmoid, x);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![0.0f32, 0.5, -0.5, 1.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), inference_result.len());
    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "TANH activation mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

// ==================== Scale/Shift (ScaleVectorized/ShiftVectorized) Tests ====================

#[test]
fn test_scale_operation_inference() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let scaled = ops::scale(vec![2.0, 3.0, 4.0, 5.0], x);
    let output = ops::dense(1, Activation::Sigmoid, scaled);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), inference_result.len());
    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "Scale operation mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

#[test]
fn test_shift_operation_inference() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let shifted = ops::shift(vec![0.1, 0.2, 0.3, 0.4], x);
    let output = ops::dense(1, Activation::Sigmoid, shifted);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), inference_result.len());
    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "Shift operation mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

#[test]
fn test_scale_shift_combined_inference() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    // Apply scale then shift (like normalization: (x * scale) + shift)
    let scaled = ops::scale(vec![2.0, 2.0, 2.0, 2.0], x);
    let shifted = ops::shift(vec![1.0, 1.0, 1.0, 1.0], scaled);
    let output = ops::dense(1, Activation::Sigmoid, shifted);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), inference_result.len());
    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "Scale+Shift combined mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

// ==================== BatchNorm Tests ====================

#[test]
fn test_batch_norm_inference() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let normalized = ops::batch_norm(x);
    let output = ops::dense(1, Activation::Sigmoid, normalized);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), inference_result.len());
    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "BatchNorm inference mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

#[test]
fn test_batch_norm_with_dense_inference() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let hidden = ops::dense(8, Activation::Relu, x);
    let normalized = ops::batch_norm(hidden);
    let output = ops::dense(1, Activation::Sigmoid, normalized);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), inference_result.len());
    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "BatchNorm+Dense inference mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

#[test]
fn test_batch_norm_multiple_inputs() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let normalized = ops::batch_norm(x);
    let output = ops::dense(1, Activation::Sigmoid, normalized);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    // Test with multiple different inputs to verify consistency
    let test_inputs = vec![
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![0.0f32, 0.0, 0.0, 0.0],
        vec![-1.0f32, -2.0, 3.0, -4.0],
        vec![100.0f32, 200.0, 300.0, 400.0],
        vec![0.001f32, 0.002, 0.003, 0.004],
    ];

    for inputs in test_inputs {
        let input_tensor =
            Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

        let burn_output = model.forward(input_tensor);
        let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

        let inference_result = inference_model
            .predict(&inputs)
            .expect("Inference should succeed");

        assert_eq!(burn_result.len(), inference_result.len());
        for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
            assert!(
                floats_close(*burn_val, *inference_val, TOLERANCE),
                "BatchNorm inference mismatch for input {:?}: burn={}, inference={}",
                inputs,
                burn_val,
                inference_val
            );
        }
    }
}

#[test]
fn test_batch_norm_preserves_output_for_initial_state() {
    let device = <TestBackend as Backend>::Device::default();

    // For initial BatchNorm state (gamma=1, beta=0, mean=0, var=1):
    // output = (x - 0) / sqrt(1 + eps) + 0 = x / sqrt(1 + eps)
    // With eps=1e-3: output ≈ x * 0.9995

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let normalized = ops::batch_norm(x);
    // Use identity-like output (just pass through the normalized values)
    let output = ops::dense(4, Activation::None, normalized);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    // Verify both produce the same results
    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "BatchNorm initial state mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

#[test]
fn test_batch_norm_chained() {
    let device = <TestBackend as Backend>::Device::default();

    // Test chaining: Dense -> BatchNorm -> Dense -> BatchNorm -> Dense
    let input = InputBuffer::new(4);
    let x = input.buffer();
    let h1 = ops::dense(8, Activation::Relu, x);
    let n1 = ops::batch_norm(h1);
    let h2 = ops::dense(4, Activation::Relu, n1);
    let n2 = ops::batch_norm(h2);
    let output = ops::dense(1, Activation::Sigmoid, n2);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), inference_result.len());
    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "Chained BatchNorm mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

#[test]
fn test_batch_norm_with_residual() {
    let device = <TestBackend as Backend>::Device::default();

    // Test BatchNorm in a residual connection: x + BatchNorm(Dense(x))
    let input = InputBuffer::new(4);
    let x = input.buffer();
    let hidden = ops::dense(4, Activation::Relu, x.clone());
    let normalized = ops::batch_norm(hidden);
    let residual = ops::add(vec![x, normalized]);
    let output = ops::dense(1, Activation::Sigmoid, residual);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), inference_result.len());
    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "BatchNorm residual mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

#[test]
fn test_batch_norm_different_sizes() {
    let device = <TestBackend as Backend>::Device::default();

    // Test BatchNorm with different feature sizes
    for size in [2, 4, 8, 16, 32] {
        let input = InputBuffer::new(size);
        let x = input.buffer();
        let normalized = ops::batch_norm(x);
        let output = ops::dense(1, Activation::Sigmoid, normalized);

        let model = GraphModel::new(vec![input], output)
            .compile::<TestBackend>(&device)
            .expect("Model creation should succeed");

        let json = model
            .export_to_instruction_model()
            .expect("Export should succeed");

        let model_info: InstructionModelInfo =
            serde_json::from_str(&json).expect("JSON should be valid");
        let inference_model =
            InstructionModel::new(model_info).expect("Inference model creation should succeed");

        let inputs: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
        let input_tensor =
            Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, size]);

        let burn_output = model.forward(input_tensor);
        let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

        let inference_result = inference_model
            .predict(&inputs)
            .expect("Inference should succeed");

        assert_eq!(burn_result.len(), inference_result.len());
        for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
            assert!(
                floats_close(*burn_val, *inference_val, TOLERANCE),
                "BatchNorm size {} mismatch: burn={}, inference={}",
                size,
                burn_val,
                inference_val
            );
        }
    }
}

// ==================== Training Tests with Export Verification ====================
// These tests verify that:
// 1. Training works with the graph API (using Autodiff backend)
// 2. The trained model can be exported to instruction model format
// 3. The exported model produces the same output as the trained burn model

use burn::optim::{AdamConfig, GradientsParams, Optimizer};

const TRAINING_TOLERANCE: f32 = 1e-5;

/// Diagnostic test: Graph API with Autodiff backend but NO training
/// This tests if the issue is with Autodiff or with training specifically
#[test]
fn test_graph_api_autodiff_no_training() {
    let device = <TrainingBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let hidden = ops::dense(8, Activation::Relu, x);
    let output = ops::dense(1, Activation::Sigmoid, hidden);

    let model = GraphModel::new(vec![input], output)
        .compile::<TrainingBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TrainingBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.into_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "Autodiff no-training mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

/// Diagnostic test: ONE training step to isolate if the issue is training itself
#[test]
fn test_graph_api_single_training_step() {
    let device = <TrainingBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let hidden = ops::dense(8, Activation::Relu, x);
    let output = ops::dense(1, Activation::Sigmoid, hidden);

    let mut model = GraphModel::new(vec![input], output)
        .compile::<TrainingBackend>(&device)
        .expect("Model creation should succeed");

    let mut optimizer = AdamConfig::new().init();

    // Single training step
    let input_tensor = Tensor::<TrainingBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
    let target = Tensor::<TrainingBackend, 2>::from_floats([[0.5]], &device);

    let output_tensor = model.forward(input_tensor);
    let diff = output_tensor.sub(target);
    let loss = diff.clone().mul(diff).mean();

    let grads = loss.backward();
    let grads_params = GradientsParams::from_grads(grads, &model);
    model = optimizer.step(0.01, model, grads_params);

    // Export and verify
    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TrainingBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.into_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (i, (burn_val, inference_val)) in
        burn_result.iter().zip(inference_result.iter()).enumerate()
    {
        let diff = (burn_val - inference_val).abs();
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "Single step training output {} mismatch: burn={}, inference={}, diff={}",
            i,
            burn_val,
            inference_val,
            diff
        );
    }
}

#[test]
fn test_trained_graph_model_export_equivalence_simple() {
    let device = <TrainingBackend as Backend>::Device::default();

    // Simple model without fan-out/concat
    let input = InputBuffer::new(4);
    let x = input.buffer();
    let hidden = ops::dense(8, Activation::Relu, x);
    let output = ops::dense(1, Activation::Sigmoid, hidden);

    let model = GraphModel::new(vec![input], output)
        .compile::<TrainingBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TrainingBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.into_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "Autodiff no-training mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

/// Test fan-out/concat pattern after training - verifies that models with
/// branches that diverge and then concatenate maintain export equivalence.
#[test]
fn test_trained_graph_model_export_equivalence() {
    let device = <TrainingBackend as Backend>::Device::default();

    // Fan-out architecture: input branches to two dense layers, then concatenates
    let input = InputBuffer::new(4);
    let x = input.buffer();

    // Two branches from the same input (fan-out)
    let branch1 = ops::dense(3, Activation::Relu, x.clone());
    let branch2 = ops::dense(2, Activation::Tanh, x);

    // Concatenate the branches
    let concatenated = ops::concat(vec![branch1, branch2]);

    // Final output layer
    let output = ops::dense(1, Activation::Sigmoid, concatenated);

    let mut model = GraphModel::new(vec![input], output)
        .compile::<TrainingBackend>(&device)
        .expect("Model creation should succeed");

    let mut optimizer = AdamConfig::new().init();

    // Train for a few iterations
    for _ in 0..10 {
        let input_tensor =
            Tensor::<TrainingBackend, 2>::from_floats([[0.5, 1.0, 1.5, 2.0]], &device);
        let target = Tensor::<TrainingBackend, 2>::from_floats([[0.7]], &device);

        let output_tensor = model.forward(input_tensor);
        let diff = output_tensor.sub(target);
        let loss = diff.clone().mul(diff).mean();

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(0.01, model, grads_params);
    }

    // Export and verify
    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![0.5f32, 1.0, 1.5, 2.0];
    let input_tensor =
        Tensor::<TrainingBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.into_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TRAINING_TOLERANCE),
            "Fan-out/concat trained model mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

#[test]
fn test_trained_model_with_inverse_activation() {
    let device = <TrainingBackend as Backend>::Device::default();

    // Model using Inverse activation (1 - x)
    let input = InputBuffer::new(4);
    let x = input.buffer();
    let hidden = ops::dense(4, Activation::Sigmoid, x);
    let inverted = ops::dense(4, Activation::Inverse, hidden);
    let output = ops::dense(1, Activation::None, inverted);

    let mut model = GraphModel::new(vec![input], output)
        .compile::<TrainingBackend>(&device)
        .expect("Model creation should succeed");

    let mut optimizer = AdamConfig::new().init();

    // Train for a few iterations
    for _ in 0..10 {
        let input_tensor =
            Tensor::<TrainingBackend, 2>::from_floats([[0.5, 1.0, 1.5, 2.0]], &device);
        let target = Tensor::<TrainingBackend, 2>::from_floats([[0.3]], &device);

        let output_tensor = model.forward(input_tensor);
        let diff = output_tensor.sub(target);
        let loss = diff.clone().mul(diff).mean();

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(0.01, model, grads_params);
    }

    // Export and verify
    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![0.5f32, 1.0, 1.5, 2.0];
    let input_tensor =
        Tensor::<TrainingBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.into_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TRAINING_TOLERANCE),
            "Inverse activation trained model mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

#[test]
fn test_trained_model_with_all_activations() {
    let device = <TrainingBackend as Backend>::Device::default();

    // Test training and export for each activation type
    let activations = [
        Activation::Relu,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Inverse,
    ];

    for activation in activations {
        let input = InputBuffer::new(4);
        let x = input.buffer();
        let hidden = ops::dense(4, activation, x);
        let output = ops::dense(1, Activation::Sigmoid, hidden);

        let mut model = GraphModel::new(vec![input], output)
            .compile::<TrainingBackend>(&device)
            .expect("Model creation should succeed");

        let mut optimizer = AdamConfig::new().init();

        // Single training step
        let input_tensor =
            Tensor::<TrainingBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let target = Tensor::<TrainingBackend, 2>::from_floats([[0.5]], &device);

        let output_tensor = model.forward(input_tensor);
        let diff = output_tensor.sub(target);
        let loss = diff.clone().mul(diff).mean();

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(0.01, model, grads_params);

        // Export and verify
        let json = model
            .export_to_instruction_model()
            .expect("Export should succeed");

        let model_info: InstructionModelInfo =
            serde_json::from_str(&json).expect("JSON should be valid");
        let inference_model =
            InstructionModel::new(model_info).expect("Inference model creation should succeed");

        let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
        let input_tensor =
            Tensor::<TrainingBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

        let burn_output = model.forward(input_tensor);
        let burn_result: Vec<f32> = burn_output.into_data().to_vec().unwrap();

        let inference_result = inference_model
            .predict(&inputs)
            .expect("Inference should succeed");

        for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
            assert!(
                floats_close(*burn_val, *inference_val, TRAINING_TOLERANCE),
                "{:?} activation trained model mismatch: burn={}, inference={}",
                activation,
                burn_val,
                inference_val
            );
        }
    }
}

#[test]
fn test_trained_fan_in_add_export() {
    let device = <TrainingBackend as Backend>::Device::default();

    // Fan-in with add (residual-like pattern)
    let input = InputBuffer::new(4);
    let x = input.buffer();
    let branch1 = ops::dense(4, Activation::Relu, x.clone());
    let branch2 = ops::dense(4, Activation::Relu, x);
    let added = ops::add(vec![branch1, branch2]);
    let output = ops::dense(1, Activation::Sigmoid, added);

    let mut model = GraphModel::new(vec![input], output)
        .compile::<TrainingBackend>(&device)
        .expect("Model creation should succeed");

    let mut optimizer = AdamConfig::new().init();

    // Train
    for _ in 0..5 {
        let input_tensor =
            Tensor::<TrainingBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let target = Tensor::<TrainingBackend, 2>::from_floats([[0.6]], &device);

        let output_tensor = model.forward(input_tensor);
        let diff = output_tensor.sub(target);
        let loss = diff.clone().mul(diff).mean();

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(0.01, model, grads_params);
    }

    // Export and verify
    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TrainingBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.into_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TRAINING_TOLERANCE),
            "Fan-in add trained model mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

#[test]
fn test_trained_model_with_scale_and_inverse() {
    let device = <TrainingBackend as Backend>::Device::default();

    // Model combining scale(-1) with sigmoid to achieve similar effect to inverse
    // This pattern: sigmoid -> scale(-1) gives values in [-1, 0]
    let input = InputBuffer::new(4);
    let x = input.buffer();
    let hidden = ops::dense(4, Activation::Sigmoid, x);
    let scaled = ops::scale(vec![-1.0, -1.0, -1.0, -1.0], hidden);
    let output = ops::dense(1, Activation::None, scaled);

    let mut model = GraphModel::new(vec![input], output)
        .compile::<TrainingBackend>(&device)
        .expect("Model creation should succeed");

    let mut optimizer = AdamConfig::new().init();

    // Train
    for _ in 0..5 {
        let input_tensor =
            Tensor::<TrainingBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let target = Tensor::<TrainingBackend, 2>::from_floats([[-0.5]], &device);

        let output_tensor = model.forward(input_tensor);
        let diff = output_tensor.sub(target);
        let loss = diff.clone().mul(diff).mean();

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(0.01, model, grads_params);
    }

    // Export and verify
    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TrainingBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.into_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TRAINING_TOLERANCE),
            "Scale(-1) trained model mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

// ==================== Fan-in/Fan-out with Multiply Tests ====================

/// Test fan-out pattern with multiply gating (Swish-like)
#[test]
fn test_fan_out_multiply_gating() {
    let device = <TestBackend as Backend>::Device::default();

    // Fan-out pattern: input -> two branches -> multiply (gating mechanism)
    let input = InputBuffer::new(4);
    let x = input.buffer();

    // Two branches that will be multiplied together (like Swish: x * sigmoid(x))
    let linear_branch = ops::dense(4, Activation::None, x.clone());
    let gate_branch = ops::dense(4, Activation::Sigmoid, x);
    let gated = ops::multiply(vec![linear_branch, gate_branch]);

    let output = ops::dense(1, Activation::Sigmoid, gated);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let test_inputs = vec![
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![-1.0f32, 0.5, -0.5, 1.0],
        vec![0.0f32, 0.0, 0.0, 0.0],
    ];

    for inputs in test_inputs {
        let input_tensor =
            Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

        let burn_output = model.forward(input_tensor);
        let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

        let inference_result = inference_model
            .predict(&inputs)
            .expect("Inference should succeed");

        for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
            assert!(
                floats_close(*burn_val, *inference_val, TOLERANCE),
                "Fan-out multiply gating mismatch for {:?}: burn={}, inference={}",
                inputs,
                burn_val,
                inference_val
            );
        }
    }
}

/// Test fan-in with multiply (element-wise product of two branches)
#[test]
fn test_fan_in_multiply() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();

    let branch1 = ops::dense(4, Activation::Relu, x.clone());
    let branch2 = ops::dense(4, Activation::Sigmoid, x);
    let multiplied = ops::multiply(vec![branch1, branch2]);

    let output = ops::dense(1, Activation::None, multiplied);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "Fan-in multiply mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

/// Test triple fan-out with multiply (three branches multiplied together)
#[test]
fn test_triple_fan_out_multiply() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();

    let branch1 = ops::dense(4, Activation::Relu, x.clone());
    let branch2 = ops::dense(4, Activation::Sigmoid, x.clone());
    let branch3 = ops::dense(4, Activation::Tanh, x);

    let multiplied = ops::multiply(vec![branch1, branch2, branch3]);
    let output = ops::dense(1, Activation::Sigmoid, multiplied);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![0.5f32, 1.0, -0.5, 0.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "Triple fan-out multiply mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

/// Test two-head architecture with multiply gating and concat
#[test]
fn test_two_head_multiply_concat() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let hidden = ops::dense(8, Activation::Relu, x);

    // Head 1: Swish-like gating (x * sigmoid(x))
    let head1_linear = ops::dense(1, Activation::None, hidden.clone());
    let head1_gate = ops::dense(1, Activation::Sigmoid, hidden.clone());
    let head1 = ops::multiply(vec![head1_linear, head1_gate]);

    // Head 2: Negative Swish-like gating (-(x * sigmoid(x)))
    let head2_linear = ops::dense(1, Activation::None, hidden.clone());
    let head2_gate = ops::dense(1, Activation::Sigmoid, hidden);
    let head2_raw = ops::multiply(vec![head2_linear, head2_gate]);
    let head2 = ops::scale(vec![-1.0], head2_raw);

    let output = ops::concat(vec![head1, head2]);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), 2);
    assert_eq!(inference_result.len(), 2);

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "Two-head multiply concat mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

// ==================== BatchNorm with All Settings Tests ====================

/// Test BatchNorm with scale only (center=false, scale=true)
#[test]
fn test_batch_norm_scale_only() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let normalized = ops::batch_norm_scale_only(x);
    let output = ops::dense(1, Activation::Sigmoid, normalized);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let test_inputs = vec![
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![-1.0f32, -2.0, 3.0, -4.0],
        vec![0.0f32, 0.0, 0.0, 0.0],
    ];

    for inputs in test_inputs {
        let input_tensor =
            Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

        let burn_output = model.forward(input_tensor);
        let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

        let inference_result = inference_model
            .predict(&inputs)
            .expect("Inference should succeed");

        for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
            assert!(
                floats_close(*burn_val, *inference_val, TOLERANCE),
                "BatchNorm scale-only mismatch for {:?}: burn={}, inference={}",
                inputs,
                burn_val,
                inference_val
            );
        }
    }
}

/// Test BatchNorm with full config (center=true, scale=true) - default
#[test]
fn test_batch_norm_full_config() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    // Full batch norm: center=true, scale=true (default)
    let normalized = ops::batch_norm_config(1e-3, true, true, x);
    let output = ops::dense(1, Activation::Sigmoid, normalized);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "BatchNorm full config mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

/// Test BatchNorm with center only (center=true, scale=false)
#[test]
fn test_batch_norm_center_only() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    // Center only: center=true, scale=false
    let normalized = ops::batch_norm_config(1e-3, true, false, x);
    let output = ops::dense(1, Activation::Sigmoid, normalized);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "BatchNorm center-only mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

/// Test BatchNorm with neither center nor scale (center=false, scale=false)
#[test]
fn test_batch_norm_neither() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    // Neither: center=false, scale=false (just variance normalization)
    let normalized = ops::batch_norm_config(1e-3, false, false, x);
    let output = ops::dense(1, Activation::Sigmoid, normalized);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "BatchNorm neither mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

/// Test BatchNorm scale-only with different epsilon values
#[test]
fn test_batch_norm_scale_only_different_epsilons() {
    let device = <TestBackend as Backend>::Device::default();

    for epsilon in [1e-5, 1e-3, 1e-1] {
        let input = InputBuffer::new(4);
        let x = input.buffer();
        let normalized = ops::batch_norm_config(epsilon, false, true, x);
        let output = ops::dense(1, Activation::Sigmoid, normalized);

        let model = GraphModel::new(vec![input], output)
            .compile::<TestBackend>(&device)
            .expect("Model creation should succeed");

        let json = model
            .export_to_instruction_model()
            .expect("Export should succeed");

        let model_info: InstructionModelInfo =
            serde_json::from_str(&json).expect("JSON should be valid");
        let inference_model =
            InstructionModel::new(model_info).expect("Inference model creation should succeed");

        let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
        let input_tensor =
            Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

        let burn_output = model.forward(input_tensor);
        let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

        let inference_result = inference_model
            .predict(&inputs)
            .expect("Inference should succeed");

        for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
            assert!(
                floats_close(*burn_val, *inference_val, TOLERANCE),
                "BatchNorm scale-only epsilon {} mismatch: burn={}, inference={}",
                epsilon,
                burn_val,
                inference_val
            );
        }
    }
}

/// Test BatchNorm scale-only with dense layers before and after
#[test]
fn test_batch_norm_scale_only_with_dense() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let normalized = ops::batch_norm_scale_only(x);
    let hidden = ops::dense(8, Activation::Relu, normalized);
    let output = ops::dense(1, Activation::Sigmoid, hidden);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "BatchNorm scale-only with dense mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

/// Test BatchNorm with multiply gating pattern
#[test]
fn test_batch_norm_with_multiply_gating() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();

    // Input normalization (scale only)
    let normalized = ops::batch_norm_scale_only(x);

    // Hidden layer
    let hidden = ops::dense(8, Activation::Relu, normalized);

    // Swish-like gating with normalized input
    let linear = ops::dense(1, Activation::None, hidden.clone());
    let gate = ops::dense(1, Activation::Sigmoid, hidden);
    let output = ops::multiply(vec![linear, gate]);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "BatchNorm with multiply gating mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

// ==================== Training Tests with Multiply and BatchNorm ====================

/// Test trained model with fan-out multiply pattern
#[test]
fn test_trained_fan_out_multiply() {
    let device = <TrainingBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();

    let linear_branch = ops::dense(4, Activation::None, x.clone());
    let gate_branch = ops::dense(4, Activation::Sigmoid, x);
    let gated = ops::multiply(vec![linear_branch, gate_branch]);
    let output = ops::dense(1, Activation::Sigmoid, gated);

    let mut model = GraphModel::new(vec![input], output)
        .compile::<TrainingBackend>(&device)
        .expect("Model creation should succeed");

    let mut optimizer = AdamConfig::new().init();

    for _ in 0..10 {
        let input_tensor =
            Tensor::<TrainingBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let target = Tensor::<TrainingBackend, 2>::from_floats([[0.7]], &device);

        let output_tensor = model.forward(input_tensor);
        let diff = output_tensor.sub(target);
        let loss = diff.clone().mul(diff).mean();

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(0.01, model, grads_params);
    }

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TrainingBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.into_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TRAINING_TOLERANCE),
            "Trained fan-out multiply mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

/// Test trained model with BatchNorm scale-only
#[test]
fn test_trained_batch_norm_scale_only() {
    let device = <TrainingBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let normalized = ops::batch_norm_scale_only(x);
    let hidden = ops::dense(8, Activation::Relu, normalized);
    let output = ops::dense(1, Activation::Sigmoid, hidden);

    let mut model = GraphModel::new(vec![input], output)
        .compile::<TrainingBackend>(&device)
        .expect("Model creation should succeed");

    let mut optimizer = AdamConfig::new().init();

    for _ in 0..10 {
        let input_tensor =
            Tensor::<TrainingBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let target = Tensor::<TrainingBackend, 2>::from_floats([[0.5]], &device);

        let output_tensor = model.forward(input_tensor);
        let diff = output_tensor.sub(target);
        let loss = diff.clone().mul(diff).mean();

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(0.01, model, grads_params);
    }

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TrainingBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.into_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TRAINING_TOLERANCE),
            "Trained BatchNorm scale-only mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

/// Test trained model with full architecture: BatchNorm + multiply gating + two heads
#[test]
fn test_trained_full_architecture() {
    let device = <TrainingBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();

    // Input normalization (scale only)
    let normalized = ops::batch_norm_scale_only(x);

    // Shared hidden layer
    let hidden = ops::dense(8, Activation::Relu, normalized);

    // Head 1: Swish-like (x * sigmoid(x))
    let head1_linear = ops::dense(1, Activation::None, hidden.clone());
    let head1_gate = ops::dense(1, Activation::Sigmoid, hidden.clone());
    let head1 = ops::multiply(vec![head1_linear, head1_gate]);

    // Head 2: Negative Swish-like (-(x * sigmoid(x)))
    let head2_linear = ops::dense(1, Activation::None, hidden.clone());
    let head2_gate = ops::dense(1, Activation::Sigmoid, hidden);
    let head2_raw = ops::multiply(vec![head2_linear, head2_gate]);
    let head2 = ops::scale(vec![-1.0], head2_raw);

    let output = ops::concat(vec![head1, head2]);

    let mut model = GraphModel::new(vec![input], output)
        .compile::<TrainingBackend>(&device)
        .expect("Model creation should succeed");

    let mut optimizer = AdamConfig::new().init();

    for _ in 0..10 {
        let input_tensor =
            Tensor::<TrainingBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        // Target: [upside, downside]
        let target = Tensor::<TrainingBackend, 2>::from_floats([[0.3, -0.2]], &device);

        let output_tensor = model.forward(input_tensor);
        let diff = output_tensor.sub(target);
        let loss = diff.clone().mul(diff).mean();

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(0.01, model, grads_params);
    }

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_tensor =
        Tensor::<TrainingBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.into_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    assert_eq!(burn_result.len(), 2);
    assert_eq!(inference_result.len(), 2);

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TRAINING_TOLERANCE),
            "Trained full architecture mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

/// Test two-head architecture with scale operation in concat pattern.
/// This verifies that scale operations work correctly when their output
/// is concatenated with other operations.
#[test]
fn test_two_head_architecture_training() {
    let device = <TrainingBackend as Backend>::Device::default();

    // Two-head architecture: one branch with dense, one with scale
    let input = InputBuffer::new(4);
    let x = input.buffer();

    // Head 1: regular dense layer
    let head1 = ops::dense(2, Activation::Relu, x.clone());

    // Head 2: scale operation (element-wise multiply by fixed vector)
    // Scale by -1 to invert values (simulating a simple transformation)
    let head2 = ops::scale(vec![-1.0, -1.0, -1.0, -1.0], x);

    // Concatenate both heads
    let concatenated = ops::concat(vec![head1, head2]);

    // Final output layer
    let output = ops::dense(1, Activation::Sigmoid, concatenated);

    let mut model = GraphModel::new(vec![input], output)
        .compile::<TrainingBackend>(&device)
        .expect("Model creation should succeed");

    let mut optimizer = AdamConfig::new().init();

    // Train for a few iterations
    for _ in 0..10 {
        let input_tensor =
            Tensor::<TrainingBackend, 2>::from_floats([[0.5, 1.0, 1.5, 2.0]], &device);
        let target = Tensor::<TrainingBackend, 2>::from_floats([[0.6]], &device);

        let output_tensor = model.forward(input_tensor);
        let diff = output_tensor.sub(target);
        let loss = diff.clone().mul(diff).mean();

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(0.01, model, grads_params);
    }

    // Export and verify
    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![0.5f32, 1.0, 1.5, 2.0];
    let input_tensor =
        Tensor::<TrainingBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.into_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TRAINING_TOLERANCE),
            "Two-head architecture mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

/// Tests nested model structure - a sub-model applied multiple times with shared weights.
/// This clones Python's test_nested_model from instmodel.
#[test]
fn test_nested_model() {
    use instmodel::graph::InstructionExport;

    let device = <TestBackend as Backend>::Device::default();

    // Build sub-model graph: BatchNorm + Dense(3) + Dense(3)
    // Equivalent to Python: ff_model([3, 3, 3], NOT_INPLACE, "ll")
    let sub_input = InputBuffer::new(3);
    let normalized = ops::batch_norm(sub_input.buffer());
    let hidden = ops::dense(3, Activation::None, normalized);
    let sub_output = ops::dense(3, Activation::None, hidden);
    let sub_graph = GraphModel::new(vec![sub_input], sub_output);

    // === Test 1: Export sub-model independently ===
    let sub_model = sub_graph
        .compile::<TestBackend>(&device)
        .expect("Sub-model creation should succeed");
    let sub_export = sub_model.to_instruction_model_info();
    assert_eq!(
        sub_export.buffer_sizes,
        vec![3, 3, 3, 3],
        "Sub-model buffer sizes: [input=3, bn_out=3, dense1=3, dense2=3]"
    );
    assert_eq!(sub_export.weights.len(), 2, "Sub-model has 2 Dense layers");
    assert_eq!(
        sub_export.parameters.len(),
        3,
        "Sub-model has 3 BatchNorm parameters"
    );

    // === Test 2: Build and export final model ===
    let main_input = InputBuffer::new(3);
    let first_iteration = sub_graph.apply_single(main_input.buffer());
    let second_iteration = sub_graph.apply_single(first_iteration.clone());

    let concat = ops::concat(vec![main_input.buffer(), first_iteration, second_iteration]);
    let final_output = ops::dense(1, Activation::None, concat);

    let final_model = GraphModel::new(vec![main_input], final_output)
        .compile::<TestBackend>(&device)
        .expect("Final model creation should succeed");

    // Test forward pass
    let input = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
    let output = final_model.forward(input);
    assert_eq!(output.dims(), [1, 1]);

    let export = final_model.to_instruction_model_info();

    // === Test 3: Validate buffer sizes ===
    assert_eq!(
        export.buffer_sizes,
        vec![3, 3, 3, 3, 3, 3, 3, 9, 1],
        "Buffer sizes should match: [input, bn1, d1, d2, bn2, d3, d4, concat, out]"
    );

    // === Test 4: Validate weight count and sharing ===
    assert_eq!(
        export.weights.len(),
        3,
        "Should have 3 weight sets (2 from sub-model + 1 final)"
    );

    // === Test 5: Validate parameter count and sharing ===
    assert_eq!(
        export.parameters.len(),
        3,
        "Should have 3 parameter vectors (BatchNorm: center, std, beta) - SHARED"
    );

    // === Test 6: Validate exact instruction sequence (matches Python test_nested_model) ===
    // Python expected:
    // COPY 0→1, ADD 1 p0, MUL 1 p1, ADD 1 p2, DOT 1→2 w0, DOT 2→3 w1,
    // COPY 3→4, ADD 4 p0, MUL 4 p1, ADD 4 p2, DOT 4→5 w0, DOT 5→6 w1,
    // COPY 0→7 idx0, COPY 3→7 idx3, COPY 6→7 idx6, DOT 7→8 w2
    assert_eq!(
        export.instructions.len(),
        16,
        "Should have 16 instructions total"
    );

    // Verify instruction sequence matches Python exactly
    let expected_instructions: Vec<ExpectedInstruction<'_>> = vec![
        // First sub-model application (BatchNorm + 2 Dense)
        ("COPY", Some(0), Some(1), None, Some(0)), // COPY 0→1, internal_index 0
        ("ADD_ELEMENTWISE", Some(1), None, Some(0), None), // ADD input=1, params=0
        ("MUL_ELEMENTWISE", Some(1), None, Some(1), None), // MUL input=1, params=1
        ("ADD_ELEMENTWISE", Some(1), None, Some(2), None), // ADD input=1, params=2
        ("DOT", Some(1), Some(2), None, None),     // DOT 1→2, weights 0
        ("DOT", Some(2), Some(3), None, None),     // DOT 2→3, weights 1
        // Second sub-model application (BatchNorm + 2 Dense) - SHARED params/weights!
        ("COPY", Some(3), Some(4), None, Some(0)), // COPY 3→4, internal_index 0
        ("ADD_ELEMENTWISE", Some(4), None, Some(0), None), // ADD input=4, params=0 (SHARED)
        ("MUL_ELEMENTWISE", Some(4), None, Some(1), None), // MUL input=4, params=1 (SHARED)
        ("ADD_ELEMENTWISE", Some(4), None, Some(2), None), // ADD input=4, params=2 (SHARED)
        ("DOT", Some(4), Some(5), None, None),     // DOT 4→5, weights 0 (SHARED)
        ("DOT", Some(5), Some(6), None, None),     // DOT 5→6, weights 1 (SHARED)
        // Concatenate
        ("COPY", Some(0), Some(7), None, Some(0)), // COPY 0→7, internal_index 0
        ("COPY", Some(3), Some(7), None, Some(3)), // COPY 3→7, internal_index 3
        ("COPY", Some(6), Some(7), None, Some(6)), // COPY 6→7, internal_index 6
        // Final Dense
        ("DOT", Some(7), Some(8), None, None), // DOT 7→8, weights 2
    ];

    for (i, (expected_type, expected_input, expected_output, expected_params, expected_internal)) in
        expected_instructions.iter().enumerate()
    {
        let instr = &export.instructions[i];
        match instr {
            InstructionExport::Copy {
                input,
                output,
                internal_index,
            } => {
                assert_eq!(*expected_type, "COPY", "Instruction {} type mismatch", i);
                assert_eq!(
                    Some(*input),
                    *expected_input,
                    "Instruction {} COPY input mismatch",
                    i
                );
                assert_eq!(
                    Some(*output),
                    *expected_output,
                    "Instruction {} COPY output mismatch",
                    i
                );
                assert_eq!(
                    Some(*internal_index),
                    *expected_internal,
                    "Instruction {} COPY internal_index mismatch",
                    i
                );
            }
            InstructionExport::AddElementwise { input, parameters } => {
                assert_eq!(
                    *expected_type, "ADD_ELEMENTWISE",
                    "Instruction {} type mismatch",
                    i
                );
                assert_eq!(
                    Some(*input),
                    *expected_input,
                    "Instruction {} ADD input mismatch",
                    i
                );
                assert_eq!(
                    Some(*parameters),
                    *expected_params,
                    "Instruction {} ADD parameters mismatch",
                    i
                );
            }
            InstructionExport::MulElementwise { input, parameters } => {
                assert_eq!(
                    *expected_type, "MUL_ELEMENTWISE",
                    "Instruction {} type mismatch",
                    i
                );
                assert_eq!(
                    Some(*input),
                    *expected_input,
                    "Instruction {} MUL input mismatch",
                    i
                );
                assert_eq!(
                    Some(*parameters),
                    *expected_params,
                    "Instruction {} MUL parameters mismatch",
                    i
                );
            }
            InstructionExport::Dot {
                input,
                output,
                weights,
                ..
            } => {
                assert_eq!(*expected_type, "DOT", "Instruction {} type mismatch", i);
                assert_eq!(
                    Some(*input),
                    *expected_input,
                    "Instruction {} DOT input mismatch",
                    i
                );
                assert_eq!(
                    Some(*output),
                    *expected_output,
                    "Instruction {} DOT output mismatch",
                    i
                );
                // Validate weight indices for DOT instructions
                let expected_weights = match i {
                    4 | 10 => 0, // First dense in both sub-model applications
                    5 | 11 => 1, // Second dense in both sub-model applications
                    15 => 2,     // Final dense
                    _ => panic!("Unexpected DOT instruction at index {}", i),
                };
                assert_eq!(
                    *weights, expected_weights,
                    "Instruction {} DOT weights mismatch",
                    i
                );
            }
            _ => panic!("Unexpected instruction type at index {}: {:?}", i, instr),
        }
    }

    // === Test 7: Verify weight sharing explicitly ===
    let mut weight_usage = std::collections::HashMap::new();
    for instr in &export.instructions {
        if let InstructionExport::Dot { weights, .. } = instr {
            *weight_usage.entry(*weights).or_insert(0) += 1;
        }
    }
    assert_eq!(
        weight_usage.get(&0),
        Some(&2),
        "Weight 0 used twice (shared)"
    );
    assert_eq!(
        weight_usage.get(&1),
        Some(&2),
        "Weight 1 used twice (shared)"
    );
    assert_eq!(weight_usage.get(&2), Some(&1), "Weight 2 used once (final)");

    // === Test 8: Verify parameter sharing explicitly ===
    let mut param_usage = std::collections::HashMap::new();
    for instr in &export.instructions {
        match instr {
            InstructionExport::AddElementwise { parameters, .. }
            | InstructionExport::MulElementwise { parameters, .. } => {
                *param_usage.entry(*parameters).or_insert(0) += 1;
            }
            _ => {}
        }
    }
    assert_eq!(param_usage.get(&0), Some(&2), "Param 0 used twice (shared)");
    assert_eq!(param_usage.get(&1), Some(&2), "Param 1 used twice (shared)");
    assert_eq!(param_usage.get(&2), Some(&2), "Param 2 used twice (shared)");

    // === Test 9: Verify inference equivalence ===
    let json = final_model
        .export_to_instruction_model()
        .expect("Export should succeed");
    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(inputs.as_slice(), &device).reshape([1, 3]);

    let burn_output = final_model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "Nested model mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

/// Test Scale and Shift with in_place=True.
/// Mirrors Python's test_scale_and_shift_vectorized.
/// With in_place=True: no COPY instructions, operates directly on input buffer.
#[test]
fn test_scale_and_shift_inplace() {
    use instmodel::graph::InstructionExport;

    let device = <TestBackend as Backend>::Device::default();

    // Create input buffer
    let input_buffer = InputBuffer::new(3);

    // Apply scaling by [2, 0.5, 3] then shift by [1, -1, 0] (in-place)
    let scaled = ops::scale_inplace(vec![2.0, 0.5, 3.0], input_buffer.buffer());
    let shifted = ops::shift_inplace(vec![1.0, -1.0, 0.0], scaled);

    let model = GraphModel::new(vec![input_buffer], shifted)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let export = model.to_instruction_model_info();

    // === Validate buffer sizes ===
    // With in_place=True, both operations work on buffer 0 - only 1 buffer needed
    assert_eq!(
        export.buffer_sizes,
        vec![3],
        "In-place operations should use only 1 buffer"
    );

    // === Validate parameters ===
    assert_eq!(
        export.parameters.len(),
        2,
        "Should have 2 parameter vectors (scale and shift)"
    );
    assert_eq!(
        export.parameters[0],
        vec![2.0, 0.5, 3.0],
        "First parameter should be scale vector"
    );
    assert_eq!(
        export.parameters[1],
        vec![1.0, -1.0, 0.0],
        "Second parameter should be shift vector"
    );

    // === Validate instructions - NO COPY, just MUL and ADD on buffer 0 ===
    assert_eq!(
        export.instructions.len(),
        2,
        "In-place mode should have exactly 2 instructions (no COPY)"
    );

    // First instruction: MUL_ELEMENTWISE on buffer 0 with parameters 0
    match &export.instructions[0] {
        InstructionExport::MulElementwise { input, parameters } => {
            assert_eq!(*input, 0, "MUL should operate on buffer 0");
            assert_eq!(*parameters, 0, "MUL should use parameters 0");
        }
        other => panic!("Expected MulElementwise for instruction 0, got {:?}", other),
    }

    // Second instruction: ADD_ELEMENTWISE on buffer 0 with parameters 1
    match &export.instructions[1] {
        InstructionExport::AddElementwise { input, parameters } => {
            assert_eq!(*input, 0, "ADD should operate on buffer 0");
            assert_eq!(*parameters, 1, "ADD should use parameters 1");
        }
        other => panic!("Expected AddElementwise for instruction 1, got {:?}", other),
    }

    // === Validate numerical correctness ===
    // Test forward pass: output = (input * scale) + shift
    let test_input = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
    let output = model.forward(test_input);
    let output_data: Vec<f32> = output.to_data().to_vec().unwrap();

    // Expected: [1*2 + 1, 2*0.5 + (-1), 3*3 + 0] = [3, 0, 9]
    let expected = [3.0, 0.0, 9.0];
    for (i, (got, exp)) in output_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            floats_close(*got, *exp, TOLERANCE),
            "Position {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }

    // === Validate inference model matches ===
    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");
    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0, 3.0];
    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (i, (got, exp)) in inference_result.iter().zip(expected.iter()).enumerate() {
        assert!(
            floats_close(*got, *exp, TOLERANCE),
            "Inference position {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }
}

/// Test Scale and Shift with in_place=False (default).
/// Mirrors Python's test_scale_and_shift_not_inplace.
/// With in_place=False: COPY instructions are emitted, new buffers allocated.
#[test]
fn test_scale_and_shift_not_inplace() {
    use instmodel::graph::InstructionExport;

    let device = <TestBackend as Backend>::Device::default();

    // Create input buffer
    let input_buffer = InputBuffer::new(2);

    // Apply scaling then shift (NOT in-place, default behavior)
    let scaled = ops::scale(vec![2.0, 3.0], input_buffer.buffer());
    let shifted = ops::shift(vec![10.0, 20.0], scaled);

    let model = GraphModel::new(vec![input_buffer], shifted)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let export = model.to_instruction_model_info();

    // === Validate buffer sizes ===
    // With in_place=False: input=0, scale_output=1, shift_output=2
    assert_eq!(
        export.buffer_sizes,
        vec![2, 2, 2],
        "Not in-place should allocate 3 buffers"
    );

    // === Validate parameters ===
    assert_eq!(
        export.parameters.len(),
        2,
        "Should have 2 parameter vectors"
    );

    // === Validate instructions - COPY + MUL, COPY + ADD ===
    assert_eq!(
        export.instructions.len(),
        4,
        "Not in-place mode should have 4 instructions (COPY+MUL, COPY+ADD)"
    );

    // Instruction 0: COPY 0 → 1
    match &export.instructions[0] {
        InstructionExport::Copy {
            input,
            output,
            internal_index,
        } => {
            assert_eq!(*input, 0, "COPY should read from buffer 0");
            assert_eq!(*output, 1, "COPY should write to buffer 1");
            assert_eq!(*internal_index, 0, "COPY internal_index should be 0");
        }
        other => panic!("Expected Copy for instruction 0, got {:?}", other),
    }

    // Instruction 1: MUL_ELEMENTWISE on buffer 1
    match &export.instructions[1] {
        InstructionExport::MulElementwise { input, parameters } => {
            assert_eq!(*input, 1, "MUL should operate on buffer 1");
            assert_eq!(*parameters, 0, "MUL should use parameters 0");
        }
        other => panic!("Expected MulElementwise for instruction 1, got {:?}", other),
    }

    // Instruction 2: COPY 1 → 2
    match &export.instructions[2] {
        InstructionExport::Copy {
            input,
            output,
            internal_index,
        } => {
            assert_eq!(*input, 1, "COPY should read from buffer 1");
            assert_eq!(*output, 2, "COPY should write to buffer 2");
            assert_eq!(*internal_index, 0, "COPY internal_index should be 0");
        }
        other => panic!("Expected Copy for instruction 2, got {:?}", other),
    }

    // Instruction 3: ADD_ELEMENTWISE on buffer 2
    match &export.instructions[3] {
        InstructionExport::AddElementwise { input, parameters } => {
            assert_eq!(*input, 2, "ADD should operate on buffer 2");
            assert_eq!(*parameters, 1, "ADD should use parameters 1");
        }
        other => panic!("Expected AddElementwise for instruction 3, got {:?}", other),
    }

    // === Validate numerical correctness ===
    let test_input = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0]], &device);
    let output = model.forward(test_input);
    let output_data: Vec<f32> = output.to_data().to_vec().unwrap();

    // Expected: [1*2 + 10, 2*3 + 20] = [12, 26]
    let expected = [12.0, 26.0];
    for (i, (got, exp)) in output_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            floats_close(*got, *exp, TOLERANCE),
            "Position {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }

    // === Validate inference model matches ===
    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");
    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    let inputs = vec![1.0f32, 2.0];
    let inference_result = inference_model
        .predict(&inputs)
        .expect("Inference should succeed");

    for (i, (got, exp)) in inference_result.iter().zip(expected.iter()).enumerate() {
        assert!(
            floats_close(*got, *exp, TOLERANCE),
            "Inference position {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }
}

/// Test BatchNorm with in_place=True.
/// Verifies that BatchNorm with in_place=True operates directly on input buffer.
#[test]
fn test_batch_norm_inplace() {
    use instmodel::graph::InstructionExport;

    let device = <TestBackend as Backend>::Device::default();

    let input_buffer = InputBuffer::new(3);
    let normalized = ops::batch_norm_inplace(input_buffer.buffer());

    let model = GraphModel::new(vec![input_buffer], normalized)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let export = model.to_instruction_model_info();

    // === Validate buffer sizes ===
    // With in_place=True, BatchNorm works directly on buffer 0
    assert_eq!(
        export.buffer_sizes,
        vec![3],
        "In-place BatchNorm should use only 1 buffer"
    );

    // === Validate parameters ===
    // BatchNorm has 3 parameters: center (beta init), scale (1/std), shift (beta final)
    assert_eq!(
        export.parameters.len(),
        3,
        "BatchNorm should have 3 parameter vectors"
    );

    // === Validate instructions ===
    // In-place BatchNorm: ADD, MUL, ADD (no COPY)
    assert_eq!(
        export.instructions.len(),
        3,
        "In-place BatchNorm should have 3 instructions (no COPY)"
    );

    // Verify all instructions operate on buffer 0
    for (i, instr) in export.instructions.iter().enumerate() {
        match instr {
            InstructionExport::AddElementwise { input, .. }
            | InstructionExport::MulElementwise { input, .. } => {
                assert_eq!(*input, 0, "Instruction {} should operate on buffer 0", i);
            }
            other => panic!("Unexpected instruction type at position {}: {:?}", i, other),
        }
    }
}

/// Test BatchNorm with in_place=False (default).
/// Verifies that BatchNorm with in_place=False allocates a new buffer.
#[test]
fn test_batch_norm_not_inplace() {
    use instmodel::graph::InstructionExport;

    let device = <TestBackend as Backend>::Device::default();

    let input_buffer = InputBuffer::new(3);
    let normalized = ops::batch_norm(input_buffer.buffer()); // Default is in_place=false

    let model = GraphModel::new(vec![input_buffer], normalized)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let export = model.to_instruction_model_info();

    // === Validate buffer sizes ===
    // With in_place=False, BatchNorm allocates buffer 1
    assert_eq!(
        export.buffer_sizes,
        vec![3, 3],
        "Not in-place BatchNorm should use 2 buffers"
    );

    // === Validate instructions ===
    // Not in-place BatchNorm: COPY, ADD, MUL, ADD
    assert_eq!(
        export.instructions.len(),
        4,
        "Not in-place BatchNorm should have 4 instructions (COPY + 3 ops)"
    );

    // First instruction should be COPY 0 → 1
    match &export.instructions[0] {
        InstructionExport::Copy {
            input,
            output,
            internal_index,
        } => {
            assert_eq!(*input, 0, "COPY should read from buffer 0");
            assert_eq!(*output, 1, "COPY should write to buffer 1");
            assert_eq!(*internal_index, 0, "COPY internal_index should be 0");
        }
        other => panic!("Expected Copy for instruction 0, got {:?}", other),
    }

    // Remaining instructions should operate on buffer 1
    for (i, instr) in export.instructions.iter().skip(1).enumerate() {
        match instr {
            InstructionExport::AddElementwise { input, .. }
            | InstructionExport::MulElementwise { input, .. } => {
                assert_eq!(
                    *input,
                    1,
                    "Instruction {} should operate on buffer 1",
                    i + 1
                );
            }
            other => panic!(
                "Unexpected instruction type at position {}: {:?}",
                i + 1,
                other
            ),
        }
    }
}

/// Test that in_place=true produces the SAME output as in_place=false.
/// This is the critical correctness test: both modes must be numerically equivalent.
#[test]
fn test_inplace_produces_same_output_as_not_inplace() {
    let device = <TestBackend as Backend>::Device::default();

    // Test 1: Scale operation
    {
        let input_inplace = InputBuffer::new(3);
        let output_inplace = ops::scale_inplace(vec![2.0, 3.0, 4.0], input_inplace.buffer());
        let model_inplace = GraphModel::new(vec![input_inplace], output_inplace)
            .compile::<TestBackend>(&device)
            .expect("Model creation should succeed");

        let input_not_inplace = InputBuffer::new(3);
        let output_not_inplace = ops::scale(vec![2.0, 3.0, 4.0], input_not_inplace.buffer());
        let model_not_inplace = GraphModel::new(vec![input_not_inplace], output_not_inplace)
            .compile::<TestBackend>(&device)
            .expect("Model creation should succeed");

        let test_input = [1.0f32, 2.0, 3.0];
        let tensor_inplace =
            Tensor::<TestBackend, 1>::from_floats(test_input.as_slice(), &device).reshape([1, 3]);
        let tensor_not_inplace =
            Tensor::<TestBackend, 1>::from_floats(test_input.as_slice(), &device).reshape([1, 3]);

        let result_inplace: Vec<f32> = model_inplace
            .forward(tensor_inplace)
            .to_data()
            .to_vec()
            .unwrap();
        let result_not_inplace: Vec<f32> = model_not_inplace
            .forward(tensor_not_inplace)
            .to_data()
            .to_vec()
            .unwrap();

        for (i, (a, b)) in result_inplace
            .iter()
            .zip(result_not_inplace.iter())
            .enumerate()
        {
            assert!(
                floats_close(*a, *b, TOLERANCE),
                "Scale: in_place vs not_inplace differ at {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    // Test 2: Shift operation
    {
        let input_inplace = InputBuffer::new(3);
        let output_inplace = ops::shift_inplace(vec![10.0, 20.0, 30.0], input_inplace.buffer());
        let model_inplace = GraphModel::new(vec![input_inplace], output_inplace)
            .compile::<TestBackend>(&device)
            .expect("Model creation should succeed");

        let input_not_inplace = InputBuffer::new(3);
        let output_not_inplace = ops::shift(vec![10.0, 20.0, 30.0], input_not_inplace.buffer());
        let model_not_inplace = GraphModel::new(vec![input_not_inplace], output_not_inplace)
            .compile::<TestBackend>(&device)
            .expect("Model creation should succeed");

        let test_input = [1.0f32, 2.0, 3.0];
        let tensor_inplace =
            Tensor::<TestBackend, 1>::from_floats(test_input.as_slice(), &device).reshape([1, 3]);
        let tensor_not_inplace =
            Tensor::<TestBackend, 1>::from_floats(test_input.as_slice(), &device).reshape([1, 3]);

        let result_inplace: Vec<f32> = model_inplace
            .forward(tensor_inplace)
            .to_data()
            .to_vec()
            .unwrap();
        let result_not_inplace: Vec<f32> = model_not_inplace
            .forward(tensor_not_inplace)
            .to_data()
            .to_vec()
            .unwrap();

        for (i, (a, b)) in result_inplace
            .iter()
            .zip(result_not_inplace.iter())
            .enumerate()
        {
            assert!(
                floats_close(*a, *b, TOLERANCE),
                "Shift: in_place vs not_inplace differ at {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    // Test 3: BatchNorm operation
    {
        let input_inplace = InputBuffer::new(4);
        let output_inplace = ops::batch_norm_inplace(input_inplace.buffer());
        let model_inplace = GraphModel::new(vec![input_inplace], output_inplace)
            .compile::<TestBackend>(&device)
            .expect("Model creation should succeed");

        let input_not_inplace = InputBuffer::new(4);
        let output_not_inplace = ops::batch_norm(input_not_inplace.buffer());
        let model_not_inplace = GraphModel::new(vec![input_not_inplace], output_not_inplace)
            .compile::<TestBackend>(&device)
            .expect("Model creation should succeed");

        let test_input = [1.0f32, 2.0, 3.0, 4.0];
        let tensor_inplace =
            Tensor::<TestBackend, 1>::from_floats(test_input.as_slice(), &device).reshape([1, 4]);
        let tensor_not_inplace =
            Tensor::<TestBackend, 1>::from_floats(test_input.as_slice(), &device).reshape([1, 4]);

        let result_inplace: Vec<f32> = model_inplace
            .forward(tensor_inplace)
            .to_data()
            .to_vec()
            .unwrap();
        let result_not_inplace: Vec<f32> = model_not_inplace
            .forward(tensor_not_inplace)
            .to_data()
            .to_vec()
            .unwrap();

        for (i, (a, b)) in result_inplace
            .iter()
            .zip(result_not_inplace.iter())
            .enumerate()
        {
            assert!(
                floats_close(*a, *b, TOLERANCE),
                "BatchNorm: in_place vs not_inplace differ at {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    // Test 4: Combined Scale + Shift chain
    {
        let input_inplace = InputBuffer::new(3);
        let scaled_inplace = ops::scale_inplace(vec![2.0, 3.0, 4.0], input_inplace.buffer());
        let output_inplace = ops::shift_inplace(vec![1.0, 1.0, 1.0], scaled_inplace);
        let model_inplace = GraphModel::new(vec![input_inplace], output_inplace)
            .compile::<TestBackend>(&device)
            .expect("Model creation should succeed");

        let input_not_inplace = InputBuffer::new(3);
        let scaled_not_inplace = ops::scale(vec![2.0, 3.0, 4.0], input_not_inplace.buffer());
        let output_not_inplace = ops::shift(vec![1.0, 1.0, 1.0], scaled_not_inplace);
        let model_not_inplace = GraphModel::new(vec![input_not_inplace], output_not_inplace)
            .compile::<TestBackend>(&device)
            .expect("Model creation should succeed");

        let test_input = [1.0f32, 2.0, 3.0];
        let tensor_inplace =
            Tensor::<TestBackend, 1>::from_floats(test_input.as_slice(), &device).reshape([1, 3]);
        let tensor_not_inplace =
            Tensor::<TestBackend, 1>::from_floats(test_input.as_slice(), &device).reshape([1, 3]);

        let result_inplace: Vec<f32> = model_inplace
            .forward(tensor_inplace)
            .to_data()
            .to_vec()
            .unwrap();
        let result_not_inplace: Vec<f32> = model_not_inplace
            .forward(tensor_not_inplace)
            .to_data()
            .to_vec()
            .unwrap();

        // Expected: [1*2+1, 2*3+1, 3*4+1] = [3, 7, 13]
        let expected = [3.0, 7.0, 13.0];

        for (i, (a, b)) in result_inplace
            .iter()
            .zip(result_not_inplace.iter())
            .enumerate()
        {
            assert!(
                floats_close(*a, *b, TOLERANCE),
                "Chain: in_place vs not_inplace differ at {}: {} vs {}",
                i,
                a,
                b
            );
            assert!(
                floats_close(*a, expected[i], TOLERANCE),
                "Chain: result {} differs from expected: {} vs {}",
                i,
                a,
                expected[i]
            );
        }
    }
}

/// Test that in_place operations chain correctly - mixed in_place modes.
/// This tests a realistic scenario where some operations are in_place and others are not.
#[test]
fn test_mixed_inplace_operations() {
    use instmodel::graph::InstructionExport;

    let device = <TestBackend as Backend>::Device::default();

    let input_buffer = InputBuffer::new(4);

    // Chain: Scale(in_place=false) → Shift(in_place=true) → Scale(in_place=false)
    // Buffer 0: input
    // Buffer 1: after first Scale (COPY 0→1, MUL 1)
    // Shift in_place on buffer 1 (ADD 1, no new buffer)
    // Buffer 2: after second Scale (COPY 1→2, MUL 2)
    let x = ops::scale(vec![1.0, 2.0, 3.0, 4.0], input_buffer.buffer()); // not in_place
    let x = ops::shift_inplace(vec![0.5, 0.5, 0.5, 0.5], x); // in_place
    let x = ops::scale(vec![2.0, 2.0, 2.0, 2.0], x); // not in_place

    let model = GraphModel::new(vec![input_buffer], x)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let export = model.to_instruction_model_info();

    // === Validate buffer sizes ===
    // Buffer 0: input, Buffer 1: first scale output, Buffer 2: second scale output
    // Shift in_place modifies buffer 1
    assert_eq!(
        export.buffer_sizes,
        vec![4, 4, 4],
        "Mixed in_place should use 3 buffers"
    );

    // === Validate instruction count ===
    // COPY 0→1, MUL 1 (scale), ADD 1 (shift in_place), COPY 1→2, MUL 2 (scale)
    assert_eq!(
        export.instructions.len(),
        5,
        "Mixed mode should have 5 instructions"
    );

    // Verify instruction sequence
    // 0: COPY 0→1
    match &export.instructions[0] {
        InstructionExport::Copy { input, output, .. } => {
            assert_eq!(*input, 0);
            assert_eq!(*output, 1);
        }
        other => panic!("Expected Copy at 0, got {:?}", other),
    }

    // 1: MUL on buffer 1
    match &export.instructions[1] {
        InstructionExport::MulElementwise { input, .. } => {
            assert_eq!(*input, 1);
        }
        other => panic!("Expected MulElementwise at 1, got {:?}", other),
    }

    // 2: ADD on buffer 1 (in_place shift)
    match &export.instructions[2] {
        InstructionExport::AddElementwise { input, .. } => {
            assert_eq!(*input, 1, "In-place shift should operate on buffer 1");
        }
        other => panic!("Expected AddElementwise at 2, got {:?}", other),
    }

    // 3: COPY 1→2
    match &export.instructions[3] {
        InstructionExport::Copy { input, output, .. } => {
            assert_eq!(*input, 1);
            assert_eq!(*output, 2);
        }
        other => panic!("Expected Copy at 3, got {:?}", other),
    }

    // 4: MUL on buffer 2
    match &export.instructions[4] {
        InstructionExport::MulElementwise { input, .. } => {
            assert_eq!(*input, 2);
        }
        other => panic!("Expected MulElementwise at 4, got {:?}", other),
    }

    // === Validate numerical correctness ===
    // input = [1, 2, 3, 4]
    // after scale: [1*1, 2*2, 3*3, 4*4] = [1, 4, 9, 16]
    // after shift: [1+0.5, 4+0.5, 9+0.5, 16+0.5] = [1.5, 4.5, 9.5, 16.5]
    // after scale: [1.5*2, 4.5*2, 9.5*2, 16.5*2] = [3, 9, 19, 33]
    let test_input = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
    let output = model.forward(test_input);
    let output_data: Vec<f32> = output.to_data().to_vec().unwrap();

    let expected = [3.0, 9.0, 19.0, 33.0];
    for (i, (got, exp)) in output_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            floats_close(*got, *exp, TOLERANCE),
            "Position {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }
}

// ==================== Standalone Activation Tests ====================

/// Test standalone activation with sigmoid (in_place=false by default)
#[test]
fn test_standalone_activation_sigmoid() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    // Apply sigmoid activation without a dense layer
    let output = ops::activation(Activation::Sigmoid, x);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    // Test: sigmoid(0) = 0.5
    let input_tensor = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0, 0.0, 0.0]], &device);
    let result = model.forward(input_tensor);
    let data: Vec<f32> = result.to_data().to_vec().unwrap();

    for val in &data {
        assert!(
            floats_close(*val, 0.5, TOLERANCE),
            "sigmoid(0) should be 0.5, got {}",
            val
        );
    }
}

/// Test standalone activation with relu
#[test]
fn test_standalone_activation_relu() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let output = ops::activation(Activation::Relu, x);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    // relu([-1, 0, 1, 2]) = [0, 0, 1, 2]
    let input_tensor = Tensor::<TestBackend, 2>::from_floats([[-1.0, 0.0, 1.0, 2.0]], &device);
    let result = model.forward(input_tensor);
    let data: Vec<f32> = result.to_data().to_vec().unwrap();

    let expected = [0.0, 0.0, 1.0, 2.0];
    for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            floats_close(*got, *exp, TOLERANCE),
            "relu mismatch at {}: got {}, expected {}",
            i,
            got,
            exp
        );
    }
}

/// Test standalone activation in-place export (no COPY instruction)
#[test]
fn test_activation_inplace_export() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    // in_place=true: no COPY instruction in export
    let output = ops::activation_inplace(Activation::Relu, x);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    // Should have ACTIVATION but no COPY
    assert!(
        json.contains("ACTIVATION"),
        "Should have ACTIVATION instruction"
    );
    assert!(
        !json.contains("COPY"),
        "In-place should not have COPY instruction"
    );
}

/// Test standalone activation NOT in-place export (has COPY instruction)
#[test]
fn test_activation_not_inplace_export() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    // in_place=false (default): COPY then ACTIVATION
    let output = ops::activation(Activation::Relu, x);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    // Should have both COPY and ACTIVATION
    assert!(
        json.contains("COPY"),
        "Not in-place should have COPY instruction"
    );
    assert!(
        json.contains("ACTIVATION"),
        "Should have ACTIVATION instruction"
    );
}

/// Test standalone activation with inference engine
#[test]
fn test_standalone_activation_inference() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();
    let output = ops::activation(Activation::Sigmoid, x);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    let json = model
        .export_to_instruction_model()
        .expect("Export should succeed");

    let model_info: InstructionModelInfo =
        serde_json::from_str(&json).expect("JSON should be valid");
    let inference_model =
        InstructionModel::new(model_info).expect("Inference model creation should succeed");

    // Compare burn and inference outputs
    let test_inputs = vec![0.0f32, 1.0, -1.0, 2.0];
    let input_tensor =
        Tensor::<TestBackend, 1>::from_floats(test_inputs.as_slice(), &device).reshape([1, 4]);

    let burn_output = model.forward(input_tensor);
    let burn_result: Vec<f32> = burn_output.to_data().to_vec().unwrap();

    let inference_result = inference_model
        .predict(&test_inputs)
        .expect("Inference should succeed");

    for (burn_val, inference_val) in burn_result.iter().zip(inference_result.iter()) {
        assert!(
            floats_close(*burn_val, *inference_val, TOLERANCE),
            "Standalone activation mismatch: burn={}, inference={}",
            burn_val,
            inference_val
        );
    }
}

/// Test activation after dense (common pattern: concat branches then sigmoid)
#[test]
fn test_activation_after_concat() {
    let device = <TestBackend as Backend>::Device::default();

    let input = InputBuffer::new(4);
    let x = input.buffer();

    // Two branches
    let branch1 = ops::dense(1, Activation::None, x.clone());
    let branch2 = ops::dense(1, Activation::None, x);

    // Concat then apply sigmoid
    let concat = ops::concat(vec![branch1, branch2]);
    let output = ops::activation(Activation::Sigmoid, concat);

    let model = GraphModel::new(vec![input], output)
        .compile::<TestBackend>(&device)
        .expect("Model creation should succeed");

    assert_eq!(model.output_size(), 2);

    let input_tensor = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
    let result = model.forward(input_tensor);

    assert_eq!(result.dims(), [1, 2]);

    // Output should be sigmoid-bounded (0, 1)
    let data: Vec<f32> = result.to_data().to_vec().unwrap();
    for val in &data {
        assert!(
            *val > 0.0 && *val < 1.0,
            "Sigmoid output should be in (0, 1), got {}",
            val
        );
    }
}

/// Tests that submodel extraction produces correct outputs matching the parent model.
/// This verifies the weight sharing implementation works correctly.
#[test]
fn test_submodel_weight_sharing() {
    let device = <TestBackend as Backend>::Device::default();

    // Create a sub-graph for buy signal
    let buy_input = InputBuffer::new(4);
    let buy_hidden = ops::dense(8, Activation::Relu, buy_input.buffer());
    let buy_out = ops::dense(1, Activation::Sigmoid, buy_hidden);
    let buy_graph = GraphModel::new(vec![buy_input], buy_out);

    // Create a sub-graph for sell signal
    let sell_input = InputBuffer::new(4);
    let sell_hidden = ops::dense(8, Activation::Relu, sell_input.buffer());
    let sell_out = ops::dense(1, Activation::Sigmoid, sell_hidden);
    let sell_graph = GraphModel::new(vec![sell_input], sell_out);

    // Create combined graph using both sub-graphs
    let main_input = InputBuffer::new(4);
    let x = main_input.buffer();
    let buy_result = buy_graph.apply_single(x.clone());
    let sell_result = sell_graph.apply_single(x);
    let combined = ops::concat(vec![buy_result, sell_result]);
    let combined_graph = GraphModel::new(vec![main_input], combined);

    // Compile only the combined graph
    let compiled = combined_graph
        .compile::<TestBackend>(&device)
        .expect("Combined model creation should succeed");

    // Extract sub-models (they use cloned weights from compiled)
    let buy_model = compiled.submodel(&buy_graph);
    let sell_model = compiled.submodel(&sell_graph);

    // Test: combined output should equal concatenated sub-model outputs
    let test_input = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
    let combined_result = compiled.forward(test_input.clone());
    let buy_result = buy_model.forward(test_input.clone());
    let sell_result = sell_model.forward(test_input);

    let combined_data: Vec<f32> = combined_result.to_data().to_vec().unwrap();
    let buy_data: Vec<f32> = buy_result.to_data().to_vec().unwrap();
    let sell_data: Vec<f32> = sell_result.to_data().to_vec().unwrap();

    // Combined[0] should exactly equal buy output
    assert!(
        (combined_data[0] - buy_data[0]).abs() < 1e-6,
        "Buy weight sharing failed: {} != {}",
        combined_data[0],
        buy_data[0]
    );

    // Combined[1] should exactly equal sell output
    assert!(
        (combined_data[1] - sell_data[0]).abs() < 1e-6,
        "Sell weight sharing failed: {} != {}",
        combined_data[1],
        sell_data[0]
    );
}
