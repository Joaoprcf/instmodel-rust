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

const TOLERANCE: f32 = 1e-5;

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

    let model = GraphModel::<TestBackend>::new(vec![input], output, &device)
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

    let model = GraphModel::<TestBackend>::new(vec![input], output, &device)
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

    let model = GraphModel::<TestBackend>::new(vec![input], output, &device)
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

    let model = GraphModel::<TestBackend>::new(vec![input], output, &device)
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

    let model = GraphModel::<TestBackend>::new(vec![input], output, &device)
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

    let model = GraphModel::<TestBackend>::new(vec![input], output, &device)
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

    let model = GraphModel::<TestBackend>::new(vec![input], output, &device)
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

    let model = GraphModel::<TestBackend>::new(vec![input], output, &device)
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

    let model = GraphModel::<TestBackend>::new(vec![input], output, &device)
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

    let model = GraphModel::<TestBackend>::new(vec![input], output, &device)
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

    let model = GraphModel::<TestBackend>::new(vec![input], output, &device)
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

    let model = GraphModel::<TestBackend>::new(vec![input], output, &device)
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

    let model = GraphModel::<TestBackend>::new(vec![input], output, &device)
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

    let model = GraphModel::<TestBackend>::new(vec![input], output, &device)
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
