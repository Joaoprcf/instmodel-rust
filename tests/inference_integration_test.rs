//! Exported models must reproduce Burn's forward pass in `instmodel_inference`.

use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, backend::Backend};
use instmodel::graph::{
    CompiledModel, DataBuffer, InputBuffer, ModelGraph as GraphModel, Operation, ops,
};
use instmodel::layers::Activation as A;
use instmodel::model_graph::{ModelGraph, ModelGraphConfig};
use instmodel::training::{Loss, TrainingConfig, train};
use instmodel_inference::{InstructionModel, InstructionModelInfo};
use ops::{batch_norm, concat, dense, multiply};
use serde_json::{Value, json};

type TestBackend = NdArray;
type TrainingBackend = Autodiff<NdArray>;

const TOLERANCE: f32 = 1e-6;
const TRAINING_TOLERANCE: f32 = 1e-5;

type Graph = (InputBuffer, DataBuffer);
/// (name, graph, buffer sizes, instructions, input, expected output)
type LayoutCase = (
    &'static str,
    Graph,
    Vec<usize>,
    Vec<Value>,
    Vec<f32>,
    Vec<f32>,
);

fn row<B: Backend>(x: &[f32]) -> Tensor<B, 2> {
    Tensor::<B, 1>::from_floats(x, &Default::default()).reshape([1, x.len()])
}

fn to_vec<B: Backend>(t: Tensor<B, 2>) -> Vec<f32> {
    t.into_data().to_vec().unwrap()
}

fn assert_close(name: &str, got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len(), "{name}: length mismatch");
    for (g, e) in got.iter().zip(expected) {
        assert!((g - e).abs() < tol, "{name}: {got:?} != {expected:?}");
    }
}

fn inference(json: &str) -> InstructionModel {
    let info: InstructionModelInfo = serde_json::from_str(json).expect("valid JSON");
    InstructionModel::new(info).expect("inference model")
}

/// Compares Burn's forward pass with the inference engine on every sample.
fn assert_matches_inference<B: Backend>(
    name: &str,
    json: &str,
    forward: impl Fn(Tensor<B, 2>) -> Tensor<B, 2>,
    samples: &[Vec<f32>],
    tol: f32,
) {
    let engine = inference(json);
    for x in samples {
        let expected = to_vec(forward(row::<B>(x)));
        assert_close(name, &engine.predict(x).unwrap(), &expected, tol);
    }
}

fn graph(size: usize, body: impl FnOnce(DataBuffer) -> DataBuffer) -> Graph {
    let input = InputBuffer::new(size);
    let output = body(input.buffer());
    (input, output)
}

fn compile<B: Backend>((input, output): Graph) -> CompiledModel<B> {
    GraphModel::new(vec![input], output)
        .compile::<B>(&Default::default())
        .expect("compile")
}

fn x4() -> Vec<Vec<f32>> {
    vec![vec![1.0, 2.0, 3.0, 4.0]]
}

/// Two-head block: [x·σ(x), -(x·σ(x))] from two independent gated projections.
fn two_heads(hidden: DataBuffer) -> DataBuffer {
    let gated = |h: &DataBuffer| {
        multiply(vec![
            dense(1, A::None, h.clone()),
            dense(1, A::Sigmoid, h.clone()),
        ])
    };
    let head1 = gated(&hidden);
    let head2 = ops::scale(vec![-1.0], gated(&hidden));
    concat(vec![head1, head2])
}

#[test]
fn sequential_models_match_inference() {
    let cases: [(usize, &[(usize, A)], _); 4] = [
        (2, &[(1, A::None)], vec![vec![1.0, 2.0]]),
        (
            3,
            &[(4, A::Relu), (2, A::None)],
            vec![
                vec![1.0, 0.0, -1.0],
                vec![0.5, 0.5, 0.5],
                vec![-1.0, 1.0, 0.0],
            ],
        ),
        (
            4,
            &[(8, A::Relu), (1, A::Sigmoid)],
            vec![vec![0.5, -0.5, 1.0, 0.0]],
        ),
        (
            5,
            &[(10, A::Relu), (8, A::Relu), (4, A::Relu), (1, A::Sigmoid)],
            vec![vec![1.0, -1.0, 0.5, 0.0, -0.5]],
        ),
    ];
    for (features, layers, samples) in cases {
        let config = layers.iter().fold(
            ModelGraphConfig::with_feature_size(features),
            |c, &(n, a)| c.dense(n, a),
        );
        let model: ModelGraph<TestBackend> = config.build(&Default::default()).unwrap();
        let json = model.export_to_instruction_model().unwrap();
        let name = format!("sequential {features} {layers:?}");
        assert_matches_inference(&name, &json, |x| model.forward(x), &samples, TOLERANCE);
    }
}

#[test]
fn sequential_export_layout() {
    let model: ModelGraph<TestBackend> = ModelGraphConfig::new(vec!["a".into(), "b".into()])
        .dense(3, A::Relu)
        .dense(1, A::Sigmoid)
        .build(&Default::default())
        .unwrap();
    let export = model.to_instruction_model_info();
    assert_eq!(export.features, Some(vec!["a".into(), "b".into()]));
    assert_eq!(export.feature_size, Some(2));
    assert_eq!(export.buffer_sizes, vec![2, 3, 1]);
    assert_eq!(export.instructions.len(), 2);
    let shapes: Vec<_> = export
        .weights
        .iter()
        .map(|w| (w.len(), w[0].len()))
        .collect();
    assert_eq!(shapes, vec![(3, 2), (1, 3)]);
    assert_eq!(export.bias.iter().map(Vec::len).collect::<Vec<_>>(), [3, 1]);
}

#[test]
fn trained_sequential_model_matches_inference() {
    let device = Default::default();
    let model: ModelGraph<TrainingBackend> = ModelGraphConfig::with_feature_size(2)
        .dense(8, A::Relu)
        .dense(1, A::Sigmoid)
        .build(&device)
        .unwrap();
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
    let history = &result.loss_history;
    assert!(
        history.last() < history.first(),
        "training should reduce loss"
    );

    let json = result.model.export_to_instruction_model().unwrap();
    assert_matches_inference(
        "trained sequential",
        &json,
        |x| result.model.forward(x),
        &inputs,
        TOLERANCE,
    );
    let engine = inference(&json);
    assert!(engine.predict(&[1.0, 1.0]).unwrap()[0] > engine.predict(&[0.0, 0.0]).unwrap()[0]);
}

#[test]
fn graph_models_match_inference() {
    let three = vec![
        vec![1.0, 0.0, -1.0, 0.5],
        vec![0.5, 0.5, 0.5, 0.5],
        vec![-1.0, 1.0, 0.0, -0.5],
    ];
    let branches = |merge: fn(Vec<DataBuffer>) -> DataBuffer| {
        graph(4, move |x| {
            let merged = merge(vec![dense(4, A::Relu, x.clone()), dense(4, A::Relu, x)]);
            dense(1, A::Sigmoid, merged)
        })
    };
    let bn_head = |size: usize, bn: fn(DataBuffer) -> DataBuffer| {
        graph(size, move |x| dense(1, A::Sigmoid, bn(x)))
    };
    let mut cases = vec![
        (
            "simple".to_string(),
            graph(3, |x| dense(1, A::Sigmoid, dense(4, A::Relu, x))),
            vec![vec![1.0, -1.0, 0.5]],
        ),
        (
            "deep".into(),
            graph(5, |x| {
                let x = dense(6, A::Relu, dense(8, A::Relu, x));
                dense(1, A::Sigmoid, dense(4, A::Relu, x))
            }),
            vec![vec![1.0, 0.0, -1.0, 0.5, -0.5], vec![0.0; 5], vec![-1.0; 5]],
        ),
        ("add".into(), branches(ops::add), three.clone()),
        ("multiply".into(), branches(ops::multiply), three),
        (
            "residual".into(),
            graph(4, |x| {
                let residual = ops::add(vec![dense(4, A::Relu, x.clone()), x]);
                dense(1, A::Sigmoid, residual)
            }),
            vec![vec![1.0, -1.0, 0.5, 0.0]],
        ),
        (
            "scale".into(),
            graph(4, |x| {
                dense(1, A::Sigmoid, ops::scale(vec![2.0, 3.0, 4.0, 5.0], x))
            }),
            x4(),
        ),
        (
            "shift".into(),
            graph(4, |x| {
                dense(1, A::Sigmoid, ops::shift(vec![0.1, 0.2, 0.3, 0.4], x))
            }),
            x4(),
        ),
        (
            "scale+shift".into(),
            graph(4, |x| {
                let x = ops::shift(vec![1.0; 4], ops::scale(vec![2.0; 4], x));
                dense(1, A::Sigmoid, x)
            }),
            x4(),
        ),
        (
            "batch_norm".into(),
            bn_head(4, batch_norm),
            vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![0.0; 4],
                vec![-1.0, -2.0, 3.0, -4.0],
                vec![100.0, 200.0, 300.0, 400.0],
                vec![0.001, 0.002, 0.003, 0.004],
            ],
        ),
        (
            "batch_norm scale-only".into(),
            bn_head(4, ops::batch_norm_scale_only),
            vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![-1.0, -2.0, 3.0, -4.0],
                vec![0.0; 4],
            ],
        ),
        (
            "dense→batch_norm→dense".into(),
            graph(4, |x| {
                dense(1, A::Sigmoid, batch_norm(dense(8, A::Relu, x)))
            }),
            x4(),
        ),
        (
            "batch_norm→dense(4)".into(),
            graph(4, |x| dense(4, A::None, batch_norm(x))),
            x4(),
        ),
        (
            "batch_norm chained".into(),
            graph(4, |x| {
                let n1 = batch_norm(dense(8, A::Relu, x));
                dense(1, A::Sigmoid, batch_norm(dense(4, A::Relu, n1)))
            }),
            x4(),
        ),
        (
            "batch_norm residual".into(),
            graph(4, |x| {
                let normalized = batch_norm(dense(4, A::Relu, x.clone()));
                dense(1, A::Sigmoid, ops::add(vec![x, normalized]))
            }),
            x4(),
        ),
        (
            "batch_norm scale-only→dense→dense".into(),
            graph(4, |x| {
                let h = dense(8, A::Relu, ops::batch_norm_scale_only(x));
                dense(1, A::Sigmoid, h)
            }),
            x4(),
        ),
        (
            "batch_norm gating".into(),
            graph(4, |x| {
                let h = dense(8, A::Relu, ops::batch_norm_scale_only(x));
                multiply(vec![dense(1, A::None, h.clone()), dense(1, A::Sigmoid, h)])
            }),
            x4(),
        ),
        (
            "fan-out gating".into(),
            graph(4, |x| {
                let gated = multiply(vec![dense(4, A::None, x.clone()), dense(4, A::Sigmoid, x)]);
                dense(1, A::Sigmoid, gated)
            }),
            vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![-1.0, 0.5, -0.5, 1.0],
                vec![0.0; 4],
            ],
        ),
        (
            "fan-in multiply".into(),
            graph(4, |x| {
                let m = multiply(vec![dense(4, A::Relu, x.clone()), dense(4, A::Sigmoid, x)]);
                dense(1, A::None, m)
            }),
            x4(),
        ),
        (
            "triple multiply".into(),
            graph(4, |x| {
                let m = multiply(vec![
                    dense(4, A::Relu, x.clone()),
                    dense(4, A::Sigmoid, x.clone()),
                    dense(4, A::Tanh, x),
                ]);
                dense(1, A::Sigmoid, m)
            }),
            vec![vec![0.5, 1.0, -0.5, 0.0]],
        ),
        (
            "two heads".into(),
            graph(4, |x| two_heads(dense(8, A::Relu, x))),
            x4(),
        ),
        (
            "standalone sigmoid".into(),
            graph(4, |x| ops::activation(A::Sigmoid, x)),
            vec![vec![0.0, 1.0, -1.0, 2.0]],
        ),
        (
            "sigmoid after concat".into(),
            graph(4, |x| {
                let c = concat(vec![dense(1, A::None, x.clone()), dense(1, A::None, x)]);
                ops::activation(A::Sigmoid, c)
            }),
            x4(),
        ),
    ];
    let activation_inputs = [
        (A::Sqrt, [1.0, 4.0, 9.0, 16.0]),
        (A::Log, [1.0, 2.0, 3.0, 4.0]),
        (A::Log10, [9.0, 99.0, 999.0, 9999.0]),
        (A::Inverse, [0.0, 0.5, 1.0, -0.5]),
        (A::Tanh, [0.0, 0.5, -0.5, 1.0]),
    ];
    for (act, x) in activation_inputs {
        let g = graph(4, |b| dense(1, A::Sigmoid, dense(4, act, b)));
        cases.push((format!("{act:?}"), g, vec![x.to_vec()]));
    }
    for size in [2, 4, 8, 16, 32] {
        let samples = vec![(1..=size).map(|i| i as f32).collect()];
        cases.push((
            format!("batch_norm size {size}"),
            bn_head(size, batch_norm),
            samples,
        ));
    }
    for (center, scale) in [(true, true), (true, false), (false, false)] {
        let g = graph(4, |x| {
            dense(
                1,
                A::Sigmoid,
                ops::batch_norm_config(1e-3, center, scale, x),
            )
        });
        cases.push((format!("batch_norm c={center} s={scale}"), g, x4()));
    }
    for eps in [1e-5, 1e-3, 1e-1] {
        let g = graph(4, |x| {
            dense(1, A::Sigmoid, ops::batch_norm_config(eps, false, true, x))
        });
        cases.push((format!("batch_norm scale-only eps={eps}"), g, x4()));
    }

    for (name, g, samples) in cases {
        let model = compile::<TestBackend>(g);
        let json = model.export_to_instruction_model().unwrap();
        assert_matches_inference(&name, &json, |x| model.forward(x), &samples, TOLERANCE);
    }
}

fn train_steps(
    mut model: CompiledModel<TrainingBackend>,
    x: &[f32],
    target: &[f32],
    steps: usize,
) -> CompiledModel<TrainingBackend> {
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..steps {
        let diff = model.forward(row(x)) - row(target);
        let loss = (diff.clone() * diff).mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.01, model, grads);
    }
    model
}

#[test]
fn trained_graph_models_match_inference() {
    let xa = vec![1.0, 2.0, 3.0, 4.0];
    let xb = vec![0.5, 1.0, 1.5, 2.0];
    let mut cases = vec![
        (
            "untrained autodiff".to_string(),
            graph(4, |x| dense(1, A::Sigmoid, dense(8, A::Relu, x))),
            xa.clone(),
            vec![0.5],
            0,
        ),
        (
            "single step".into(),
            graph(4, |x| dense(1, A::Sigmoid, dense(8, A::Relu, x))),
            xa.clone(),
            vec![0.5],
            1,
        ),
        (
            "fan-out concat".into(),
            graph(4, |x| {
                let c = concat(vec![dense(3, A::Relu, x.clone()), dense(2, A::Tanh, x)]);
                dense(1, A::Sigmoid, c)
            }),
            xb.clone(),
            vec![0.7],
            10,
        ),
        (
            "inverse".into(),
            graph(4, |x| {
                dense(1, A::None, dense(4, A::Inverse, dense(4, A::Sigmoid, x)))
            }),
            xb.clone(),
            vec![0.3],
            10,
        ),
        (
            "fan-in add".into(),
            graph(4, |x| {
                let added = ops::add(vec![dense(4, A::Relu, x.clone()), dense(4, A::Relu, x)]);
                dense(1, A::Sigmoid, added)
            }),
            xa.clone(),
            vec![0.6],
            5,
        ),
        (
            "scale(-1)".into(),
            graph(4, |x| {
                dense(
                    1,
                    A::None,
                    ops::scale(vec![-1.0; 4], dense(4, A::Sigmoid, x)),
                )
            }),
            xa.clone(),
            vec![-0.5],
            5,
        ),
        (
            "fan-out gating".into(),
            graph(4, |x| {
                let gated = multiply(vec![dense(4, A::None, x.clone()), dense(4, A::Sigmoid, x)]);
                dense(1, A::Sigmoid, gated)
            }),
            xa.clone(),
            vec![0.7],
            10,
        ),
        (
            "batch_norm scale-only".into(),
            graph(4, |x| {
                dense(
                    1,
                    A::Sigmoid,
                    dense(8, A::Relu, ops::batch_norm_scale_only(x)),
                )
            }),
            xa.clone(),
            vec![0.5],
            10,
        ),
        (
            "batch_norm + two heads".into(),
            graph(4, |x| {
                two_heads(dense(8, A::Relu, ops::batch_norm_scale_only(x)))
            }),
            xa.clone(),
            vec![0.3, -0.2],
            10,
        ),
        (
            "dense + scale heads".into(),
            graph(4, |x| {
                let heads = concat(vec![
                    dense(2, A::Relu, x.clone()),
                    ops::scale(vec![-1.0; 4], x),
                ]);
                dense(1, A::Sigmoid, heads)
            }),
            xb,
            vec![0.6],
            10,
        ),
    ];
    for act in [A::Relu, A::Sigmoid, A::Tanh, A::Inverse] {
        let g = graph(4, |x| dense(1, A::Sigmoid, dense(4, act, x)));
        cases.push((format!("one step {act:?}"), g, xa.clone(), vec![0.5], 1));
    }

    for (name, g, x, target, steps) in cases {
        let model = train_steps(compile(g), &x, &target, steps);
        let json = model.export_to_instruction_model().unwrap();
        let tol = if steps == 0 {
            TOLERANCE
        } else {
            TRAINING_TOLERANCE
        };
        assert_matches_inference(&name, &json, |t| model.forward(t), &[x], tol);
    }
}

fn instructions(model: &CompiledModel<TestBackend>) -> Vec<Value> {
    let export = model.to_instruction_model_info();
    export
        .instructions
        .iter()
        .map(|i| serde_json::to_value(i).unwrap())
        .collect()
}

fn copy(input: usize, output: usize, at: usize) -> Value {
    json!({"type": "COPY", "input": input, "output": output, "internal_index": at})
}

fn add_p(input: usize, parameters: usize) -> Value {
    json!({"type": "ADD_ELEMENTWISE", "input": input, "parameters": parameters})
}

fn mul_p(input: usize, parameters: usize) -> Value {
    json!({"type": "MUL_ELEMENTWISE", "input": input, "parameters": parameters})
}

fn dot(input: usize, output: usize, weights: usize) -> Value {
    json!({"type": "DOT", "input": input, "output": output, "weights": weights})
}

fn activation(input: usize, name: &str) -> Value {
    json!({"type": "ACTIVATION", "input": input, "activation": name})
}

/// Exact export layout for weight-free graphs, plus exact outputs from Burn and inference.
#[test]
fn export_layouts_and_values() {
    let bn = |x: f32| x / (1.0f32 + 1e-3).sqrt();
    let bn_ops = |buf: usize| vec![add_p(buf, 0), mul_p(buf, 1), add_p(buf, 2)];
    let cases: Vec<LayoutCase> = vec![
        (
            "scale+shift in place",
            graph(3, |x| {
                ops::shift_inplace(
                    vec![1.0, -1.0, 0.0],
                    ops::scale_inplace(vec![2.0, 0.5, 3.0], x),
                )
            }),
            vec![3],
            vec![mul_p(0, 0), add_p(0, 1)],
            vec![1.0, 2.0, 3.0],
            vec![3.0, 0.0, 9.0],
        ),
        (
            "scale+shift copied",
            graph(2, |x| {
                ops::shift(vec![10.0, 20.0], ops::scale(vec![2.0, 3.0], x))
            }),
            vec![2, 2, 2],
            vec![copy(0, 1, 0), mul_p(1, 0), copy(1, 2, 0), add_p(2, 1)],
            vec![1.0, 2.0],
            vec![12.0, 26.0],
        ),
        (
            "scale in place",
            graph(3, |x| ops::scale_inplace(vec![2.0, 3.0, 4.0], x)),
            vec![3],
            vec![mul_p(0, 0)],
            vec![1.0, 2.0, 3.0],
            vec![2.0, 6.0, 12.0],
        ),
        (
            "shift copied",
            graph(3, |x| ops::shift(vec![10.0, 20.0, 30.0], x)),
            vec![3, 3],
            vec![copy(0, 1, 0), add_p(1, 0)],
            vec![1.0, 2.0, 3.0],
            vec![11.0, 22.0, 33.0],
        ),
        (
            "identity activation copied",
            graph(2, |x| ops::activation(A::None, x)),
            vec![2, 2],
            vec![copy(0, 1, 0)],
            vec![-1.0, 2.0],
            vec![-1.0, 2.0],
        ),
        (
            "identity activation in place",
            graph(2, |x| {
                ops::activation_inplace(A::None, ops::scale_inplace(vec![2.0, 3.0], x))
            }),
            vec![2],
            vec![mul_p(0, 0)],
            vec![-1.0, 2.0],
            vec![-2.0, 6.0],
        ),
        (
            "mixed in-place chain",
            graph(4, |x| {
                let x = ops::shift_inplace(vec![0.5; 4], ops::scale(vec![1.0, 2.0, 3.0, 4.0], x));
                ops::scale(vec![2.0; 4], x)
            }),
            vec![4, 4, 4],
            vec![
                copy(0, 1, 0),
                mul_p(1, 0),
                add_p(1, 1),
                copy(1, 2, 0),
                mul_p(2, 2),
            ],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![3.0, 9.0, 19.0, 33.0],
        ),
        (
            "batch_norm in place",
            graph(3, ops::batch_norm_inplace),
            vec![3],
            bn_ops(0),
            vec![1.0, -2.0, 3.0],
            vec![bn(1.0), bn(-2.0), bn(3.0)],
        ),
        (
            "batch_norm copied",
            graph(3, batch_norm),
            vec![3, 3],
            [vec![copy(0, 1, 0)], bn_ops(1)].concat(),
            vec![1.0, -2.0, 3.0],
            vec![bn(1.0), bn(-2.0), bn(3.0)],
        ),
        (
            "activation copied",
            graph(4, |x| ops::activation(A::Sigmoid, x)),
            vec![4, 4],
            vec![copy(0, 1, 0), activation(1, "SIGMOID")],
            vec![0.0; 4],
            vec![0.5; 4],
        ),
        (
            "activation in place",
            graph(4, |x| ops::activation_inplace(A::Relu, x)),
            vec![4],
            vec![activation(0, "RELU")],
            vec![-1.0, 0.0, 1.0, 2.0],
            vec![0.0, 0.0, 1.0, 2.0],
        ),
    ];
    for (name, g, sizes, instrs, x, expected) in cases {
        let model = compile::<TestBackend>(g);
        assert_eq!(
            model.to_instruction_model_info().buffer_sizes,
            sizes,
            "{name}"
        );
        assert_eq!(instructions(&model), instrs, "{name}");
        assert_close(name, &to_vec(model.forward(row(&x))), &expected, TOLERANCE);
        let engine = inference(&model.export_to_instruction_model().unwrap());
        assert_close(name, &engine.predict(&x).unwrap(), &expected, TOLERANCE);
    }

    let export = compile::<TestBackend>(graph(3, |x| {
        ops::shift_inplace(
            vec![1.0, -1.0, 0.0],
            ops::scale_inplace(vec![2.0, 0.5, 3.0], x),
        )
    }))
    .to_instruction_model_info();
    assert_eq!(
        export.parameters,
        vec![vec![2.0, 0.5, 3.0], vec![1.0, -1.0, 0.0]]
    );
}

#[test]
fn shared_dense_exports_one_weight_set() {
    let shared = Operation::dense(4, A::Relu);
    let model = compile::<TestBackend>(graph(4, |x| {
        dense(1, A::Sigmoid, shared.apply(shared.apply(x)))
    }));
    let export = model.to_instruction_model_info();
    assert_eq!(export.weights.len(), 2);
    assert_eq!(export.buffer_sizes, vec![4, 4, 4, 1]);
    let json = model.export_to_instruction_model().unwrap();
    assert_matches_inference(
        "shared dense",
        &json,
        |x| model.forward(x),
        &x4(),
        TOLERANCE,
    );
}

/// Mirrors Python's test_nested_model: a sub-model applied twice shares weights and parameters.
#[test]
fn nested_model() {
    let sub_input = InputBuffer::new(3);
    let sub_output = dense(
        3,
        A::None,
        dense(3, A::None, batch_norm(sub_input.buffer())),
    );
    let sub_graph = GraphModel::new(vec![sub_input], sub_output);

    let sub_export = sub_graph
        .compile::<TestBackend>(&Default::default())
        .unwrap()
        .to_instruction_model_info();
    assert_eq!(sub_export.buffer_sizes, vec![3, 3, 3, 3]);
    assert_eq!(
        (sub_export.weights.len(), sub_export.parameters.len()),
        (2, 3)
    );

    let model = compile::<TestBackend>(graph(3, |x| {
        let first = sub_graph.apply_single(x.clone());
        let second = sub_graph.apply_single(first.clone());
        dense(1, A::None, concat(vec![x, first, second]))
    }));
    let export = model.to_instruction_model_info();
    assert_eq!(export.buffer_sizes, vec![3, 3, 3, 3, 3, 3, 3, 9, 1]);
    assert_eq!((export.weights.len(), export.parameters.len()), (3, 3));

    let application = |input: usize, base: usize| {
        let bn = base;
        vec![
            copy(input, bn, 0),
            add_p(bn, 0),
            mul_p(bn, 1),
            add_p(bn, 2),
            dot(bn, bn + 1, 0),
            dot(bn + 1, bn + 2, 1),
        ]
    };
    let expected = [
        application(0, 1),
        application(3, 4),
        vec![copy(0, 7, 0), copy(3, 7, 3), copy(6, 7, 6), dot(7, 8, 2)],
    ]
    .concat();
    assert_eq!(instructions(&model), expected);

    let json = model.export_to_instruction_model().unwrap();
    let samples = vec![vec![1.0, 2.0, 3.0]];
    assert_matches_inference("nested", &json, |x| model.forward(x), &samples, TOLERANCE);
}

#[test]
fn submodels_reuse_parent_weights() {
    let head = || {
        let input = InputBuffer::new(4);
        let out = dense(1, A::Sigmoid, dense(8, A::Relu, input.buffer()));
        GraphModel::new(vec![input], out)
    };
    let (buy, sell) = (head(), head());
    let model = compile::<TestBackend>(graph(4, |x| {
        concat(vec![buy.apply_single(x.clone()), sell.apply_single(x)])
    }));
    let x = [1.0, 2.0, 3.0, 4.0];
    let combined = to_vec(model.forward(row(&x)));
    let parts = [buy, sell].map(|g| to_vec(model.submodel(&g).forward(row(&x)))[0]);
    assert_close("submodels", &parts, &combined, TOLERANCE);
}
