# Edge AI Toolchain

A hands-on project for exploring the complete lifecycle of machine learning models from training to deployment on edge devices.

## Goals

Experiment with:

* training models from scratch with PyTorch
* transfer learning with pretrained models
* model evaluation on desktop
* model compression, including quantization and pruning
* ONNX export and model conversion
* deployment and inference on embedded targets such as the ESP32-S3
* comparison of accuracy, model size, memory usage and latency

The project uses notebooks for experimentation and a small reusable codebase for tested components.

## Approach

```text
Dataset
  ↓
Model
  ↓
Training / Transfer Learning
  ↓
Evaluation
  ↓
Compression
  ↓
Export
  ↓
Edge deployment
  ↓
Benchmark
```

Python and PyTorch handle the ML workflow.

Rust may be used for strongly typed orchestration, validation and tooling where compile-time guarantees are useful.

Tests define the expected behaviour of reusable components and model transformations.

## First milestone

Start with a single end-to-end pipeline:

```text
MNIST
→ Small CNN
→ PyTorch training
→ Desktop inference
→ ONNX export
→ INT8 quantization
→ ESP32-S3 inference
→ Desktop vs edge comparison
```

More datasets, pretrained models, compression methods and hardware targets can be added once this pipeline works reliably.
