# MASI

This repository contains the code and artifact evaluation guide for *"MASI: Memory-Adaptive Inference Framework for Spiking Neural Networks on Edge Devices"* (****). [[paper]]()

![end-to-end](<assets/end-to-end.png>)

Please refer to [Evaluation.md](Evaluation.md) for detailed instructions to evaluate MASI and reproduce the results.

## Introduction

MASI is  an adaptive memory management framework for on-device SNNs under dynamic memory constraints.

For a given memory budget and SNN model, MASI uses a memory-adaptive slicing strategy to partition bottleneck layers, a timestep-agnostic scheduler that maximizes memory utilization while avoiding the overhead of per-timestep planning, and a timestep-aware early-exit mechanism that skips redundant timesteps to accelerate inference.
With the support of these mechanisms,MASI dynamically adapts to varying memory budgets with minimal overhead, enabling efficient, responsive,
and scalable on-device SNN inference.

## Design

![overview](<assets/overview.png>)

MASI adopts a 2-stages design: The first stage is offline planning, where model slicing, loading, and computation are jointly optimized based on the memory budget and the given SNN architecture. 
The second stage is an online execution phase, where model inference is performed on devices according to the offline plan, 
augmented with a timestep-aware early-exit mechanism to reduce latency. It is efficient and adaptive with the following designs:

- **Memory-Adaptive Layer Slicing Strategy** ($\S 3.2$) that partitions the heavy computational load into multiple slices, computes each slice independently, and merges their outputs to obtain the final result, thereby reducing peak memory consumption.
- **Timestep-Agnostic Memory Scheduler** ($\S 3.3$) that effectively reducing fragmentation and minimizing I/O waiting time during SNN inference.
- **Timestep-Aware Execution with Early Exit** ($\S 3.4$) that eliminates redundant computation for easy inputs, thereby reducing latency often below that of the original implementation and achieving a more favorable trade-off between accuracy and efficiency.

For more details, please refer to our [paper]().

## Implementation

MASI is built atop [NCNN](https://github.com/Tencent/ncnn), which is a high-performance Neural Network inference framework optimized for the mobile platform. It is implemented and best-optimized for floating-point CNN inference on ARMv8 CPUs.

With a given memory budget and SNN model, MASI flexibly slices the model and plans memory allocations and the execution order. The complete steps of `flexnndemo` are listed as follows:

- Offline Planning
  - Layer Slicing - implemented in `run_slice`.
  - Memory Profiling - implemented in `run_profile`.
  - Memory Planning - implemented in `run_schedule`.
- Online Execution
  - Model Inference - implemented in `run_infer_flexnn_random_data/run_infer_flexnn_random_data/run_infer_flexnn`.

For more details, please refer to our [paper]() and [Evaluation.md](Evaluation.md).

## Evaluation

Please refer to [Evaluation.md](Evaluation.md) for detailed instructions to deploy and evaluate MASI.

The current MASI implementation is mainly for research purpose and has a few limitations. Please also refer to [Evaluation.md](Evaluation.md) for details.

## Citation

If you find MASI useful for your research, please cite our [paper]().

```bibtex

```
