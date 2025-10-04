# FlexNN Evaluation Guide

This is the artifact evaluation guide for paper *"MASI: Memory-Adaptive Inference Framework for Spiking Neural Networks on Edge Devices"*.

In this paper, we proposed MASI, an adaptive memory management framework for on-device SNNs under dynamic memory constraints. MASI uses a memory-adaptive slicing strategy to partition bottleneck layers, a timestep-agnostic scheduler that maximizes memory utilization while avoiding the overhead of per-timestep planning, and a timestep-aware early-exit mechanism that skips redundant timesteps to accelerate inference.

We implemented MASI atop [NCNN](https://github.com/Tencent/ncnn), and conducted comprehensive evaluations with common model architectures on various devices. This documentation describes the complete workflow of MASI, and provides detailed instructions to build, deploy, and evaluate MASI on your device.

## Preparation

The complete evaluation will cost about 2~3 hours.

### Devices

To conduct the complete evaluation, it is recommended that you have:

- *(Optional since we provide alternative ways other than building from source)* a **host machine** (e.g., a PC) to build MASI and process raw results, meeting the following requirements.
  - Hardware:
    - x86-64 CPUs.
    - Available RAM >= 2GB.
    - Available Disk >= 4GB.
  - OS: Ubuntu
  - Libs: git, cmake, g++, protobuf, OpenMP.
  - Tools: ADB (for Android target), or SSH (for Linux target).
- A **target device** (e.g., a smartphone) to deploy and evaluation MASI, meeting the following requirements:
  - Hardware:
    - ARMv8 CPUs (>= 2 cores).
    - Available RAM >= 2GB.
    - Available Storage >= 3GB.
  - OS: Ubuntu/Android with root access.

Note that MASI should also support other hardware and software configurations, but the provided workflows and scripts are verified only with the above configuration.

### Necessary Files

All necessary files to evaluate FlexNN have been uploaded to a zip. [[Share Link]]() Files include:

- **models**: SNN models required during evaluation. About 2GB in total.
- *(Optional)* **prebuilt**: Pre-built binaries (Linux/Android) of MASI.

## Overview

### MASI Workflow

With a given memory budget and SNN model, MASI flexibly slices the model and plans memory allocations and the execution order. The complete workflow contains the following steps (details in $\S 3$ in the paper):

- Offline Planning
  - Layer Slicing - implemented in `run_slice`.
  - Memory Profiling - implemented in `run_profile`.
  - Memory Planning - implemented in `run_schedule`.
- Online Execution
  - Model Inference - implemented in `run_infer_flexnn_random_data/run_infer_flexnn_random_data/run_infer_flexnn`.

### Evaluation Steps

The evaluation of MASI involves the following steps:

- [Building](#building): we provide 2 options for you to build MASI
  - Pre-built binaries.
  - Build with your own environment.
- [Deploying](#deployment): we provide 2 options to push files to different target OS
  - ADB for Android.
  - SSH for Linux.
- [Conducting Experiments](#evaluation): there are 4 different groups of experiments, you can verify them by setting some specific parameters.
  - latency-memory-tradeoff comparison.
  - power monitoring.
  - MASI ablation study.
  - Real-time latency and memory usage monitoring.

## Building

Before continuing, ensure that you have **downloaded the [necessary files](#necessary-files).**

### Pre-built Binaries

The pre-built binaries are provided together with their building folders. After downloading and unzipping them, directly **copy the folders to the root directory** of MASI. For example:

```bash
# Android build
cp -r predata/build-android-aarch <flexnn-root-dir>
# Linux build
cp -r predata/build-aarch64-linux-gnu <flexnn-root-dir> 
```

This is to ensure that our scripts to push files to the target device can work as expected. After this, **go ahead to [deploying](#deployment).**

## Deployment

Deploying MASI to the device involves pushing all the required binaries, models, and scripts to the device, and setting up the workspace on the device side. Before continuing with this step, **ensure that you have downloaded and unzipped the model files**, and put the `models` folder in `<flexnn-root-dir>`.

The total size of the files is around 2 GB. The required time may vary from seconds to minutes, depending on how you connect to the device (e.g., USB, Wi-Fi, etc). We've provided scripts to push all the necessary files to the target device, or you can follow the script command to push the files when it fails.

### Android Script

We recommend using [Android Debug Bridge (ADB)](https://developer.android.com/tools/adb) for Android devices. After installing ADB and successfully connecting to your device (either with a USB link or Wi-Fi, use `adb devices` to confirm the connection), run:

```bash
chmod -R 0777 ./scripts/host/
./scripts/host/adb-push-all.sh
```

Note that the Docker container will need access to your USB or WiFi interface to connect to the Android device.

### Linux Script

We recommend using SSH and SFTP for Linux devices. **Ensure that you have set up SSH key pair authentication to the device**, because the script uses an SFTP batch file, which is not supported by password authentication. Refer to [What is SSH Public Key Authentication?](https://www.ssh.com/academy/ssh/public-key-authentication#setting-up-public-key-authentication-for-ssh)

In `scripts/host/sftp-push-all.sh`, modify `device_tmp_dir`, `device_root_dir`, `username`, and `hostname` accordingly. Then run:

```bash
chmod -R 0777 ./scripts/host/
./scripts/host/sftp-push-all.sh
```

### Manual

In case the provided scripts don't work as expected, you can also push the necessary files manually by any possible means. Just make sure that your working directory is organized like:

```bash
- bin
  - ... (binaries)
- models
  - ncnn
    - ... (ncnn models)
  - flexnn
    - ... (flexnn models)
- profiles (empty folder)
- schedules (empty folder)
- ... (scripts)
```

Otherwise the scripts might not run the experiments normally.

## Execution

In this part, we will introduce the usage of binary as mentioned in [MASI Workflow](#masi-workflow), and provide an example to automatically run the workflow. **If you are looking for steps to run the experiments, head directly to the [Evaluation](#evaluation) part.**

Typically, to slice, plan, and inference under a given memory budget, the user should run `flexnndemo` with specific parameters. In most cases, for some memory budget, `flexnndemo` need to be executed again with a set of more proper parameters to ensure the program run correctly.

### Binaries Usage

Usage of `flexnndemo`:

```bash
Usage: ./bin/flexnndemo --<ncnn_param> [--<key value>...]
  --ncnn_param the path of ncnn.param file
  --ncnn_bin the path of ncnn.bin file
  --flexnn_param the path to save flexnn.param file
  --flexnn_bin the path to save flexnn.bin file
  --input_shape the input shape of the model, e.g., [1,3,32,32]
  --result_path the path to save profile and schedule result files
  --data_path the path to load dataset data file (if the data path is none, MASI will use random data to infer)
  --conv_sz the size limitation of convolutional layers
  --fc_sz the size limitation of fully-connected layers
  --memory_budgets the size of each runtime memory limitations
  --loop_num the number of loops to run inference for each config
  --log_num the number of loop number to log
  --mode the mode to run flexnndemo, including:
    - normal: run inference after layer slicing, memory profiling, and memory scheduling
    - ncnn_default: run inference by ncnn default method
    - ncnn_direct_conv: run inference directly 
    - ncnn_ondemand: run inerence on-demand 
  --timestep timestep number for each inference
  --record the record level, including:
    - 0: no record
    - 1: record memory and energy usage info into memory_energy file, record input inference info inference file
    - 2: record event info into memory_energy file
  --seenn_threshold the threshold parameter to exit early
  --waittime the waiting duration
  --sampletime the time between two sample node
  --jump the jump level will be use in the ablation experiment:
    - 0: no jump
    - 1: jump preloading part
    - 2: jump memory scheduling part

Example: ./flexnndemo --ncnn_param /home/user/models/ncnn/sewresnet18.ncnn.param --ncnn_bin /home/user/models/ncnn/sewresnet18.ncnn.bin --flexnn_param /home/user/models/flexnn/sewresnet18.flexnn.param --flexnn_bin /home/user/models/flexnn/sewresnet18.flexnn.bin --result_path /home/user/results/ --conv_sz 20 --fc_sz 10 --loop_num 32 --log_num 1 --mode normal --seenn_threshold 0.3 --input_shape [1,1,128,128] --timestep 10 --waittime 300000000 --sampletime 50000 --record 2 --memory_budgets 24,134,68,180
```

You can modify arguments in this command to try other configurations.

## Evaluation

### Experimental Setup

We hereby provide some implementation-related experimental details. You can refer to the overall settings in [conduct the experiments](#experiment-workflow).

#### Evaluated Models

We use the following models during evaluation. Since our approach doesn't change the model's output, and the evaluation is focused on the system performance, we use pre-trained or random weights, and random inputs for all models.

| Full Name          | Alias in Files |
| ------------------ | -------------- |
| SewResnet18        | sewresnet18    |
| SewResnet34        | sewresnet34    |
| Spikformer256      | spikformer256  |
| Spikformer384      | spikformer384  |
| SpikingVGG9        | spikingvgg9    |
| SpikingVGG16       | spikingvgg16   |

#### Baselines

All the baselines are based on NCNN:

- **"NCNN-Default"**: NCNN with default settings.
- **"NCNN-Direct"**: NCNN without memory-consuming optimizations.
- **"On-Demand"**: on-demand layer streaming implemented atop NCNN.

#### Key Metrics

- **Inference latency**. Measured in the program itself. Unless specified otherwise, we run 32 loops of inference for one of four configs, and calculate the latency for each config.
- **Memory Usage**. We measure memory usage through reading the VmRSS values from proc file, so the results are bigger from the memory budgets you set, but the trend is in line with expectations.
- **Energy Consumption**. Indirectly calculated by the real-time voltage and current, which are obtained through the Linux hwmon hardware monitor.

#### Number of cores

Regarding the ARM big.LITTLE technology, we use all the big cores for computing and one little core for loading (or middle core, regarding the device specification) when running inference with MASI, and we use only the same number of big cores in baselines.

We don't add a little/middle core for computing in baselines because results have shown that the little/middle core will become the bottleneck and further increase inference latency in baselines.

When running on devices that don't have a little/middle core, we use a big core for loading instead.

### Experiment Workflow

Before running the experiments, double-check that:

- You have **root access** to the device for specific experiment (Root access is a must for measure memory usage and energy consumption).
  - Linux: `sudo -i`
  - Android: `su` 
- You have turned off any power-saving settings.
  - See settings if you use a smartphone.
- For energy evaluation (wireless devices only, smartphones for example):
  - The device is unplugged (use ADB WiFi connection).
  - The screen is turned off.
  - There are no other background processes.

#### latency-memory-tradeoff comparison

The result of this experiment shows in Figure 5, the time spent on it depends on the device, model, runtime parameters and, most important, the memory budget you set.

```bash
./bin/flexnndemo --ncnn_param your_ncnn_param_file --ncnn_bin your_ncnn_bin_file --flexnn_param your_flexnn_param_save_path --flexnn_bin your_flexnn_bin_save_path --result_path your_profile_file_save_path --conv_sz conv_limit --fc_sz fc_limit --loop_num loop_number --mode normal/ncnn_default/ncnn_direct_conv/ncnn_ondemand --input_shape input_shape --timestep T --memory_budgets memory_budget_size [--sampletime sample_time_interval --record record_level --seenn_threshold seenn_threshold_parameter]
```

If you want to testify the result of FlexNN or MASI, the mode should be set as 'normal', in addition, you need set the seenn_threshold to open the early exit mode so that you can testify the result of MASI.

#### power monitoring

The result of this experiment shows in Figure 7.

```bash
./bin/flexnndemo --ncnn_param your_ncnn_param_file --ncnn_bin your_ncnn_bin_file --flexnn_param your_flexnn_param_save_path --flexnn_bin your_flexnn_bin_save_path --result_path your_profile_file_save_path --conv_sz conv_limit --fc_sz fc_limit --loop_num loop_number --mode normal/ncnn_default --input_shape input_shape --timestep T --memory_budgets memory_budget_size --sampletime sample_time_interval --record 1/2 --seenn_threshold seenn_threshold_parameter
```

You can modify the sampletime to adjust the time interval for data sampling, we advise you to set it to around 50000. The record can be set as 1 if you want to check the pure information of memory and power usage, 
or set it as 2 if you want to check the additional event information.

#### Ablation Study

The result of this experiment shows in Figure 9.

```bash
./bin/flexnndemo --ncnn_param your_ncnn_param_file --ncnn_bin your_ncnn_bin_file --flexnn_param your_flexnn_param_save_path --flexnn_bin your_flexnn_bin_save_path --result_path your_profile_file_save_path --conv_sz conv_limit --fc_sz fc_limit --loop_num loop_number --mode normal --input_shape input_shape --timestep T --memory_budgets memory_budget_size [--jump jump_part --seenn_threshold seenn_threshold_parameter]
```

If you want to testify the result of MASI, please set mode as 'normal' and set the seenn_threshold to open the early_exit mode. If you want to testify the result of w/o preloading and w/o planning, please set jump as '1' and '2' desperately.
If you are not set the seenn_threshold, the result will be w/o early exit.

#### Real-time latency and memory usage monitoring

The result of this experiment shows in Figure 8.

```bash
./bin/flexnndemo --ncnn_param your_ncnn_param_file --ncnn_bin your_ncnn_bin_file --flexnn_param your_flexnn_param_save_path --flexnn_bin your_flexnn_bin_save_path --result_path your_profile_file_save_path --conv_sz conv_limit --fc_sz fc_limit --loop_num 32 --mode normal --input_shape input_shape --timestep T --memory_budgets memory_budget_size --seenn_threshold 0.3 --sampletime 50000 --record 1
```

You can check the change of memory usage and the inference latency in different memory budgets you set in parameter 'memory_budgets', e.g., `--memory_budgets 24,68,134,180`. The result will save in result_path/infer_time.txt.

## Limitations

There are some known issues and limitations with the current MASI implementation.

GitHub issues and emails are welcome if you find any other issues.
