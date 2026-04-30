# Weighted-FD

Official PyTorch implementation of **["Weighted Federated Distillation: A Knowledge-Quality-Aware, Teacher-Less Strategy"](https://www.sciencedirect.com/science/article/pii/S0167739X26001512)**.

This repository contains the framework and code to run experiments evaluating the Weighted-FD method alongside baseline federated distillation approaches.

## Overview

Federated Learning has emerged as a promising privacy-preserving collaborative learning paradigm, highly suitable for deploying Artificial Intelligence at the network's edge. However, the performance of FL systems is often undermined by the heterogeneity of edge devices and the non-Independent and Identically Distributed (non-IID) nature of data across them.

While Federated Distillation mitigates these issues by transferring knowledge across heterogeneous models, existing methods typically rely on uniform aggregation strategies that fail to consider the quality or reliability of client predictions.

To address this critical gap, we introduce **Weighted-FD**, a novel collaborative framework that incorporates a quality-aware aggregation strategy. By dynamically evaluating the quality of knowledge contributed by each client, Weighted-FD adjusts their influence on the global knowledge representation, prioritizing highly reliable predictions and ensuring a more accurate aggregation process without requiring a centralized teacher model.

### Key Features and Contributions

- **Quality-Aware Aggregation**: Introduces a lightweight weighting mechanism that estimates the reliability of client predictions (using the probability assigned to the ground-truth class) to compute high-quality global soft labels.
- **Robustness in Strong Non-IID Settings**: Consistently outperforms existing baselines under severe data heterogeneity. For instance, on the CIFAR-10 dataset, it achieves accuracy gains of up to 57.12% over FedMD and 49.34% over Selective-FD.
- **Edge-Computing Efficiency**: Operates without imposing additional computational overhead on peripheral clients and maintains extremely low memory consumption, making it ideal for resource-constrained edge environments.
- **Theoretical Foundation**: Provides rigorous mathematical analysis demonstrating the convergence behavior of the local distillation phase and formalizing the conditions under which quality-aware aggregation surpasses standard uniform averaging.

## Overall framework of Weighted-FD

![Schema](./assets/schema.png)

Federated Distillation consists of four iterative steps: (I) clients train personalized models on their local data; (II) they generate local soft labels for shared proxy samples and send these to the server; (III) the server aggregates the predictions into global soft labels; (IV) clients refine their models using KD with the global soft labels.
In Weighted-FD, the aggregation function is adapted based on the quality of knowledge each client provides for each proxy sample.

## Features

- **Algorithms Supported:**
  - `weighted-fd` (Ours)
  - `selective-fd`
  - `fed-md` (Baseline)
- **Datasets Supported:** `mnist`, `fashion-mnist`, `cifar10`
- **Data Partitions Supported:** `IID`, `WEAK_NON_IID`, `STRONG_NON_IID`

## Installation

The project requires Python 3.9+. You can install the package and its dependencies locally via `pip`:

```bash
git clone https://github.com/your-username/weighted-fd.git
cd weighted-fd
pip install -e .
```

This will install the `wfd` package from the `src` directory along with dependencies like `torch`, `torchvision`, `scikit-learn`, `einops`, `rich`, and `flwr-datasets`.

## Usage

Experiments can be launched using the `train.py` script. The script takes arguments to specify the dataset, the data distribution, the algorithm, and hyperparameters.

### Example Run

```bash
python train.py \
    --dataset cifar10 \
    --datasets_dir ./data \
    --dataset_type STRONG_NON_IID \
    --output_path ./results \
    --algorithm weighted-fd \
    --seed 42 \
    --n_clients 10 \
    --rounds 100
```

### Arguments

**Core Arguments:**

- `--dataset`: Dataset to use (`mnist`, `fashion-mnist`, `cifar10`).
- `--datasets_dir`: Directory where the processed dataset `.pt` files are stored.
- `--dataset_type`: How data is distributed among clients (`IID`, `WEAK_NON_IID`, `STRONG_NON_IID`).
- `--output_path`: Path to save the execution logs and output files.
- `--algorithm`: Algorithm to run (`fed-md`, `selective-fd`, `weighted-fd`).
- `--seed`: Random seed for reproducibility.

**Training Hyperparameters:**

- `--n_clients`: Number of federated clients (default: `10`).
- `--rounds`: Number of federated communication rounds (default: `100`).
- `--start_iters`: Number of initial local training iterations (default: `200`).
- `--local_iters`: Local iterations per round (default: `1`).
- `--distl_iters`: Distillation iterations per round (default: `10`).
- `--local_batchsize`: Batch size for local training (default: `64`).
- `--proxy_batchsize`: Batch size for distillation on the proxy dataset (default: `512`).
- `--lr`: Learning rate (default: `0.1`).
- `--proxy_fraction`: Fraction of proxy dataset to use (default: `1.0`).

**KuLSIF Parameters (for `selective-fd` only):**

- `--lamda`: (default: `0.0632`)
- `--gauss`: (default: `5`)
- `--tfrac`: (default: `0.05`)
- `--nunif`: (default: `250`)
- `--mnval`: Maximum validation samples to limit memory usage.

## Repository Structure

- `src/wfd/`: Core package containing the implementations of the clients, servers, neural networks, estimators, and datasets.
- `train.py`: The entry point script for running federated distillation experiments.
- `notebooks/`: Jupyter notebooks used for visualizing data distributions, analyzing results (accuracy, proxy fractions, client scaling), and exploring the `flwr-datasets`.
- `pyproject.toml`: Python project configuration containing the build system and dependencies.
- `memory-profile-analisys.sh` & `nvidea-setup.sh`: Utility scripts for profiling and environment setup.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{DELLACQUA2026108517,
  title = {Weighted Federated Distillation: A knowledge-quality-aware, teacher-less strategy},
  journal = {Future Generation Computer Systems},
  volume = {183},
  pages = {108517},
  year = {2026},
  issn = {0167-739X},
  doi = {https://doi.org/10.1016/j.future.2026.108517},
  url = {https://www.sciencedirect.com/science/article/pii/S0167739X26001512},
author = {Pierluigi Dell’Acqua and Lemuel Puglisi and Francesco La Rosa and Lorenzo Carnevale and Daniele Ravì and Massimo Villari},
keywords = {Knowledge distillation, Federated learning, Edge computing}
}
```
