# Weighted-FD

Official PyTorch implementation of **"Weighted Federated Distillation: A Knowledge-Quality-Aware, Teacher-Less Strategy"**.

This repository contains the framework and code to run experiments evaluating the Weighted-FD method alongside baseline federated distillation approaches.

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
