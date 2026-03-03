# Emergent collective intelligence via multi-agent reinforcement learning in particle swarm optimization

MARLPSO is a hybrid optimization framework that combines Multi-Agent Reinforcement Learning (MARL) with Particle Swarm Optimization (PSO). It uses RL agents to dynamically control perturbation mechanisms, helping particles escape local optima and improve convergence on complex, high-dimensional optimization problems.

This repository provides the core environment, CEC benchmark function interfaces, and training scripts to reproduce MARLPSO.

## Features

- **Multi-Agent Environment**: A specialized RL environment where each PSO particle is an agent.
- **Intelligent Perturbations**: State-based perturbation mechanisms (Random Jump, Lévy Flight, Opposition-based exploration, Dimension Shuffle) controlled by a PPO agent.
- **CEC Benchmark Support**: Full integration with `opfunu` to support CEC2013, 2014, 2017, and other benchmark suites.
- **Ray RLlib Integration**: Optimized for training using the Ray RLlib library.

## Project Structure

```text
MARLPSO/
├── marlpso/                # Core package
│   ├── env.py              # MARLPSO Multi-Agent Environment
│   └── functions.py        # CEC Function wrappers (using opfunun)
├── train.py                # Main training script
├── requirements.txt        # Dependency list
└── test_script.py          # Basic functionality test
```

## Installation

It is recommended to use a Conda environment. The code is tested with the following key dependencies:

- Python 3.12+
- Ray [RLlib] 2.53.0
- Gymnasium 1.1.1
- PyTorch 2.9.1
- Opfunu 1.0.1 (for CEC functions)
- setuptools<71.0.0

To install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Verify Installation
Run the simple test script to ensure the environment works:

```bash
python test_script.py
```

### 2. Training
To train MARLPSO on a specific CEC function (e.g., CEC2017 F1):

```bash
python train.py --function cec2017_f1 --dim 10 --particles 20 --train_iterations 100
```

### Configuration Options:
- `--function`: CEC function name (format: `cec{year}_f{id}`).
- `--dim`: Problem dimension.
- `--particles`: Number of particles (agents) in the swarm.
- `--train_iterations`: Number of training episodes.
- `--perturbation_mode`: `scheme3` (RL-controlled) or `pure_pso` (Baseline).

## Data Logging

By default, the training process logs the best fitness progress of each episode into a JSON file (default: `default.json`). You can customize this file path in the environment configuration.

### Log File Format (`default.json`)
The log consists of a list of episode data objects with the following fields:

- `episode`: The sequence number of the training episode.
- `function`: Name of the optimization function used.
- `dim`: Dimension of the optimization problem.
- `perturbation_mode`: The perturbation strategy used (`scheme3` or `pure_pso`).
- `final_fitness`: The ultimate best fitness value achieved in that episode.
- `iterations`: Amount of iterations performed in the episode.
- `fitness_history`: A list containing the global best fitness at each iteration step.
- `timestamp`: ISO-formatted time of the log entry.
- `theoretical_optimum`: The target global minimum for the benchmark function.

## Usage as a Library

You can import the MARLPSO environment for your own RL experiments:

```python
from marlpso.env import MARLPSOEnvironment
from marlpso.functions import get_cec_function

# Initialize a function
func = get_cec_function(2017, 1, dim=30)

# Create environment
env = MARLPSOEnvironment({
    "dim": 30,
    "fitness_func": func.evaluate,
    "bounds": func.bounds,
    "num_particles": 20
})
```

## Citation
