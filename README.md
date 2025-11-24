# Phase-Transitional Scaling (PTS)

This repository contains the code for generating the figures from the Phase-Transitional Scaling paper at NeurIPS 2025 LLM-evaluation.

##  Overview

Phase-Transitional Scaling (PTS) provides a novel framework for understanding capability emergence in large language models through sigmoid-shaped transitions. This repository generates comprehensive validation figures demonstrating:

- **Universal Curve Collapse** - Sigmoid transitions across different architectures and tasks
- **Parameter Disentanglement** - Independent control of threshold (T) and sharpness (γ) parameters  
- **Predictive Validation** - Superior forecasting compared to power-law approaches
- **Theoretical Validation** - Mean-field theory, percolation thresholds, and finite-size scaling

## Repository Structure

```
LLM-eval/
├── .venv/                          # Python virtual environment
├── experiments/
│   ├── generate_figures.py         # Main figure generation (PTSVisualizer class)
│   └── pts_validation.py          # Experimental validation suite
├── figures/                        # Generated publication figures
│   ├── comprehensive_validation.png/pdf
│   ├── theoretical_validation.png/pdf
│   └── prior_work_comparison.png/pdf
├── results/
│   └── complete_pts_validation.json # Experimental data for figures
├── run_experiments.py             # Main entry point
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## Quick Start

### 1. Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
# Clone/navigate to the repository
cd path\to\LLM-eval

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
# Clone/navigate to the repository
cd path/to/LLM-eval

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Figures

**Run Complete Validation and Figure Generation:**
```bash
# With virtual environment activated
python run_experiments.py
```

**Generate Only Figures (using existing data):**
```bash
# With virtual environment activated
python -c "from experiments.generate_figures import generate_all_figures; generate_all_figures()"
```

### 3. View Results

Generated figures will be saved in the `figures/` directory:
- `comprehensive_validation.png/pdf` - Main validation results
- `theoretical_validation.png/pdf` - Theoretical mechanism validation  
- `prior_work_comparison.png/pdf` - Comparison with existing scaling laws

##  Figures

### 1. Comprehensive Validation Figure
**Panels:**
- **A.** Universal Curve Collapse - Sigmoid transitions across architectures/tasks
- **B.** Data Complexity Controls T - Threshold parameter T varies with task complexity
- **C.** Training Dynamics Control γ - Sharpness parameter γ controlled by training
- **D.** Predictive Performance - PTS outperforms power-law baselines

### 2. Theoretical Validation Figure  
**Panels:**
- **A.** Mean-Field Finite-Size Scaling - Transition width ∝ N^(-1/2)
- **B.** Percolation Threshold Validation - Critical density predictions
- **C.** Absorbing-State Dynamics - Exponential transition time distributions
- **D.** Effective Free Energy Landscape - Energy barrier evolution

### 3. Prior Work Comparison Figure
**Panels:**
- **A.** Scaling Law Comparison - PTS vs power-law approaches
- **B.** Emergence Prediction Accuracy - Cross-validation performance

## Dependencies

The codebase requires the following Python packages (automatically installed via `requirements.txt`):

- **numpy** (≥1.21.0) - Numerical computing
- **scipy** (≥1.7.0) - Scientific computing and optimization
- **matplotlib** (≥3.4.0) - Plotting and visualization
- **seaborn** (≥0.11.0) - Statistical data visualization
- **pandas** (≥1.3.0) - Data manipulation and analysis
- **torch** (≥1.9.0) - Deep learning framework
- **scikit-learn** (≥1.0.0) - Machine learning utilities

##  Key Components

### Core Classes and Functions

**`PTSVisualizer`** (in `experiments/generate_figures.py`):
- Main class for generating publication figures
- Methods for each figure type (`create_comprehensive_figure()`, etc.)
- Sigmoid fitting and statistical analysis utilities

**`CapabilityExperiments`** (in `experiments/pts_validation.py`):
- Experimental validation suite
- Synthetic data generation with ground truth
- Multi-architecture capability testing

**Key Functions:**
- `generate_all_figures()` - Generate complete figure suite
- `run_complete_validation()` - Execute experimental validation
- `sigmoid(x, A, gamma, T)` - Core sigmoid function

## 🔬 Experimental Validation

The validation suite includes:

1. **Synthetic Validation** - Ground truth sigmoid data with known parameters
2. **Modular Arithmetic** - GPT-2 models on arithmetic tasks of varying complexity  
3. **Compositional Generalization** - SCAN-style compositional reasoning tasks
4. **Training Dynamics** - Analysis of parameter evolution during training
5. **Universal Collapse** - Cross-architecture and cross-task validation

## Usage Examples

### Generate Specific Figures

```python
from experiments.generate_figures import PTSVisualizer

# Load experimental results
visualizer = PTSVisualizer("results/complete_pts_validation.json")

# Generate individual figures
visualizer.create_comprehensive_figure()
visualizer.create_theoretical_validation_figure()
visualizer.create_comparison_with_prior_work_figure()
```

### Run Custom Experiments

```python
from experiments.pts_validation import CapabilityExperiments

# Initialize experiments
experiments = CapabilityExperiments()

# Run specific validation
synthetic_results = experiments.generate_synthetic_validation()
arithmetic_results = experiments.run_modular_arithmetic_experiments()
```

## License & Citation

This code is part of the Phase-Transitional Scaling research project. If you use this code or the generated figures, please cite the original paper.

```
@inproceedings{
cherukuri2025phasetransitional,
title={Phase-Transitional Scaling},
author={Kalyan Cherukuri and Aarav Lala},
booktitle={NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling},
year={2025},
url={https://openreview.net/forum?id=y7mBTQM7c4}
} 
```

##  Contributing

This is a research codebase focused on reproducible figure generation. The code has been streamlined for clarity and ease of use while maintaining all essential functionality for experimental validation and visualization.

---

**Repository Status**: Maintained  
**Last Updated**: October 2025  
**Python Version**: Tested with Python 3.8+
