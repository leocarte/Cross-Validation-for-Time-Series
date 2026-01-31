# Cross-Validation for Time Series with Dependent Errors

**Authors:** Leonardo Cartesegna, Filippo Reina, Dario Liuzzo   
**Course:** MATH-517 Statistical Computation and Visualization  
**Institution:** EPFL

## Abstract

Nonparametric regression techniques rely heavily on the optimal selection of smoothing parameters. Standard cross-validation (CV) methods, such as Leave-One-Out (LOOCV) or k-fold CV, operate under the assumption of independent errors. When applied to time series exhibiting short-range dependence, these "naive" methods underestimate prediction error, leading to systematically over-fitted models. This project investigates the theoretical failure of standard CV in correlated settings and evaluates dependence-aware alternatives: Block CV, Buffered (Leave-(2l+1)-out) CV, and Walk-Forward Validation. We further elaborate on efficient computational strategies for Neighborhood CV (NCV) in penalized spline settings. We compare these methods using Penalized Splines, Kernel Regression, and Gradient Boosted Trees (XGBoost) across synthetic autoregressive processes and real-world financial and environmental datasets. Our findings provide practical guidelines for model selection in the presence of temporal autocorrelation.

## Research Question

This project studies why naive (randomized) CV can fail under temporal dependence, and evaluates CV schemes designed to reduce leakage. Specifically, using synthetic data with autocorrelated errors, we aim to study:

1. The size of the bias of naive CV as an estimator of test MSE under different dependence strengths and signal complexities
2. Which dependence-aware CV schemes have the highest stability and reliable hyperparameter selection
3. How these conclusions vary across regression models (kernel regression, penalized splines, and gradient-boosted trees)

We aim to provide insights on the cost-performance tradeoff of these models, to assess whether the computational cost of each CV scheme results in better performance. While LOOCV is statistically appealing for dependent data, it can be computationally impossible to implement for large datasets. Recent developments show how a broad class of neighborhood CV criteria for quadratically penalized models can be computed at a cost comparable to a single model fit, making them usable in practice.

## Repository Structure

```
.
├── README.md                          # This file
├── Code/                              # Main code directory
│   ├── utils/                         # Core utilities and implementations
│   │   ├── models/                    # Regression model implementations
│   │   │   ├── base.py                # Base regressor interface
│   │   │   ├── kernel_regression.py   # Nadaraya-Watson kernel estimator
│   │   │   ├── penalized_spline.py    # B-spline with penalty
│   │   │   └── xgb_regression.py      # XGBoost wrapper
│   │   ├── cv.py                      # Cross-validation schemes
│   │   ├── core.py                    # Random number generation utilities
│   │   ├── errors.py                  # Error model specifications (AR, MA, etc.)
│   │   ├── signals.py                 # Signal generation (smooth trend, local bump, etc.)
│   │   ├── synthetic_series.py        # Synthetic time series generation
│   │   ├── experiment_helper.py       # Experiment orchestration
│   │   ├── search.py                  # Hyperparameter search via CV
│   │   ├── optimized_model.py         # Efficient NCV implementation for splines
│   │   ├── evaluation.py              # Model evaluation metrics
│   │   ├── plotting.py                # Visualization utilities
│   │   ├── plot_signals.py            # Signal plotting utilities
│   │   └── acf.py                     # Autocorrelation function analysis
│   ├── smooth_trend_runs.ipynb        # Experiments: smooth trend signals
│   ├── local_bump_runs.ipynb          # Experiments: local bump signals
│   ├── non_linear_runs.ipynb          # Experiments: non-linear signals
│   ├── cv_schemes_on_real_data.ipynb  # Real-world dataset experiments
│   ├── time_comparison.ipynb          # Computational cost comparisons
│   ├── heatmap_running_times.ipynb    # Running time analysis
│   ├── additional_plots.ipynb         # Supplementary visualizations
│   ├── figures/                       # Generated plots and figures
│   ├── real_datasets/                 # Real-world data and analysis
│   │   ├── Air Quality/               # Air Quality dataset
│   │   ├── CNN-based stock market prediction/  # Stock market indices
│   │   └── real_data_diagnostics.ipynb         # Real datasets diagnostics
│   └── runs/                          # Simulation results (CSV files)
│       ├── air_quality_run/
│       ├── sp500_run/
│       ├── smooth_trend_*/
│       ├── local_bump_*/
│       └── non_linear_*/
├── report/                            # LaTeX source for final report
│   ├── report.tex                     # Main report file
│   ├── report.pdf                     # Compiled PDF
│   ├── references.bib                 # Bibliography
│   ├── template.sty                   # Custom style file
│   ├── Sections/                      # Report sections
│   │   ├── Intro.tex
│   │   ├── Theory.tex
│   │   ├── Methods.tex
│   │   ├── Results.tex
│   │   └── ...
│   └── images/                        # Figures included in report
├── papers/                            # Reference papers
└── Writeup/                           # Initial project structure
```

## Installation and Dependencies

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Required Packages

The project requires the following Python packages:

```bash
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd project-k-fold-quants
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install numpy pandas scipy scikit-learn xgboost matplotlib seaborn jupyter
```


## Reproducing the Results

### Running Simulation Experiments

The main simulation studies are organized in separate notebooks:

1. **Smooth Trend Signals**:
   ```bash
   jupyter notebook Code/smooth_trend_runs.ipynb
   ```
   Experiments with smooth trend signals under AR(1), ARIMA, MA(5), and seasonal error structures.

2. **Local Bump Signals**:
   ```bash
   jupyter notebook Code/local_bump_runs.ipynb
   ```
   Experiments with localized bump signals to test CV robustness to sharp features.

3. **Non-linear Signals**:
   ```bash
   jupyter notebook Code/non_linear_runs.ipynb
   ```
   Experiments with complex non-linear signal patterns.

### Analyzing Real-World Data

Real-world dataset experiments are available in:
```bash
jupyter notebook Code/cv_schemes_on_real_data.ipynb
```

This notebook analyzes:
- **Air Quality Data**: UCI Air Quality dataset with temporal correlation
- **Financial Time Series**: S&P 500, NASDAQ, Dow Jones Industrial, NYSE, and Russell indices

### Computational Cost Analysis

To reproduce the running time comparisons:
```bash
jupyter notebook Code/time_comparison.ipynb
jupyter notebook Code/heatmap_running_times.ipynb
```

### Generating Plots

Additional visualizations and diagnostic plots can be generated using:
```bash
jupyter notebook Code/additional_plots.ipynb
jupyter notebook Code/real_datasets/real_data_diagnostics.ipynb
```

### Compiling the Report

The final report is written in LaTeX. To compile:

```bash
cd report
pdflatex report.tex
biber report
pdflatex report.tex
pdflatex report.tex
```


## Key Components

### Cross-Validation Schemes

The following CV schemes are implemented in `Code/utils/cv.py`:

- **Naive CV**: Standard k-fold cross-validation (assumes independence)
- **Block CV**: Contiguous blocks to respect temporal structure
- **Buffered CV (Leave-(2l+1)-out)**: Excludes neighbors around validation point
- **Walk-Forward Validation**: Expanding window for time series forecasting
- **Neighborhood CV (NCV)**: Efficient implementation for penalized splines

### Regression Models

Three regression models are compared (`Code/utils/models/`):

1. **Kernel Regression** (Nadaraya-Watson): Non-parametric local averaging with Gaussian kernel
2. **Penalized Splines**: B-spline basis with roughness penalty (second-order differences)
3. **Gradient Boosted Trees** (XGBoost): Ensemble learning method

### Error Structures

Synthetic time series are generated with different error models (`Code/utils/errors.py`):

- **AR(1)**: Autoregressive process with lag 1
- **MA(5)**: Moving average process with window 5
- **ARIMA(1,1,1)**: Combined autoregressive and moving average
- **Seasonal**: Seasonal component with specified period


## License

This project is part of academic coursework at EPFL for the MATH-517 course.
