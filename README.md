# XGBoost for Spatial Point Processes (XGBoostPP)

A research codebase implementing **XGBoost-based intensity estimation** for spatial point processes (SPP) and linear point processes (LPP). This project covers both simulation studies and real-world applications using custom Poisson and logistic log-likelihood objectives within the XGBoost / LightGBM framework.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Simulation Study](#simulation-study)
  - [SPP Simulation (Planar)](#spp-simulation-planar)
  - [LPP Simulation (Linear Network)](#lpp-simulation-linear-network)
- [Real-World Applications](#real-world-applications)
  - [Available Datasets](#available-datasets)
  - [Running the Application Pipeline](#running-the-application-pipeline)
- [Loss Functions](#loss-functions)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Outputs](#outputs)
- [Data Description](#data-description)
- [Citation](#citation)

---

## Overview

This project estimates the **intensity function** λ(u) of a spatial point process using gradient boosting (XGBoost / LightGBM) with custom objectives derived from:

- **Poisson log-likelihood** — standard Poisson process likelihood
- **Weighted Poisson log-likelihood** — accounts for clustering via an F_prime correction
- **Logistic log-likelihood** — Baddeley-style logistic regression for point processes
- **Weighted logistic log-likelihood** — weighted variant of the above

The model is trained on a **case-control** dataset: observed event points (label = +1) mixed with randomly generated dummy/quadrature points (label = −1). Intensity predictions at dummy locations are used to reconstruct the continuous intensity surface.

Key features:
- Works on **planar windows** (SPP) and **linear networks** (LPP)
- Supports **fixed parameters** and **Bayesian hyperparameter tuning** via Optuna
- Seamlessly integrates R (`spatstat`) with Python (`xgboost`, `lightgbm`) via `reticulate`
- Includes **MISE** and **log-likelihood** evaluation metrics

---

## Repository Structure

```
xgboost-spatial-point-process/
│
├── python/                          # Core Python modules (XGBoost/LightGBM objectives)
│   ├── xgbpp.py                     # XGBoostPP: custom objectives + Optuna tuner (simulation)
│   ├── lgbpp.py                     # LightGBMPP: custom objectives + Optuna tuner (simulation)
│   └── xgbpp_application.py        # XGBoostPP objectives adapted for application pipeline
│
├── R/
│   ├── simulation/                  # Simulation study (planar SPP)
│   │   ├── simulate_processes.R     # Simulate Poisson, Thomas, LGCP, Strauss processes
│   │   ├── wrapper_spp.R            # run_analysis(): train + evaluate + visualize per run
│   │   └── run_simulation_spp.R    # Main entry point for SPP simulation loop
│   │
│   ├── lpp/                         # Linear Point Process (LPP) simulation
│   │   ├── simulate_lpp.R           # Simulate LPP on a linear network (Nganjuk data)
│   │   ├── wrapper_lpp.R            # run_analysis_LPP(): LPP-specific train + evaluate
│   │   ├── run_simulation_lpp.R    # Main entry point for LPP simulation loop
│   │   └── linquad.R                # Linear quadrature helpers
│   │
│   └── application/                 # Real-world application pipeline
│       ├── application.R            # Main universal pipeline (BCI, Crime, Accident)
│       ├── application_raw.R        # Raw/exploratory version of the application script
│       ├── prepare_bei_ryno_data.R  # Data preparation for BEI (bei / ryno datasets)
│       └── prepare_crime_data.R     # Data preparation for Crime dataset
│
├── data/                            # Input datasets
│   ├── bci.covars.Rda               # BCI covariate images (elevation, slope, etc.)
│   ├── bci.tree1.rdata              # BCI tree species point pattern
│   ├── bei/                         # BEI tropical forest data
│   ├── ryno/                        # BCI Ryno species data
│   ├── crime/                       # Crime data (Kennedy dataset)
│   └── accident/                    # Traffic accident data (Nganjuk, Java)
│
├── output/                          # Auto-generated outputs (gitignored)
│   ├── figures/                     # Intensity maps, importance plots
│   ├── tables/                      # CSV results, grid search logs
│   └── models/                      # Saved model objects (.rds)
│
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

---

## Requirements

### Python (≥ 3.9)

```
xgboost>=1.7
lightgbm>=3.3
optuna>=3.0
numpy>=1.23
pandas>=1.5
tqdm
```

### R (≥ 4.1)

```r
spatstat         # Core spatial statistics
spatstat.geom
spatstat.explore
spatstat.linnet
spatstat.random
reticulate       # R–Python bridge
xgboost          # R XGBoost (for DMatrix usage)
lightgbm
ggplot2
viridis
RColorBrewer
openxlsx
tictoc
dplyr
sf
sp
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/xgboost-spatial-point-process.git
cd xgboost-spatial-point-process
```

### 2. Set up a Python environment (conda recommended)

```bash
conda create -n xgb-env python=3.10
conda activate xgb-env
pip install -r requirements.txt
```

### 3. Install R packages

```r
install.packages(c(
  "reticulate", "spatstat", "spatstat.geom", "spatstat.explore",
  "spatstat.linnet", "spatstat.random", "xgboost", "lightgbm",
  "ggplot2", "viridis", "RColorBrewer", "openxlsx", "tictoc",
  "dplyr", "sf", "sp"
))
```

### 4. Link R to your conda environment

In any R script, set this at the top before loading `reticulate`:

```r
library(reticulate)
use_condaenv("xgb-env", required = TRUE)
```

---

## Quick Start

### Run a minimal SPP simulation (Poisson process, XGBoost, fixed params)

```r
# In R
library(reticulate)
use_condaenv("xgb-env", required = TRUE)

source_python("python/xgbpp.py")
source("R/simulation/simulate_processes.R")
source("R/simulation/wrapper_spp.R")

xgb <- import("xgboost")
pd  <- import("pandas")

# Load BCI covariates
load("data/bci.covars.Rda")
data(bei)

# Simulate a single Poisson process and run XGBoost
sim <- simulate_poisson_process(
  covariate_names = c("elev", "grad"),
  intercept = 0,
  coefficients = c(1, -1),
  bci_covars = bci.covars,
  bei_window = Window(bci.covars[[1]]),
  scale_factor = 500,
  n_points = 2000
)

train_xgbpp_fixed <- function(X, y, vol, loss, F_prime, base_params) {
  final_params <- c(base_params, list(eta = 0.001, alpha = 0, lambda = 0, max_depth = 6L))
  dtrain <- xgb$DMatrix(data = as.matrix(X), label = pd$Series(y))
  xgbpp_py(dtrain = dtrain, vol = pd$Series(vol), params = final_params,
           loss = loss, F_prime = F_prime, num_boost_round = 5000L,
           evals = list(list(dtrain, "train")), early_stopping_rounds = 50L,
           verbose_eval = FALSE)
}
```

---

## Simulation Study

### SPP Simulation (Planar)

The simulation study evaluates XGBoostPP on three types of spatial point processes over a planar window (BCI Forest Plot, 1000×500m rescaled).

**Entry point:** `R/simulation/run_simulation_spp.R`

#### Supported process types

| Process | Description |
|---|---|
| `Poisson` | Inhomogeneous Poisson process with covariate-driven log-intensity |
| `Thomas` | Neyman–Scott cluster process |
| `LGCP` | Log-Gaussian Cox Process |
| `Strauss` | Inhibition process via Metropolis-Hastings |

#### How to run

1. Open `R/simulation/run_simulation_spp.R`
2. Update the conda path:
   ```r
   Sys.setenv(RETICULATE_PYTHON = "/path/to/your/anaconda3/envs/xgb-env/python.exe")
   use_condaenv("xgb-env", conda = "/path/to/conda.exe", required = TRUE)
   ```
3. Set global parameters:
   ```r
   N_SIMULATIONS <- 50    # Number of Monte Carlo runs
   num_sim_points <- 2000 # Number of points per simulation
   scale_factor <- 500
   ```
4. Run the script. Results are saved per run to `output/`.

#### Key functions

```r
# Simulate a Poisson process
simulate_poisson_process(covariate_names, intercept, coefficients,
                         bci_covars, bei_window, scale_factor, n_points)

# Simulate a Thomas (cluster) process
simulate_thomas_process(covariate_names, intercept, coefficients,
                        bci_covars, bei_window, scale_factor, n_points)

# Simulate a Log-Gaussian Cox Process
simulate_lgcp_process(covariate_names, intercept, coefficients,
                      bci_covars, bei_window, scale_factor, n_points)

# Run analysis for one simulation replicate
run_analysis(model_type, loss_type, sim_data, sim_intensity, sim_points,
             base_output_dir, run_number, base_params,
             analysis_type = "fixed",  # or "tuned"
             scale_factor)
```

---

### LPP Simulation (Linear Network)

Extends the framework to **linear point processes** on a road/street network (Nganjuk regency, East Java).

**Entry point:** `R/lpp/run_simulation_lpp.R`

#### How to run

1. Open `R/lpp/run_simulation_lpp.R`
2. Update the conda and data paths
3. The network object is loaded from `data/accident/nganjuk_ln.rds`
4. Run the script

#### Key difference from SPP

- The observation window is a `linnet` object (linear network), not a polygon
- Volumes (`vol`) are computed using linear quadrature weights (`linquad.R`)
- MISE is computed over the **total network length** rather than area
- Intensity maps use `linim` objects from `spatstat.linnet`

---

## Real-World Applications

### Available Datasets

| Dataset | Type | Description |
|---|---|---|
| **BEI** (`data/bei/`) | SPP | *Beilschmiedia pendula* tropical tree positions, BCI forest plot |
| **Ryno** (`data/ryno/`) | SPP | *Rynostomus* tree species, BCI forest plot |
| **Crime** (`data/crime/`) | SPP | Urban crime events, Kennedy crime dataset |
| **Accident** (`data/accident/`) | LPP | Traffic accidents on the Nganjuk road network |

Each dataset folder contains:
- `*nonscale.csv` — raw coordinates + covariates (unscaled)
- `*loginonscale*.csv` — case-control dataset ready for logistic fitting

### Running the Application Pipeline

**Entry point:** `R/application/application.R`

#### Step 1 — Prepare the data (if needed)

```r
# For BEI / Ryno
source("R/application/prepare_bei_ryno_data.R")

# For Crime
source("R/application/prepare_crime_data.R")
```

#### Step 2 — Configure the pipeline

Open `R/application/application.R` and set:

```r
# Choose dataset: "bei", "ryno", "crime", or "accident"
dataset_name <- "crime"

# Choose loss function: "poisson", "logistic",
#                       "weighted_poisson", "weighted_logistic"
loss_type <- "poisson"
```

#### Step 3 — Run

```r
source("R/application/application.R")
```

The pipeline will:
1. Load and configure data for the chosen dataset
2. Build the case-control training matrix
3. Train XGBoostPP with the chosen loss
4. Tune hyperparameters (optional, set `analysis_type = "tuned"`)
5. Predict the intensity surface over dummy points
6. Produce intensity maps, feature importance plots, and summary tables
7. Save everything to `output/`

---

## Loss Functions

All objectives are implemented in `python/xgbpp.py`. The model predicts `f(u)` (log-intensity on the log scale), so the estimated intensity is `λ̂(u) = exp(f(u))`.

| Loss | Function | Use case |
|---|---|---|
| `"poisson"` | Poisson log-likelihood with case-control normalisation | Standard IPP |
| `"weighted_poisson"` | Poisson + F_prime cluster correction | Clustered patterns |
| `"logistic"` | Baddeley logistic log-likelihood | Alternative to Poisson |
| `"weighted_logistic"` | Weighted logistic with cluster correction | Clustered + logistic |

The `F_prime` value is estimated from the inhomogeneous K-function:

```r
k_func <- Kinhom(points, lambda = intensity, correction = "translation")
r_med  <- median(nndist(points))
F_prime <- with(k_func[which.min(abs(k_func$r - r_med)), ], trans - theo)
```

---

## Hyperparameter Tuning

XGBoostPP supports two modes:

### Fixed parameters (`analysis_type = "fixed"`)

Uses sensible defaults — fast, good for exploring data:

```python
params = {
    "eta": 0.001,
    "alpha": 0,
    "lambda": 0,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 1/3,
    "tree_method": "hist"
}
```

### Bayesian tuning (`analysis_type = "tuned"`)

Uses **Optuna** to search over `eta`, `alpha`, `lambda`, and `max_depth` with up to 5000 boosting rounds and early stopping:

```r
# In R (via reticulate)
tuning_results <- tune_xgbpp(
  X_df = X,
  y_series = y,
  vol_series = vol,
  loss = "poisson",
  F_prime = 0,
  n_trials = 100L,
  constrain_events = TRUE,    # penalise if predicted N ≠ observed N
  constraint_strength = 1.0
)
```

All trial results are saved to `xgboost/<loss>/optuna_search_results.csv`.

---

## Outputs

After running either simulation or application scripts, results are saved under `output/`:

```
output/
├── figures/
│   ├── <dataset>_intensity.png          # Predicted intensity heatmap
│   ├── <dataset>_importance.png         # Feature importance bar chart
│   └── simulated_intensity_run_*.png    # Per-run simulation plots
├── tables/
│   ├── <dataset>_grid_search.csv        # Tuning trial results
│   └── <dataset>_feature_importance.csv
└── models/
    └── <dataset>_xgbpp_model.rds        # Saved final model
```

Simulation runs also produce per-run Excel summaries (`summary_results.xlsx`) containing:
- True log-likelihood
- Model log-likelihood (Poisson and logistic)
- MISE (Mean Integrated Squared Error)
- Predicted vs. true event count
- Computation time

---

## Data Description

### BCI Forest Plot (`data/bci.covars.Rda`, `bci.tree1.rdata`)

Covariate rasters for the 50ha Barro Colorado Island forest plot:
- `elev` — elevation (m)
- `grad` — slope gradient
- `aspect`, `convex`, `beers` — terrain features

### BEI / Ryno (`data/bei/`, `data/ryno/`)

Case-control datasets derived from `spatstat::bei` and BCI tree surveys. Columns: `x`, `y`, `label` (1 = event, −1 = dummy), `vol` (quadrature weight), covariate columns.

### Crime (`data/crime/`)

Kennedy crime dataset for urban spatial analysis. Pre-processed into scaled and unscaled versions.

### Accident (`data/accident/`)

Traffic accident locations on the Nganjuk, East Java road network (`nganjuk_ln.rds`). Linear network stored as a `linnet` object.

---

## Citation

If you use this code in your research, please cite:

```
[Your paper citation here]
```

---

## Notes

- The `output/` folder is gitignored. All result files are generated locally.
- The `{python,R/...}` folders at the repo root are an artefact of shell brace expansion and can be safely deleted.
- Indonesian comments in the source code (`#` lines in Bahasa Indonesia) are original annotations from the development phase and reflect the authors' working language.
