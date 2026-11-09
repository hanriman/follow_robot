# AI Research Directory Template & Best Practices Guide

A production-grade, reproducible directory structure designed for AI, Machine Learning, and Computational Biology research projects.

---

## 1. Project Directory Layout

```text
ai-research-project/
├── README.md                  # Project overview, key findings, setup instructions, & CLI quickstart
├── pyproject.toml             # Environment dependencies, build system, & tool configs (ruff, pytest)
├── uv.lock / environment.yml  # Exact pinned dependency lockfile
├── .python-version            # Python version specification (e.g., 3.12)
├── .gitignore                 # Exclude heavy binaries, checkpoints, local caches, and secret keys
├── .pre-commit-config.yaml    # Pre-commit hooks for formatting, linting, and secret detection
│
├── configs/                   # Experiment & model configurations (YAML / Hydra / Pydantic)
│   ├── base.yaml              # Default hyperparameter defaults
│   ├── model/                 # Model architecture configs
│   ├── dataset/               # Dataset & preprocessing configs
│   └── experiment/            # Named experiment specs (e.g., baseline.yaml, ablation_loss.yaml)
│
├── src/                       # Core python library package (reusable modules)
│   └── my_project/
│       ├── __init__.py
│       ├── config.py          # Centralized path & env variable resolution
│       ├── models/            # Model architectures & neural net blocks
│       │   ├── __init__.py
│       │   ├── encoder.py
│       │   └── predictor.py
│       ├── data/              # Data loaders, dataset classes, & preprocessing pipelines
│       │   ├── __init__.py
│       │   └── dataset.py
│       ├── losses/            # Custom loss functions & regularizers
│       │   ├── __init__.py
│       │   └── anti_collapse.py
│       ├── metrics/           # Custom evaluation probes & metrics
│       │   ├── __init__.py
│       │   └── downstream.py
│       └── utils/             # Logging, seeding, device handlers, & I/O helpers
│           ├── __init__.py
│           └── seed.py
│
├── scripts/                   # Executable CLI runners & benchmark entry points
│   ├── train.py               # Main training script
│   ├── evaluate.py            # Model evaluation & probing runner
│   ├── run_ablation.py        # Automated ablation sweep runner
│   └── generate_figures.py    # Generates publication-ready figures & tables
│
├── notebooks/                 # Exploratory data analysis & interactive visualizations
│   ├── 01_eda.ipynb           # Initial dataset exploration
│   ├── 02_latent_analysis.ipynb
│   └── .gitignore             # Excludes .ipynb_checkpoints
│
├── tests/                     # Automated test suite (pytest)
│   ├── outputs/               # Isolated test outputs & mock fixtures (never pollutes real outputs/)
│   ├── conftest.py            # Shared fixtures (dummy tensors, mini datasets)
│   ├── test_models.py         # Shape & gradient sanity checks
│   ├── test_losses.py         # Loss invariance & boundary tests
│   └── test_data.py           # Data loading & transformation tests
│
├── docs/                      # Research documentation & design specifications
│   ├── project_desc.md        # Research question, hypotheses, & execution roadmap
│   ├── data_dictionary.md     # Feature definitions & data lineage details
│   └── setup_guide.md         # Environment setup & compute hardware instructions
│
├── paper/                     # Manuscript source code & submission artifacts
│   ├── latex/                 # Tracked LaTeX source files (.tex, .bib, figures)
│   │   ├── main.tex
│   │   ├── references.bib
│   │   └── figures/           # Symlinked or copied publication figures
│   └── submission_plan.md    # Target journals/conferences, deadlines, & review checklists
│
├── data/                      # Local data storage (GIT IGNORED)
│   ├── raw/                   # Immutable raw datasets (read-only)
│   ├── interim/               # Intermediate transformed states
│   └── processed/             # Ready-to-train tensors or processed feature files
│
└── outputs/                   # Generated artifacts & run logs (GIT IGNORED)
    ├── experiments/           # Finished experiment artifacts by model/run (e.g., jepa/, unet/, diffusion/)
    │   ├── checkpoints/       # Model weight checkpoints (.pt, .ckpt, .safetensors)
    │   ├── figures/           # Output plots (.pdf, .png, .svg)
    │   ├── metrics/           # Evaluation JSONs, CSV summaries, & WandB logs
    │   └── logs/              # Console output logs & error tracebacks
    ├── reports/               # Compiled analysis outputs, cross-experiment summaries, & reports
    └── archieve/              # Archived previous runs and deprecated artifacts no longer in active use
```

---

## 2. Key Architectural Principles

### A. Strict Decoupling of Library vs. Execution
- **`src/` is a Python Library**: Holds purely functional, tested, reusable code. It should **never** execute training loops or hardcode file paths upon import.
- **`scripts/` are Entry Points**: CLI entry points that import from `src/`, parse arguments (e.g. via `argparse`, `click`, or `typer`), load configs, and run workflows.

### B. Immutable Data Rule
- **`data/raw/` is Read-Only**: Original raw datasets (e.g. `.h5ad`, `.csv`, `.parquet`, `.tar.gz`) must never be modified in-place by any script.
- All transformations write new files to `data/interim/` or `data/processed/`.

### C. Standardized Output Structure & Single Solid Directory Rule
- Avoid dumping generated plots or model files into the repository root.
- **Single Solid Directory Rule**: Always save and access each artifact from exactly ONE unambiguous, solid directory path. Never allow multiple fallback locations or speculative path probing.
- **Fail-Loud Policy**: Never guess or assume where an artifact or checkpoint is located. If an expected artifact is missing, fail loud immediately with an explicit error to prevent expensive compute waste and silent bugs.
- Subdirectories within `outputs/`:
  - `outputs/experiments/`: Dedicated artifact storage for each finished experiment (e.g. `jepa/`, `unet/`, `diffusion/`). Holds:
    - `outputs/experiments/checkpoints/`: Model weight checkpoints (`.pt`, `.ckpt`, `.safetensors`).
    - `outputs/experiments/figures/`: Publication-ready output plots (`.pdf`, `.png`, `.svg`).
    - `outputs/experiments/metrics/`: Evaluation JSONs, CSV summaries, & WandB logs.
    - `outputs/experiments/logs/`: Console output logs & execution tracebacks.
  - `outputs/reports/`: Location where compiled outputs of experiment analyses, cross-model benchmark comparisons, and aggregated reports are compiled.
  - `outputs/archieve/`: Dedicated location where previous runs that are not being used anymore are moved, keeping active workspaces uncluttered.
- Ensure scripts programmatically construct required output subdirectories if they do not exist.

### D. Strict Test Output Isolation Rule
- **`tests/outputs/` for Test Outputs**: All unit tests, integration tests, and smoke test suites MUST write generated test artifacts, mock tensors, and dummy logs strictly inside `tests/outputs/` (or ephemeral tempdirs).
- **Zero Pollution of Real Outputs**: Tests must **never** read from, write to, or overwrite files in the real root `outputs/` directory. This ensures test suites never corrupt legitimate experiment results, overwrite trained checkpoints, or pollute empirical metrics.

### E. Versioning LaTeX Source Code
- Include LaTeX source files under `paper/latex/` inside git tracking.
- Exclude compiled LaTeX intermediate build artifacts (`*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out`, `*.toc`) in `.gitignore`.

---

## 3. Recommended `.gitignore` Snippet for AI Research

```gitignore
# Environment & Dependencies
.venv/
env/
venv/
*.egg-info/
dist/
build/

# Local Data & Outputs (Heavy Binaries)
data/
outputs/
tests/outputs/
runs/
wandb/
.hydra/

# Python & Jupyter Caches
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.pytest_cache/
.ruff_cache/

# Operating System & IDE
.DS_Store
Thumbs.db
.vscode/
.idea/

# LaTeX Build Artifacts (Keep .tex, .bib, .cls, .pdf figures tracked)
paper/latex/*.aux
paper/latex/*.bbl
paper/latex/*.blg
paper/latex/*.fdb_latexmk
paper/latex/*.fls
paper/latex/*.log
paper/latex/*.out
paper/latex/*.synctex.gz
paper/latex/*.toc
```

---

## 4. Modern Tooling Recommendations

| Category | Tool | Purpose |
| :--- | :--- | :--- |
| **Package Manager** | `uv` or `pixi` | Ultra-fast dependency resolution and lockfile management |
| **Config Management** | `Hydra` or `Pydantic-Settings` | Hierarchical configuration composition with CLI overrides |
| **Linting & Formatting**| `ruff` | Fast, unified linter and code formatter |
| **Testing** | `pytest` | Unit testing for neural net shapes, loss bounds, and data pipelines |
| **Experiment Tracking** | `Weights & Biases` / `MLflow` | Dynamic logging of metrics, hyperparameters, and artifacts |
| **Hardware Acceleration**| `PyTorch` / `Lightning` | Portable acceleration across CUDA, MPS (Apple Silicon), and CPU |

---

## 5. Quickstart: Using This Template

```bash
# 1. Initialize repository structure
mkdir -p configs src/my_project scripts notebooks tests/outputs docs paper/latex data/{raw,processed} outputs/{experiments/{checkpoints,figures,metrics,logs},reports,archieve}

# 2. Initialize uv Python environment
uv init --lib

# 3. Add core AI research dependencies
uv add torch numpy pandas scipy matplotlib seaborn scikit-learn tqdm pytest ruff
```
