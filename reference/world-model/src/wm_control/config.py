"""
Centralized Configuration and Deterministic Path Management for wm-control.

Enforces:
- The Solid 1-Directory Rule (agents.md & dir_structure.md)
- Zero-Assumption & Fail-Loud Policy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Canonical Root Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
REPORTS_DIR = OUTPUTS_DIR / "reports"
ARCHIEVE_DIR = OUTPUTS_DIR / "archieve"

TESTS_DIR = PROJECT_ROOT / "tests"
TESTS_OUTPUT_DIR = TESTS_DIR / "outputs"


def fail_loud(
    missing_resource: str | Path,
    caller_context: str,
    remediation_cmd: str,
    exception_type: type[Exception] = FileNotFoundError,
) -> None:
    """
    Fail loud immediately with an explicit, actionable error message.

    Enforces Section 7 of agents.md.
    """
    msg = (
        f"\n[FAIL LOUD] Missing required resource: {missing_resource}\n"
        f"Caller Context: {caller_context}\n"
        f"Remediation: Run the following upstream step to generate it:\n"
        f"  >> {remediation_cmd}\n"
    )
    raise exception_type(msg)


def get_experiment_dir(experiment_name: str, create_if_missing: bool = True) -> Path:
    """
    Resolve exactly ONE solid, deterministic canonical experiment path.

    Outputs map strictly to outputs/experiments/<experiment_name>/
    """
    if not experiment_name or "/" in experiment_name or "\\" in experiment_name:
        raise ValueError(f"Invalid experiment_name '{experiment_name}'. Must be a single alphanumeric identifier.")

    exp_dir = EXPERIMENTS_DIR / experiment_name
    if create_if_missing:
        (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (exp_dir / "figures").mkdir(parents=True, exist_ok=True)
        (exp_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    return exp_dir


@dataclass
class EnvConfig:
    env_id: str = "CarRacing-v3"
    continuous: bool = True
    crop_dashboard: bool = True
    target_height: int = 64
    target_width: int = 64
    frame_stack: int = 1
    action_dim: int = 3  # [steering, gas, brake]
    max_episode_steps: int = 1000


@dataclass
class DatasetConfig:
    raw_dir: Path = DATA_RAW_DIR
    processed_dir: Path = DATA_PROCESSED_DIR
    total_transitions: int = 50_000
    chunk_size: int = 2_000
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class SigRegConfig:
    latent_dim: int = 128
    action_dim: int = 3
    in_channels: int = 3
    predictor_hidden_dim: int = 512
    predictor_layers: int = 3
    sigreg_weight: float = 1.0
    num_projections: int = 1024
    transition_loss_type: str = "smooth_l1"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    num_epochs: int = 100
