"""
Sanity tests for package setup, device resolution, and fail-loud policy.
"""

from pathlib import Path

import pytest
import torch

from wm_control.config import (
    OUTPUTS_DIR,
    TESTS_OUTPUT_DIR,
    fail_loud,
)
from wm_control.utils.device import get_device
from wm_control.utils.seed import seed_everything


def test_package_imports():
    """Verify primary exports from wm_control."""
    import wm_control

    assert hasattr(wm_control, "TeacherFreeLeWorldModel")
    assert hasattr(wm_control, "SigRegLoss")
    assert hasattr(wm_control, "ActionConditionedTransitionLoss")
    assert hasattr(wm_control, "compute_effective_rank")
    assert hasattr(wm_control, "get_device")
    assert hasattr(wm_control, "seed_everything")


def test_device_resolution():
    """Verify auto-detection and explicit device selection."""
    device = get_device()
    assert isinstance(device, torch.device)

    cpu_device = get_device("cpu")
    assert cpu_device.type == "cpu"

    with pytest.raises(ValueError):
        get_device("invalid_device_name")


def test_seed_everything():
    """Verify reproducibility after seed_everything."""
    seed_everything(1234)
    r1 = torch.randn(10)
    seed_everything(1234)
    r2 = torch.randn(10)
    assert torch.equal(r1, r2)


def test_fail_loud_policy():
    """Verify fail_loud raises explicit diagnostic exception."""
    with pytest.raises(FileNotFoundError) as exc_info:
        fail_loud(
            missing_resource="/path/to/missing_model.pt",
            caller_context="Unit test verification",
            remediation_cmd="python scripts/train_vae.py",
        )
    assert "[FAIL LOUD]" in str(exc_info.value)
    assert "missing_model.pt" in str(exc_info.value)
    assert "python scripts/train_vae.py" in str(exc_info.value)


def test_test_output_isolation(isolated_test_dir: Path):
    """Verify test outputs write strictly inside tests/outputs/ and never root outputs/."""
    assert isolated_test_dir.is_relative_to(TESTS_OUTPUT_DIR)

    test_file = isolated_test_dir / "sample_test_artifact.txt"
    test_file.write_text("isolated test content")
    assert test_file.exists()

    # Verify root outputs/ is completely unaffected
    assert not test_file.is_relative_to(OUTPUTS_DIR)
