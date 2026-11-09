#!/usr/bin/env python3
"""
Pre-encode Offline Dataset into Spatial VAE Latents.

Encodes raw pixel observations from data/raw/ into compact continuous spatial latents
using the frozen best Spatial VAE checkpoint, saving to data/processed/latent_shards/.
Accelerates downstream Action-Conditioned Latent Diffusion training by 10x-20x.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from wm_control.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    fail_loud,
    get_experiment_dir,
)
from wm_control.models.vae import SpatialVAE
from wm_control.utils.device import get_device


def preencode_dataset(
    vae_ckpt_path: Path | None = None,
    raw_dir: Path = DATA_RAW_DIR,
    output_dir: Path | None = None,
    batch_size: int = 256,
    device_str: str | None = None,
) -> Path:
    """
    Encode all raw dataset shards into latent shards.
    """
    if vae_ckpt_path is None:
        vae_ckpt_path = get_experiment_dir("vae") / "checkpoints" / "best_vae.pt"
    if not vae_ckpt_path.exists():
        fail_loud(
            missing_resource=vae_ckpt_path,
            caller_context="preencode_dataset",
            remediation_cmd="python scripts/train_vae.py --epochs 15",
        )

    if output_dir is None:
        output_dir = DATA_PROCESSED_DIR / "latent_shards"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(device_str)
    print(f"\nPre-encoding raw dataset into VAE latents on device: {device}")
    print(f"  VAE Checkpoint: {vae_ckpt_path}")
    print(f"  Raw Source:     {raw_dir}")
    print(f"  Destination:    {output_dir}")

    # Load and freeze VAE
    ckpt = torch.load(vae_ckpt_path, map_location=device)
    latent_channels = ckpt.get("latent_channels", 4)
    base_channels = ckpt.get("base_channels", 32)
    vae = SpatialVAE(in_channels=3, latent_channels=latent_channels, base_channels=base_channels).to(device)
    vae.load_state_dict(ckpt["model_state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    raw_shards = sorted(list(raw_dir.glob("shard_*.npz")))
    if not raw_shards:
        fail_loud(
            missing_resource=raw_dir / "shard_*.npz",
            caller_context="preencode_dataset",
            remediation_cmd="python scripts/harvest_data.py --num-transitions 50000",
        )

    latent_shard_paths: list[str] = []
    total_transitions = 0
    start_time = time.time()

    pbar = tqdm(raw_shards, desc="Encoding shards", unit="shard")
    for shard_path in pbar:
        data = np.load(shard_path)
        obs_raw = data["obs"]          # [N, 3, 64, 64] uint8
        next_obs_raw = data["next_obs"]
        action = data["action"]
        reward = data["reward"]
        terminal = data["terminal"]
        telemetry = data["telemetry"]
        next_telemetry = data["next_telemetry"]

        n_samples = len(obs_raw)
        total_transitions += n_samples

        encoded_z: list[np.ndarray] = []
        encoded_next_z: list[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                b_obs = torch.from_numpy(obs_raw[i : i + batch_size]).float().div(255.0).to(device)
                b_next_obs = torch.from_numpy(next_obs_raw[i : i + batch_size]).float().div(255.0).to(device)

                m_z, _ = vae.encode(b_obs)
                m_next_z, _ = vae.encode(b_next_obs)

                encoded_z.append(m_z.cpu().numpy().astype(np.float32))
                encoded_next_z.append(m_next_z.cpu().numpy().astype(np.float32))

        z_array = np.concatenate(encoded_z, axis=0)
        next_z_array = np.concatenate(encoded_next_z, axis=0)

        out_name = f"latent_{shard_path.name}"
        out_path = output_dir / out_name
        np.savez_compressed(
            out_path,
            z_t=z_array,
            next_z=next_z_array,
            action=action,
            reward=reward,
            terminal=terminal,
            telemetry=telemetry,
            next_telemetry=next_telemetry,
        )
        latent_shard_paths.append(str(out_path))

    elapsed = time.time() - start_time
    print(f"\nPre-encoding Complete: {total_transitions:,} transitions across {len(latent_shard_paths)} shards in {elapsed:.1f}s.")

    # Read original split ratios
    index_file = DATA_PROCESSED_DIR / "dataset_index.json"
    if index_file.exists():
        with open(index_file) as f:
            meta = json.load(f)
        n_train = len(meta["train_shards"])
        n_val = len(meta["val_shards"])
    else:
        num_shards = len(latent_shard_paths)
        n_train = round(num_shards * 0.8)
        n_val = round(num_shards * 0.1)

    latent_index = {
        "total_transitions": total_transitions,
        "latent_channels": latent_channels,
        "latent_shape": [latent_channels, 16, 16],
        "train_shards": latent_shard_paths[:n_train],
        "val_shards": latent_shard_paths[n_train : n_train + n_val],
        "test_shards": latent_shard_paths[n_train + n_val :],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    latent_index_file = DATA_PROCESSED_DIR / "latent_dataset_index.json"
    with open(latent_index_file, "w") as f:
        json.dump(latent_index, f, indent=2)

    print(f"Latent index written to: {latent_index_file}")
    return latent_index_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-encode raw CarRacing observations into VAE latents.")
    parser.add_argument("--vae-ckpt", type=Path, default=None, help="Path to best_vae.pt.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding.")
    parser.add_argument("--device", type=str, default=None, help="Device override.")
    args = parser.parse_args()

    preencode_dataset(
        vae_ckpt_path=args.vae_ckpt,
        batch_size=args.batch_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
