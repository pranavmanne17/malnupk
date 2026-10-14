"""Torch dataset plumbing: tensor conversion, normalisation stats, batching."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .drugs import DRUGS, DRUG_NAMES
from .schedule import N_DOSES_DEFAULT

TENSOR_KEYS_F = ["anthro", "labs", "lab_mask", "tox", "drug_feat", "regimen",
                 "aux", "times", "conc_obs", "conc_true", "dt", "bolus",
                 "inf_rate", "pk_params", "sample_mask", "metrics"]
TENSOR_KEYS_L = ["obs_segment", "pid", "drug_id", "setting_id", "class_id",
                 "n_transit", "is_iv"]


class PKDataset(Dataset):
    def __init__(self, packed: Dict[str, np.ndarray]):
        self.d: Dict[str, torch.Tensor] = {}
        for k in TENSOR_KEYS_F:
            self.d[k] = torch.as_tensor(packed[k], dtype=torch.float32)
        for k in TENSOR_KEYS_L:
            self.d[k] = torch.as_tensor(packed[k], dtype=torch.long)
        # Convenience scalars used by the solver and the metric heads.
        self.d["weight_kg"] = self.d["anthro"][:, 2].clone()
        self.d["dose_mg"] = torch.exp(self.d["regimen"][:, 1])
        self.d["tau"] = self.d["regimen"][:, 2].clone()
        self.d["n_doses"] = torch.full_like(self.d["tau"],
                                            float(N_DOSES_DEFAULT))
        mic = torch.tensor([DRUGS[DRUG_NAMES[i]].mic_mg_l
                            for i in packed["drug_id"]], dtype=torch.float32)
        self.d["mic"] = mic
        self.n = len(self.d["pid"])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {k: v[i] for k, v in self.d.items()}


def make_loader(packed: Dict[str, np.ndarray], batch_size: int = 256,
                shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(PKDataset(packed), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=False)


def compute_stats(packed: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Normalisation statistics, computed on the training split only.

    Laboratory statistics are computed over *observed* values only, so that
    the missing-value token is not pulled toward an artefact of the
    missingness rate.
    """
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for k in ["anthro", "tox", "drug_feat", "regimen", "aux"]:
        x = packed[k].astype(np.float64)
        stats[k] = (x.mean(0), x.std(0) + 1e-8)
    labs, mask = packed["labs"].astype(np.float64), packed["lab_mask"]
    n_obs = mask.sum(0) + 1e-8
    mu = (labs * mask).sum(0) / n_obs
    var = ((labs - mu) ** 2 * mask).sum(0) / n_obs
    stats["labs"] = (mu, np.sqrt(var) + 1e-8)
    return stats


def apply_stats(model, stats: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    model.enc_anthro.set_stats(*stats["anthro"])
    model.enc_tox.set_stats(*stats["tox"])
    model.enc_drug.set_stats(*stats["drug_feat"])
    model.enc_regimen.set_stats(*stats["regimen"])
    model.enc_labs.set_stats(*stats["labs"])


def to_device(batch: Dict[str, torch.Tensor], device: torch.device
              ) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
