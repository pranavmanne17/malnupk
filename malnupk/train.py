"""Training loop for MalnuPK."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .data import (apply_stats, compute_stats, make_loader, to_device)
from .dataset import load_npz
from .features import AUX_PHYSIOLOGY, PK_PARAM_NAMES
from .model import (BlackBoxNet, LossWeights, MalnuPKNet, ModelConfig,
                    compute_loss, masked_log_mse)


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    grad_clip: float = 2.0
    patience: int = 12
    hidden: int = 128
    drug_hidden: int = 64
    dropout: float = 0.12
    depth: int = 2
    w_params: float = 0.25
    w_aux: float = 0.10
    supervise_params: bool = True
    model_type: str = "hybrid"       # 'hybrid' | 'blackbox'
    #: Input blocks to zero out, for information ablations. The model keeps
    #: the same capacity, so a difference in performance is attributable to
    #: the information and not to the architecture.
    ablate_blocks: tuple = ()
    seed: int = 0
    device: str = "cpu"


def ablate(packed: Dict[str, np.ndarray], blocks) -> Dict[str, np.ndarray]:
    """Return a copy with the named input blocks zeroed."""
    if not blocks:
        return packed
    out = dict(packed)
    for b in blocks:
        if b not in out:
            raise KeyError(f"unknown input block to ablate: {b}")
        out[b] = np.zeros_like(out[b])
        if b == "labs":
            out["lab_mask"] = np.zeros_like(out["lab_mask"])
    return out


def build_model(packed: Dict[str, np.ndarray], cfg: TrainConfig) -> nn.Module:
    mcfg = ModelConfig(
        anthro_dim=packed["anthro"].shape[1],
        lab_dim=packed["labs"].shape[1],
        tox_dim=packed["tox"].shape[1],
        drug_dim=packed["drug_feat"].shape[1],
        regimen_dim=packed["regimen"].shape[1],
        aux_dim=packed["aux"].shape[1],
        hidden=cfg.hidden, drug_hidden=cfg.drug_hidden,
        dropout=cfg.dropout, depth=cfg.depth, predict_aux=cfg.w_aux > 0,
        regimen_in_encoder=(cfg.model_type == "blackbox"),
    )
    if cfg.model_type == "blackbox":
        model = BlackBoxNet(mcfg, n_obs=packed["conc_obs"].shape[1])
    else:
        model = MalnuPKNet(mcfg)
    return model


def _init_param_centre(model: MalnuPKNet, packed: Dict[str, np.ndarray]) -> None:
    """Start the structural head near the data's own parameter scale.

    Without this the first forward pass produces clearances many orders of
    magnitude off, the log-likelihood saturates, and training spends its first
    epochs just finding the right decade.
    """
    p = packed["pk_params"].astype(np.float64)
    w = packed["anthro"][:, 2].astype(np.float64)
    logw = np.log(np.maximum(w, 0.5) / 13.0)
    allo = model.allo.cpu().numpy().reshape(1, -1)
    centred = np.log(np.clip(p, 1e-8, None)) - allo * logw.reshape(-1, 1)
    centre = np.median(centred, axis=0)
    # F and fu pass through a sigmoid, so invert it for those two columns.
    for j in (5, 6):
        med = float(np.clip(np.median(p[:, j]), 1e-4, 1 - 1e-4))
        centre[j] = math.log(med / (1 - med))
    with torch.no_grad():
        model.param_centre.copy_(torch.as_tensor(centre, dtype=torch.float32))


def evaluate(model: nn.Module, loader, device: torch.device,
             aux_stats: Optional[Tuple] = None) -> Dict[str, float]:
    model.eval()
    tot_nll, tot_mse, n_batch = 0.0, 0.0, 0
    param_err = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch)
            from .model import masked_lognormal_nll
            tot_nll += float(masked_lognormal_nll(
                out["conc"], batch["conc_obs"], out["sigma"],
                batch["sample_mask"]))
            # Fit quality against the *noise-free* truth at every time point,
            # which is the quantity we actually care about.
            full = torch.ones_like(batch["sample_mask"])
            tot_mse += float(masked_log_mse(out["conc"], batch["conc_true"], full))
            if "CL" in out:
                pred = torch.stack([out["CL"], out["V1"], out["Q"], out["V2"],
                                    out["ka"], out["F"], out["fu"]], dim=1)
                lp = torch.log(pred.clamp(min=1e-8))
                lt = torch.log(batch["pk_params"].clamp(min=1e-8))
                cm = torch.ones_like(lt)
                cm[:, 4] = 1.0 - batch["is_iv"].to(lt.dtype)
                param_err.append(float((((lp - lt) ** 2) * cm).sum()
                                       / cm.sum().clamp(min=1.0)))
            n_batch += 1
    res = {"nll": tot_nll / max(n_batch, 1),
           "log_mse_true": tot_mse / max(n_batch, 1)}
    if param_err:
        res["param_log_mse"] = float(np.mean(param_err))
    return res


def train(train_path: str, val_path: str, out_dir: str,
          cfg: TrainConfig) -> Dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(cfg.device)

    tr = ablate(load_npz(train_path), cfg.ablate_blocks)
    va = ablate(load_npz(val_path), cfg.ablate_blocks)
    stats = compute_stats(tr)

    model = build_model(tr, cfg).to(device)
    apply_stats(model, stats)
    if isinstance(model, MalnuPKNet):
        _init_param_centre(model, tr)

    aux_mu = torch.as_tensor(stats["aux"][0], dtype=torch.float32, device=device)
    aux_sd = torch.as_tensor(stats["aux"][1], dtype=torch.float32, device=device)
    aux_stats = (aux_mu, aux_sd)

    tl = make_loader(tr, cfg.batch_size, shuffle=True)
    vl = make_loader(va, cfg.batch_size, shuffle=False)

    w = LossWeights(conc=1.0, params=cfg.w_params, aux=cfg.w_aux)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(len(tl), 1)
    warmup = max(int(cfg.warmup_frac * total_steps), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history = []
    best = {"val": float("inf"), "epoch": -1}
    bad = 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        agg: Dict[str, float] = {}
        nb = 0
        for batch in tl:
            batch = to_device(batch, device)
            out = model(batch)
            loss, parts = compute_loss(
                out, batch, w, aux_stats=aux_stats,
                supervise_params=cfg.supervise_params)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        agg = {k: v / max(nb, 1) for k, v in agg.items()}
        val = evaluate(model, vl, device, aux_stats)
        # Model selection on fit to the noise-free truth, not on the noisy NLL.
        score = val["log_mse_true"]
        history.append({"epoch": epoch, "train": agg, "val": val,
                        "lr": sched.get_last_lr()[0]})
        flag = ""
        if score < best["val"] - 1e-5:
            best = {"val": score, "epoch": epoch}
            torch.save({"state_dict": model.state_dict(),
                        "cfg": asdict(cfg),
                        "stats": {k: (v[0].tolist(), v[1].tolist())
                                  for k, v in stats.items()},
                        "shapes": {k: list(tr[k].shape[1:])
                                   for k in ["anthro", "labs", "tox",
                                             "drug_feat", "regimen", "aux"]}},
                       os.path.join(out_dir, "best.pt"))
            bad = 0
            flag = " *"
        else:
            bad += 1
        if epoch % 1 == 0:
            pm = val.get("param_log_mse", float("nan"))
            print(f"  epoch {epoch:3d}  train {agg.get('total', 0):8.4f}  "
                  f"val_nll {val['nll']:8.4f}  val_logMSE {score:8.5f}  "
                  f"param_logMSE {pm:7.4f}  lr {sched.get_last_lr()[0]:.2e}{flag}",
                  flush=True)
        if bad >= cfg.patience:
            print(f"  early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    summary = {"best_epoch": best["epoch"], "best_val_log_mse": best["val"],
               "epochs_run": len(history), "seconds": elapsed,
               "config": asdict(cfg),
               "n_params": sum(p.numel() for p in model.parameters())}
    with open(os.path.join(out_dir, "history.json"), "w") as fh:
        json.dump({"summary": summary, "history": history}, fh, indent=2)
    print(f"  best epoch {best['epoch']} val_logMSE {best['val']:.5f} "
          f"({elapsed:.0f}s, {summary['n_params']:,} params)")
    return summary


def load_model(ckpt_path: str, packed: Dict[str, np.ndarray],
               device: str = "cpu") -> Tuple[nn.Module, TrainConfig]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckcfg = dict(ck["cfg"])
    ckcfg["ablate_blocks"] = tuple(ckcfg.get("ablate_blocks", ()))
    cfg = TrainConfig(**ckcfg)
    model = build_model(packed, cfg)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, cfg
