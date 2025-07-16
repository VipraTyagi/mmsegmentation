#!/usr/bin/env python3
"""
Compute a diagonal Fisher information matrix for a trained MMSegmentation model
and print progress + summary statistics.
"""

import argparse
import os
import pathlib
import time
import torch
from tqdm import tqdm
from mmengine.config import Config

# -------------------------------------------------------------------
#  Version‑agnostic dataloader builder
# -------------------------------------------------------------------
try:  # MMEngine ≥0.10: dict interface
    from mmengine.runner import Runner

    def build_dl(dataset, cfg_dict):
        cfg = cfg_dict.copy()
        cfg["dataset"] = dataset
        return Runner.build_dataloader(cfg)

except ImportError:  # MMEngine 0.7‑0.9: keyword interface
    from mmengine.dataset.utils import build_dataloader as _legacy_build

    def build_dl(dataset, cfg_dict):
        return _legacy_build(dataset, **cfg_dict)
# -------------------------------------------------------------------

from mmseg.registry import DATASETS
from mmseg.apis import init_model

# Disable stray pdb breakpoints
os.environ.setdefault("PYTHONBREAKPOINT", "0")


def main() -> None:
    ap = argparse.ArgumentParser(description="Post‑hoc Fisher computation")
    ap.add_argument("cfg", help="Config used for task‑1 training")
    ap.add_argument("checkpoint", help="Task‑1 checkpoint (.pth)")
    ap.add_argument("--out", default="fisher_diag.pth",
                    help="Filename for Fisher dict (placed next to checkpoint)")
    args = ap.parse_args()

    cfg = Config.fromfile(args.cfg)

    # 1 ── model ------------------------------------------------------------------
    model = init_model(cfg, args.checkpoint, device="cuda")
    model.eval()

    # 2 ── dataloader with finite sampler ----------------------------------------
    raw_dl = cfg.train_dataloader
    dataset = DATASETS.build(raw_dl.dataset)

    dl_cfg = {k: v for k, v in raw_dl.items() if k != "dataset"}
    dl_cfg["sampler"] = dict(type="DefaultSampler", shuffle=False, round_up=False)
    loader = build_dl(dataset, dl_cfg)

    bar = tqdm(
        loader,
        total=len(loader),  # batches
        desc="train :",
        ncols=80,
        bar_format="{desc} {l_bar}{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    # 3 ── Fisher containers ------------------------------------------------------
    fisher = {
        n: torch.zeros_like(p, device="cuda")
        for n, p in model.named_parameters() if p.requires_grad
    }

    batch_times = []
    for batch in bar:
        tic = time.perf_counter()

        # forward + task loss  (MMSeg 1.x)
        batch = model.data_preprocessor(batch, training=False)
        loss = sum(model(batch["inputs"], batch["data_samples"], mode="loss").values())

        # backward
        model.zero_grad()
        loss.backward()

        # accumulate Fisher
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2)

        # progress‑bar bookkeeping
        batch_times.append(time.perf_counter() - tic)
        if len(batch_times) > 20:
            batch_times.pop(0)
        eta_min = (sum(batch_times) / len(batch_times)) * (len(loader) - bar.n) / 60
        bar.set_postfix_str(f"loss={loss.item():.3f}, ETA={eta_min:.1f} min")

    # 4 ── normalise --------------------------------------------------------------
    for n in fisher:
        fisher[n] /= len(loader)

    # 5 ── save -------------------------------------------------------------------
    ckpt_path = pathlib.Path(args.checkpoint).resolve()
    out_path = ckpt_path.with_name(args.out)
    torch.save(
        {
            "fisher": fisher,
            "params": {n: p.detach().cpu() for n, p in model.named_parameters()},
        },
        out_path,
    )

    # 6 ── print Fisher stats -----------------------------------------------------
    flat_diag = torch.cat([v.flatten() for v in fisher.values()])
    print("\n✓ Diagonal Fisher written to", out_path)
    print(f"   stats → min {flat_diag.min():.4e} "
          f"mean {flat_diag.mean():.4e} max {flat_diag.max():.4e}")


if __name__ == "__main__":
    main()
