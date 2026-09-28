"""Train the SA-PFE-EMRF aggregator q_phi(M|obs) on TS union labels (Stage S1.1b).

Setup
-----
  * Data:        data/phase1_ts_iso15/labels_U/ (749 cases, 42 union channels + body silhouette).
                 We use case_NNNN.nii.gz under labels_U/, no need to touch CT.
  * Backbone:    src.aggregator.PartialMaskAggregator (MONAI BasicUNet).
  * Synthetic partials: src.aggregator.build_partial_inputs() per tex Algorithm 1.
  * Loss:        L = presence_aware_bce(prob, gt) + lambda_anat * E_anat(prob)
  * Long-running training should be launched detached (systemd/tmux), not tied to a shell session.
                 For dev iteration we keep --smoke flag that runs a tiny in-process loop.

Usage
-----
  Smoke (CPU/GPU, 3 cases, 2 steps):
    python scripts/train_aggregator.py --smoke

  Full run:
    python scripts/train_aggregator.py --epochs 20 --bs 1 --lr 1e-4 \\
      --features 32 32 64 128 256 32 --lambda-anat 0.1 --out checkpoints/aggregator_v0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aggregator import (
    AggregatorConfig,
    PartialMaskAggregator,
    build_partial_inputs,
    presence_aware_bce,
)
from src.anatomy_prior import AnatomyPrior, AnatomyStats


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data/phase1_ts_iso15"
DEFAULT_STATS = DEFAULT_DATA_DIR / "anatomy_stats.json"


# =============================================================================
# Dataset
# =============================================================================


class TSUnionLabelDataset(Dataset):
    """Loads 256^3 union-label volumes and emits soft-one-hot tensors.

    Item: (label_volume_onehot, gt_mask_onehot) where both are identical:
      (K, D, H, W) float32 one-hot tensor.

    Partial-observation synthesis happens *in the training loop* (so each call
    can use a different random S_j subset and erasure prob -- tex Algorithm 1).
    """

    def __init__(
        self,
        data_dir: Path,
        scored_ids: list[int],
        split: str = "training",
        dataset_json: str = "expert_U_dataset.json",
        max_cases: int = 0,
    ):
        self.data_dir = Path(data_dir)
        ds = json.loads((self.data_dir / dataset_json).read_text())
        entries = ds.get(split, [])
        if max_cases > 0:
            entries = entries[:max_cases]
        self.entries = entries
        self.scored_ids = list(scored_ids)
        self._id_lut: torch.Tensor | None = None  # built lazily

    def __len__(self) -> int:
        return len(self.entries)

    def _build_lut(self, max_id: int) -> torch.Tensor:
        # Map any MAISI id -> position in scored_ids, or -1 if not tracked.
        lut = torch.full((max_id + 1,), -1, dtype=torch.long)
        for pos, sid in enumerate(self.scored_ids):
            if 0 <= sid <= max_id:
                lut[sid] = pos
        self._id_lut = lut
        return lut

    def __getitem__(self, i: int) -> torch.Tensor:
        e = self.entries[i]
        lpath = self.data_dir / e["label"]
        data = nib.load(str(lpath)).get_fdata(caching="unchanged").astype(np.int32)
        t = torch.from_numpy(data).long()
        max_id = int(t.max())
        if self._id_lut is None or self._id_lut.numel() <= max_id:
            self._build_lut(max(max_id, max(self.scored_ids) if self.scored_ids else 0))
        # Translate ids to channel positions; -1 means background/unused.
        ch_pos = self._id_lut[t.clamp(min=0)]
        K = len(self.scored_ids)
        oh = F.one_hot(ch_pos.clamp(min=0), num_classes=K).movedim(-1, 0).float()  # (K,D,H,W)
        # Zero out channels where ch_pos was -1 (background id not tracked).
        bg_mask = (ch_pos < 0).unsqueeze(0)  # (1,D,H,W)
        oh = oh * (~bg_mask).float()
        return oh


# =============================================================================
# Training loop
# =============================================================================


def per_channel_dice(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Returns (B,K) per-channel soft-Dice."""
    inter = (prob * target).flatten(2).sum(dim=2)
    s_prob = prob.flatten(2).sum(dim=2)
    s_targ = target.flatten(2).sum(dim=2)
    return (2.0 * inter + eps) / (s_prob + s_targ + eps)


def sample_observed_groups(B: int, n_experts: int, p_both: float = 0.7, rng: random.Random | None = None) -> list[list[int]]:
    """For each batch element decide which subset of experts observed."""
    rng = rng or random
    out: list[list[int]] = []
    for _ in range(B):
        u = rng.random()
        if u < p_both:
            out.append(list(range(n_experts)))
        elif u < (1 + p_both) / 2:
            out.append([0])
        else:
            out.append([1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--label-partition", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data/label_partition.json")
    ap.add_argument("--anatomy-stats", type=Path, default=DEFAULT_STATS)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "checkpoints/aggregator_v0")
    ap.add_argument("--features", type=int, nargs=6, default=[32, 32, 64, 128, 256, 32])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-anat", type=float, default=0.1)
    ap.add_argument("--w-obs", type=float, default=1.5)
    ap.add_argument("--w-unobs", type=float, default=1.0)
    ap.add_argument("--erasure-prob", type=float, default=0.1)
    ap.add_argument("--p-both", type=float, default=0.7)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=1)
    ap.add_argument("--max-cases", type=int, default=0, help="Cap dataset for smoke runs.")
    ap.add_argument("--grid-size", type=int, default=0,
                    help="If > 0, average-pool labels to (grid,grid,grid) for fast iteration.")
    ap.add_argument("--smoke", action="store_true", help="Tiny in-process smoke run; overrides several args.")
    ap.add_argument("--amp", action="store_true", help="Run forward+loss under bf16 autocast.")
    ap.add_argument("--pos-weight", type=float, default=1.0,
                    help="BCE-with-logits positive-class weight. >1 to fight sparse-mask imbalance.")
    ap.add_argument("--lambda-dice", type=float, default=0.0,
                    help="Weight on the per-channel soft-Dice loss term (averaged over channels that have any positives in the batch).")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="Gradient-accumulation steps (effective batch = bs * grad_accum).")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 1
        args.bs = 1
        args.max_cases = 2
        args.features = [16, 16, 32, 64, 128, 16]
        args.lambda_anat = 0.0  # may skip anatomy if stats not ready
        if args.grid_size <= 0:
            args.grid_size = 64  # smoke shrinks 256^3 -> 64^3 for fast iteration

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    # ---- partition & channel layout ------------------------------------------------
    partition = json.loads(args.label_partition.read_text())
    expert_A_ids = list(partition["expert_A_abdominal"])
    expert_B_ids = list(partition["expert_B_spine"])
    union_ids = sorted(set(expert_A_ids) | set(expert_B_ids))
    scored_ids = list(union_ids)  # body silhouette excluded from aggregator targets
    id_to_idx = {sid: i for i, sid in enumerate(scored_ids)}
    expert_channel_groups = [
        [id_to_idx[sid] for sid in expert_A_ids if sid in id_to_idx],
        [id_to_idx[sid] for sid in expert_B_ids if sid in id_to_idx],
    ]
    print(f"[info] K_union = {len(scored_ids)}  (A: {len(expert_channel_groups[0])} ch, B: {len(expert_channel_groups[1])} ch)")

    # ---- anatomy prior (optional) -------------------------------------------------
    anatomy_prior = None
    if args.lambda_anat > 0:
        if not args.anatomy_stats.exists():
            print(f"[warn] anatomy stats missing at {args.anatomy_stats}; disabling anatomy term.")
            args.lambda_anat = 0.0
        else:
            stats = AnatomyStats.from_json(args.anatomy_stats, device=args.device)
            # The aggregator does not model body silhouette (200); align prior to its K.
            stats_ids_aligned = list(stats.scored_ids)
            if stats_ids_aligned[: len(scored_ids)] != scored_ids:
                print(f"[warn] scored_ids mismatch between aggregator ({scored_ids[:5]}...) "
                      f"and stats ({stats_ids_aligned[:5]}...); using stats order.")
                scored_ids = stats_ids_aligned[: len(scored_ids)]  # best-effort align prefix
            anatomy_prior = AnatomyPrior(stats).to(args.device)
            print(f"[info] anatomy prior loaded ({len(anatomy_prior.adj_idx)} adjacency pairs).")

    # ---- data ---------------------------------------------------------------------
    ds = TSUnionLabelDataset(args.data_dir, scored_ids=scored_ids, max_cases=args.max_cases)
    n_workers = 0 if args.smoke else 4
    dl = DataLoader(
        ds, batch_size=args.bs, shuffle=True,
        num_workers=n_workers,
        persistent_workers=(n_workers > 0),
        pin_memory=args.device.startswith("cuda"),
    )
    print(f"[info] dataset size: {len(ds)} cases, bs={args.bs}, steps/epoch={len(dl)}")

    def maybe_downsample(x: torch.Tensor) -> torch.Tensor:
        """Average-pool (B,K,D,H,W) one-hot to (B,K,g,g,g) if args.grid_size > 0."""
        if args.grid_size and x.shape[-1] != args.grid_size:
            return F.adaptive_avg_pool3d(x, args.grid_size)
        return x

    # ---- model --------------------------------------------------------------------
    cfg = AggregatorConfig(n_union_channels=len(scored_ids), base_features=tuple(args.features))
    net = PartialMaskAggregator(cfg).to(args.device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[info] aggregator params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(len(dl), 1))

    # ---- train --------------------------------------------------------------------
    step = 0
    history = []
    for ep in range(args.epochs):
        net.train()
        t_ep = time.time()
        for batch in dl:
            gt = batch.to(args.device, non_blocking=True)  # (B, K, D, H, W)
            gt = maybe_downsample(gt)
            B = gt.shape[0]
            obs_groups = sample_observed_groups(B, n_experts=2, p_both=args.p_both, rng=rng)
            with torch.no_grad():
                partial, presence = build_partial_inputs(
                    gt, expert_channel_groups, obs_groups,
                    erasure_prob=args.erasure_prob,
                )
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.amp and args.device.startswith("cuda")
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with amp_ctx:
                logits = net(partial, presence)
                # Fused BCE-with-logits with per-channel observed weight + global pos_weight.
                ch_presence = presence.mean(dim=(2, 3, 4))  # (B,K)
                ch_weight = args.w_obs * ch_presence + args.w_unobs * (1.0 - ch_presence)
                ch_weight = ch_weight.view(*ch_weight.shape, 1, 1, 1).to(logits.dtype)
                pos_w = torch.full((1,), args.pos_weight, device=logits.device, dtype=logits.dtype)
                bce_perv = F.binary_cross_entropy_with_logits(
                    logits, gt.to(logits.dtype), reduction="none",
                    pos_weight=pos_w,
                )
                bce_perv.mul_(ch_weight)
                loss_bce = bce_perv.mean()
                del bce_perv, ch_weight
                prob = torch.sigmoid(logits)
                # Optional soft-Dice loss to fight class imbalance.
                loss_dice = torch.tensor(0.0, device=args.device, dtype=loss_bce.dtype)
                if args.lambda_dice > 0:
                    gt_d = gt.to(prob.dtype)
                    inter = (prob * gt_d).flatten(2).sum(dim=2)             # (B,K)
                    s_prob = prob.flatten(2).sum(dim=2)
                    s_targ = gt_d.flatten(2).sum(dim=2)
                    soft_dice = (2.0 * inter + 1e-6) / (s_prob + s_targ + 1e-6)
                    # Only count channels that have any GT positives (others give Dice=0 trivially
                    # and would bias the mean toward "predict zero").
                    has_pos = (s_targ > 0).float()
                    denom = torch.clamp(has_pos.sum(), min=1.0)
                    loss_dice = ((1.0 - soft_dice) * has_pos).sum() / denom
                loss_anat = torch.tensor(0.0, device=args.device)
                if anatomy_prior is not None:
                    prob_for_prior = prob.float()
                    if prob_for_prior.shape[1] != anatomy_prior.K:
                        # Aggregator omits body silhouette (last stats channel); pad zero.
                        pad = torch.zeros_like(prob_for_prior[:, :anatomy_prior.K - prob_for_prior.shape[1]])
                        prob_for_prior = torch.cat([prob_for_prior, pad], dim=1)
                    loss_anat = anatomy_prior(prob_for_prior).mean()
                    del prob_for_prior
                loss = loss_bce + args.lambda_dice * loss_dice + args.lambda_anat * loss_anat
                loss = loss / max(args.grad_accum, 1)

            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                with torch.no_grad():
                    dice_all = per_channel_dice(prob, gt)  # (B,K)
                    # Dice averaged only over channels with positives (the meaningful metric)
                    s_targ_ch = gt.flatten(2).sum(dim=2)  # (B,K)
                    has_pos_ch = (s_targ_ch > 0).float()
                    denom_ch = torch.clamp(has_pos_ch.sum(), min=1.0)
                    dice_pos = ((dice_all * has_pos_ch).sum() / denom_ch).item()
                    dice_all_mean = dice_all.mean().item()
                    p_max = float(prob.max().detach().cpu())
                    p_mean = float(prob.mean().detach().cpu())
                msg = (f"ep {ep} step {step}: loss={loss.item():.4f} "
                       f"bce={loss_bce.item():.4f} dice_loss={loss_dice.item():.4f} "
                       f"anat={loss_anat.item():.4f} "
                       f"dice_pos={dice_pos:.3f} dice_all={dice_all_mean:.3f} "
                       f"p_max={p_max:.2f} p_mean={p_mean:.2f} "
                       f"lr={opt.param_groups[0]['lr']:.2e}")
                print(msg)
                history.append({"step": step, "loss": loss.item(), "bce": loss_bce.item(),
                                "dice_loss": loss_dice.item(),
                                "anat": loss_anat.item(),
                                "dice_pos": dice_pos, "dice_all": dice_all_mean,
                                "p_max": p_max, "p_mean": p_mean})

        print(f"[ep {ep}] done in {time.time()-t_ep:.1f}s")

        if (ep + 1) % args.ckpt_every == 0:
            ck_path = args.out / f"aggregator_ep{ep:02d}.pt"
            torch.save({
                "epoch": ep,
                "step": step,
                "model_state": net.state_dict(),
                "opt_state": opt.state_dict(),
                "cfg": cfg.__dict__,
                "scored_ids": scored_ids,
                "expert_channel_groups": expert_channel_groups,
                "args": vars(args),
                "history_tail": history[-10:],
            }, ck_path)
            (args.out / "history.json").write_text(json.dumps(history, indent=2, default=str))
            print(f"[ckpt] wrote {ck_path}")

    print("[done]")


if __name__ == "__main__":
    main()
