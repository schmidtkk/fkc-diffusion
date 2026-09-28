# FKC-Diffusion

Composable multi-expert **ControlNet** sampling, **mask-conditioned inpainting**, and **flow-GRPO reinforcement learning** on a [MONAI](https://monai.io) / [NVIDIA MAISI](https://github.com/NVIDIA/maisi) latent-diffusion backbone for 3D medical image (CT) synthesis.

This is research code, released as-is. It is **not** a clinical tool and has **no** medical approval of any kind.

## What is in here

| Component | Entry points | What it does |
|---|---|---|
| FKC-PoE sampler | `src/fkc_poe_sampler.py`, `scripts/eval_path2_cfg_poe.py`, `scripts/infer_fkc_poe_pilot.py` | Composes **K independent ControlNet experts** (one per anatomy / label group) on one shared MAISI `DiffusionModelUNetMaisi`. Selectable composition modes: `naive_mean`, `cfg_poe` (`v_un + Σ_k(v_k − v_un)`), `residual_sum`. Optional **Feynman–Kac correctors** (weight increments from expert-velocity inner products, Skreta et al., *Feynman-Kac Correctors in Diffusion*, ICML 2025) with SMC particles and ESS-triggered systematic resampling. |
| Spatial-aware FK prior | `src/anatomy_prior.py`, `scripts/verify_sa_fkc.py`, `scripts/compose_experiment_sa.py` | Differentiable anatomy energy `E_anat = E_vol + E_excl + E_adj` (organ volume z-scores, pairwise overlap, adjacency distances) used as a spatial prior inside the FK weight increment, and as a violation-rate evaluator for ablations. |
| Mask-conditioned inpainting | `src/cond_encoding.py` (`to_inpaint_cond`), `scripts/train_controlnet_inpaint.py`, `configs/config_network_inpaint45.json` | A ControlNet variant whose conditioning carries `[masked image latents, binary edit mask, label map]`, so generation is constrained outside the mask and free inside it — the inpainting/expert path used for the "path-2" consensus screens (`scripts/run_path2_consensus_inpaint_screen.sh`). |
| Flow-GRPO RL fine-tune | `scripts/train_controlnet_flowgrpo.py`, `src/flow_grpo_sampler.py`, `src/rl_reward.py`, `scripts/test_flowgrpo_math.py` | Group-relative policy optimization on the rectified-flow sampler: sample a group per prompt, score each member with a reward (Dice of a frozen segmentation network, `dice_cf_reward`), update with the group-normalised advantage. |
| Custom ControlNet trainer | `scripts/train_controlnet_custom.py` | Drop-in replacement for MAISI's `train_controlnet.py` that avoids the MONAI `CacheDataset` + multi-worker fork crashes: direct nibabel loading in the main thread, fixed-seed shuffling, AMP, per-iteration logging, checkpoint resume. Includes a run-contract / training-proof layer (`src/fov_training_proof.py`) that hashes the code + configs + data roots into signed receipts, and latent-mask loss contracts (`src/fov_latent_mask.py`). |
| Aggregator | `src/aggregator.py`, `scripts/train_aggregator.py`, `scripts/probe_aggregator.py` | Learned partial-mask aggregator that fuses per-expert outputs where each expert only owns part of the label space. |
| Mask & eval tooling | `src/shield_masks.py`, `scripts/build_eval_masks.py`, `scripts/diag_shielded.py`, `scripts/compute_fkc_metrics.py`, `scripts/verify_fkc_k4.py` | Region "shield" masks for compositional evaluation (score each expert only inside its own territory), eval-mask construction, per-region metrics, K=4 composition verification. |

## Install

Python ≥ 3.10, CUDA GPU strongly recommended.

```bash
pip install torch nibabel numpy scipy scikit-image "monai>=1.4"
```

`monai >= 1.4` provides the MAISI networks in-tree
(`monai.apps.generation.maisi.networks.{autoencoderkl_maisi, diffusion_model_unet_maisi, controlnet_maisi}`
and `monai.networks.schedulers.RFlowScheduler`).

The trainers additionally import two helpers from the NVIDIA MAISI repository
(`maisi/scripts/utils.py`: `define_instance`, `binarize_labels`; `maisi/scripts/diff_model_setting.py`: `load_config`)
and read MAISI config files (`config_network_rflow.json`, `modality_mapping.json`). Clone it **next to this
repository** — the scripts resolve `<repo_root>/maisi` relative to their own location:

```bash
git clone https://github.com/NVIDIA/maisi.git maisi
```

## Usage sketch

```bash
# 1. build a ControlNet training set (label-conditioned latents) from your data
TS_ROOT=/path/to/TotalSegmentator python scripts/build_controlnet_datasets.py

# 2. train one ControlNet expert (one label group)
python -u scripts/train_controlnet_custom.py \
  --env-config configs/env_your_dataset.json \
  --model-config configs/config_smoke_2ep.json \
  --network-config configs/config_network_rflow_4ch.json

# 3. inpainting expert variant
python -u scripts/train_controlnet_inpaint.py ...

# 4. compose K experts and sample with the FKC-PoE sampler
python scripts/infer_fkc_poe_pilot.py ...
```

`configs/config_network_*.json` define the backbone + conditioning encodings used in this project
(4-, 7-, 15-, 22-, 43- and 45-channel condition masks); `configs/config_*.json` are training-run presets
(epochs / lr / CFG dropout). The `env_*` dataset configs from the original project are machine-specific
and intentionally not shipped — write your own pointing at your data.

## Notes & caveats

- No datasets, checkpoints or pretrained weights are included. Evaluation cohorts used in the
  original project: TotalSegmentator, AMOS-22, MMWHS.
- `src/fov_training_proof.py` pins a run-contract target hostname (`FARM04`) — an internal cluster node.
  Adapt or bypass `validate_execution_environment_record` when porting.
- Everything was run single-seed on internal hardware; treat numbers as reproducibility anchors,
  not benchmarks.

## License

Apache-2.0. This project builds on [MONAI](https://github.com/Project-MONAI/MONAI) and
[NVIDIA MAISI](https://github.com/NVIDIA/maisi), both Apache-2.0.
