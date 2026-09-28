"""FKC-PoE-N sampler for K independent ControlNet experts on a shared MAISI v2 backbone.

Implements 4 composition modes (selectable at eval time):

  ┌─────────────────┬────────────────┬───────────────────┐
  │ mode            │ velocity comp. │ FK weight increm. │
  ├─────────────────┼────────────────┼───────────────────┤
  │ naive_mean_off  │ mean(v_k)      │ 0 (no FK)         │
  │ naive_mean_fkc  │ mean(v_k)      │ sum_{i<j}<v_i,v_j>/D dt │
  │ cfg_poe_off     │ v_un + sum_k(v_k-v_un) │ 0          │
  │ cfg_poe_fkc     │ v_un + sum_k(v_k-v_un) │ sum_{i<j}<v_i-v_un, v_j-v_un>/D dt │
  │ residual_sum   │ v = UNet(z, Σ CN_residuals)             │ N/A (no FK)   │
  └─────────────────┴────────────────┴───────────────────┘

Particles (K_SMC) run as a batch dim in MAISI's UNet/ControlNet forward passes.
ESS is logged per step; if ESS < K_SMC / 2 inside the active window, SMC systematic
resampling is performed.

Reference for FK-PoE theory:
  Skreta et al., "Feynman-Kac Correctors in Diffusion", ICML 2025, Prop. 3.3.
  Adapted: per-step weight increment for K experts is
      d log w = beta * (1/D) * sum_{i<j} <s_i, s_j> * dt
  where s_i is the score-proxy velocity for expert i (= v_i for naive_mean,
  = v_i - v_un for cfg_poe).

The samplers expose `sample(...)` returning final latents (which the caller decodes
through MAISI's VAE) and per-step diagnostics (ESS curve, FK weight curve).

Logical caveats baked in (see memory/LIMITS.md):
  - LIM-POE-FORMULA: naive_mean over-counts the prior by (K-1); cfg_poe is the
    principled form. Both implemented; sweep at eval time.
  - LIM-PARTIAL-LABEL-BG-AMBIGUITY: experts disagree on each other's vocab
    positions (each sees "background-0" there). If <v_A, v_B> is negative
    everywhere, FKC weights collapse to single-expert; the ESS metric reveals it.
  - LIM-FKC-K-MAISI: K_SMC=4 inherited from previous experiment; may be below
    FK theory's quasi-asymptotic regime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# =============================================================================
# Helpers
# =============================================================================


def _systematic_resample(particles: torch.Tensor, log_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """SMC systematic resampling.

    particles: (K, ...)
    log_w:     (K,)
    Returns resampled (K, ...) particles + zeroed log_w of length K.
    """
    K = log_w.shape[0]
    device = log_w.device
    w = F.softmax(log_w, dim=0)
    cumsum = w.cumsum(dim=0)
    u0 = torch.rand(1, device=device) / K
    grid = u0 + torch.arange(K, device=device).float() / K  # (K,)
    idx = torch.searchsorted(cumsum, grid, right=False).clamp(max=K - 1)
    resampled = particles[idx]
    return resampled, torch.zeros_like(log_w)


def _ess(log_w: torch.Tensor) -> torch.Tensor:
    """Effective sample size from log-weights."""
    w = F.softmax(log_w, dim=0)
    return 1.0 / (w * w).sum()


def _fk_weight_increment(velocities_or_deltas: List[torch.Tensor], dt: float, beta: float = 1.0) -> torch.Tensor:
    """Generalized FK weight increment for N experts.

    velocities_or_deltas: list of N tensors, each shaped (K_SMC, C, D, H, W).
      For naive_mean: pass v_k directly.
      For cfg_poe:    pass v_k - v_uncond (the conditional deltas).

    Returns per-particle weight increment: (K_SMC,)
    """
    N = len(velocities_or_deltas)
    if N < 2:
        return torch.zeros(velocities_or_deltas[0].shape[0], device=velocities_or_deltas[0].device)

    # Flatten spatial dims for inner products
    K_SMC = velocities_or_deltas[0].shape[0]
    D = velocities_or_deltas[0][0].numel()  # per-particle dim
    flat = [v.reshape(K_SMC, -1) for v in velocities_or_deltas]

    ip = torch.zeros(K_SMC, device=flat[0].device)
    for i in range(N):
        for j in range(i + 1, N):
            ip = ip + (flat[i] * flat[j]).sum(dim=-1) / D
    return beta * ip * dt


# =============================================================================
# ControlNet expert runner (shared MAISI backbone, one ControlNet head per expert)
# =============================================================================


@dataclass
class ControlNetExpert:
    """One partial-label expert: trained ControlNet weights + its conditioning."""
    name: str                                  # "expert_A_abd" etc
    controlnet: torch.nn.Module                # loaded MAISI ControlNet module
    controlnet_cond: torch.Tensor              # binarized one-hot mask, shape (1, n_classes, D, H, W) at latent res

    def run_one_step(
        self,
        latents: torch.Tensor,         # (K_SMC, C, D_lat, H_lat, W_lat)
        t: float,
        modality_tensor: torch.Tensor, # (K_SMC,) int
        include_modality: bool,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Returns ControlNet's (down_block_residuals, mid_block_residual)."""
        K_SMC = latents.shape[0]
        controlnet_inputs = {
            "x": latents,
            "timesteps": torch.full((K_SMC,), t, device=latents.device, dtype=latents.dtype),
            "controlnet_cond": self.controlnet_cond.expand(K_SMC, *self.controlnet_cond.shape[1:]),
        }
        if include_modality:
            controlnet_inputs["class_labels"] = modality_tensor
        down, mid = self.controlnet(**controlnet_inputs)
        return down, mid


@dataclass
class SamplerConfig:
    K_SMC: int = 4
    n_steps: int = 30
    beta: float = 1.0
    fkc_eta: float = 0.0
    resample_t_lo: float = 0.0     # in [0, 1] normalized time (0 = data, 1 = noise)
    resample_t_hi: float = 0.8
    ess_threshold_frac: float = 0.5
    composition: str = "cfg_poe"   # "naive_mean" | "cfg_poe" | "residual_sum"
    use_fkc_weights: bool = True
    seed: int = 1234
    cfg_scale: float = 1.0         # CFG-style guidance scale: v_uncond + cfg_scale * sum(deltas). 1.0 = original.
    cfg_scale_per_expert: Optional[List[float]] = None  # asymmetric per-expert guidance: v_uncond + sum(cfg_k * delta_k). Overrides cfg_scale when set; order matches `experts`.
    # ----- auto per-expert guidance from localization confidence (own/foreign increment ratio) -----
    auto_guidance: bool = False                         # if set (with own/foreign masks + auto_rho_base), s_k derived per expert
    auto_s_max: float = 1.5                             # guidance for a confidently in-dist expert (confidence c_k=1)
    auto_tau_lo: float = 0.8                            # rho_test/rho_base at/below which expert is treated OOD -> s_k=1.0
    auto_rho_base: Optional[List[float]] = None         # per-expert in-dist own/foreign baseline (order matches `experts`)
    # ----- soft-pull "expert-repaint": in each expert's own region, blend v_composed toward that expert's velocity -----
    # v_composed += gamma * sum_k own_mask_k * (v_k - v_composed).  Seam-free soft regional coupling; equals the
    # position-pull toward each expert's own-region x0 solution up to a tau factor on gamma (so it is tau-free).
    # 0 = off (default, no change). Requires own_masks. gamma in ~[0.05, 0.5].
    soft_pull_gamma: float = 0.0
    # ----- A1: annealed-Langevin predictor-corrector (Lever A; breaks the PoE-approximation wall) -----
    # After each ODE (predictor) step, run corrector_M Langevin steps at the NEW level using the
    # composed score derived from RFlow v-prediction: score = -(x - (1-s)*v_comp)/s,  s = t/T.
    # Update: x += 0.5*eta*score + sqrt(eta)*z, eta = corrector_c * s^2.  M=0 => OFF (no-op, identical
    # to the base sampler). Applied only for normalized level s in [corrector_s_lo, corrector_s_hi].
    corrector_M: int = 0
    corrector_c: float = 0.05
    corrector_s_lo: float = 0.2
    corrector_s_hi: float = 0.8
    corrector_snr: float = 0.0     # A1: if >0, SNR-adaptive step (Song PC): eta=2(snr*||z||/||score||)^2, overrides corrector_c
    # ----- B0: per-expert unconditional (Lever B, no retraining): d_k = v_ck - v_uncond_k (each from its OWN net) -----
    per_expert_uncond: bool = False


@dataclass
class SamplerDiagnostics:
    ess_curve: List[float] = field(default_factory=list)
    logw_min_curve: List[float] = field(default_factory=list)
    logw_max_curve: List[float] = field(default_factory=list)
    resample_events: List[int] = field(default_factory=list)   # step indices where SMC fired
    velocity_inner_product_curve: List[float] = field(default_factory=list)
    expert_region_mag: List[List[float]] = field(default_factory=list)   # per step: [own-region mean ||d_k|| per expert]
    expert_full_mag: List[List[float]] = field(default_factory=list)     # per step: [full-volume mean ||d_k|| per expert]


# =============================================================================
# Main sampler
# =============================================================================


@torch.no_grad()
def sample_fkc_poe_n(
    experts: List[ControlNetExpert],
    diffusion_unet: torch.nn.Module,
    noise_scheduler,
    scale_factor: float,
    spacing_tensor: torch.Tensor,         # (K_SMC, 3)
    modality_tensor: torch.Tensor,        # (K_SMC,)
    latent_shape: Tuple[int, int, int, int],  # (C, D, H, W) for one particle
    config: SamplerConfig,
    top_region_index_tensor: Optional[torch.Tensor] = None,
    bottom_region_index_tensor: Optional[torch.Tensor] = None,
    include_modality: bool = True,
    include_body_region: bool = False,
    uncond_controlnet_cond: Optional[torch.Tensor] = None,   # required if composition == "cfg_poe"
    region_masks: Optional[List[torch.Tensor]] = None,       # per-region latent-res masks for confidence diagnostics (matrix expert x region)
    own_masks: Optional[List[torch.Tensor]] = None,          # auto_guidance: per-expert own (organ) latent mask
    foreign_masks: Optional[List[torch.Tensor]] = None,      # auto_guidance: per-expert foreign (ID-200 body) latent mask
) -> Tuple[torch.Tensor, torch.Tensor, SamplerDiagnostics]:
    """Run FKC-PoE-N sampling.

    Returns:
      particles: (K_SMC, C, D, H, W)  final latent particles
      log_w:     (K_SMC,)             final SNIS log-weights
      diag:      SamplerDiagnostics    per-step traces
    """
    device = next(diffusion_unet.parameters()).device
    dtype = next(diffusion_unet.parameters()).dtype
    N = len(experts)
    K_SMC = config.K_SMC

    if config.composition == "cfg_poe" and uncond_controlnet_cond is None:
        raise ValueError("cfg_poe composition requires uncond_controlnet_cond (binarized all-zero mask).")

    g = torch.Generator(device=device); g.manual_seed(config.seed)
    latents = torch.randn(K_SMC, *latent_shape, device=device, dtype=dtype, generator=g)
    log_w = torch.zeros(K_SMC, device=device)
    diag = SamplerDiagnostics()

    # Build the timesteps via the scheduler (Rectified Flow / DDPM)
    if hasattr(noise_scheduler, "set_timesteps"):
        # RFlowScheduler needs input_img_size_numel = product of spatial dims of latent
        c, x, y, z = latent_shape
        latent_numel = int(x * y * z)
        try:
            noise_scheduler.set_timesteps(config.n_steps, input_img_size_numel=latent_numel)
        except TypeError:
            noise_scheduler.set_timesteps(config.n_steps)
    all_t = noise_scheduler.timesteps.to(device).tolist()
    # Pair t with next_t for RFlow's two-arg step API; for DDPM next_t is None
    next_ts = list(all_t[1:]) + [0]

    # Build the "uncond" expert (zeros mask) once if needed
    uncond_expert = None
    if config.composition == "cfg_poe":
        uncond_expert = ControlNetExpert(
            name="uncond",
            controlnet=experts[0].controlnet,         # share weights (zero mask works on any expert)
            controlnet_cond=uncond_controlnet_cond,
        )

    for step_idx, (t, next_t) in enumerate(zip(all_t, next_ts)):
        # ----- per-expert ControlNet forward (collect residuals) -----
        all_down: List[List[torch.Tensor]] = []  # [expert_idx][level] -> Tensor
        all_mid: List[torch.Tensor] = []
        velocities: List[torch.Tensor] = []

        for expert in experts:
            down, mid = expert.run_one_step(latents, t, modality_tensor, include_modality)
            all_down.append(down)
            all_mid.append(mid)

        if config.composition == "residual_sum":
            # Sum ControlNet residuals → single UNet forward
            down_sum = [sum(level) for level in zip(*all_down)]
            mid_sum = sum(all_mid)
            v_composed = _unet_velocity(
                latents, t, spacing_tensor, modality_tensor,
                top_region_index_tensor, bottom_region_index_tensor,
                down_sum, mid_sum, diffusion_unet, include_modality, include_body_region,
            )
            fk_source = None  # residual_sum has no per-expert velocities for FK

        elif config.composition == "cfg_poe" or config.composition == "naive_mean":
            # Per-expert UNet forward (existing behavior)
            for ei, expert in enumerate(experts):
                v = _unet_velocity(
                    latents, t, spacing_tensor, modality_tensor,
                    top_region_index_tensor, bottom_region_index_tensor,
                    all_down[ei], all_mid[ei], diffusion_unet,
                    include_modality, include_body_region,
                )
                velocities.append(v)

            if config.composition == "cfg_poe":
                down_u, mid_u = uncond_expert.run_one_step(latents, t, modality_tensor, include_modality)
                v_uncond = _unet_velocity(
                    latents, t, spacing_tensor, modality_tensor,
                    top_region_index_tensor, bottom_region_index_tensor,
                    down_u, mid_u, diffusion_unet, include_modality, include_body_region,
                )
                if config.per_expert_uncond:
                    # B0: each delta uses its OWN net's uncond -> coherent deltas (fixes the cross-ref bug)
                    v_unconds = []
                    for ex in experts:
                        ue = ControlNetExpert(name=f"u_{ex.name}", controlnet=ex.controlnet,
                                              controlnet_cond=uncond_controlnet_cond)
                        du_k, mu_k = ue.run_one_step(latents, t, modality_tensor, include_modality)
                        v_unconds.append(_unet_velocity(latents, t, spacing_tensor, modality_tensor,
                                                        top_region_index_tensor, bottom_region_index_tensor,
                                                        du_k, mu_k, diffusion_unet, include_modality, include_body_region))
                    deltas = [v - vu for v, vu in zip(velocities, v_unconds)]
                    v_uncond = sum(v_unconds) / len(v_unconds)        # base = mean per-expert uncond
                else:
                    deltas = [v - v_uncond for v in velocities]
                # ----- per-expert confidence diagnostic: region-restricted increment magnitude -----
                # ||d_k|| per voxel = L2 over channels; aggregate over expert's own latent region.
                per_voxel_mag = [torch.sqrt((d ** 2).sum(dim=1) + 1e-12) for d in deltas]  # each (K_SMC, D, H, W)
                diag.expert_full_mag.append([float(m.mean().detach().cpu()) for m in per_voxel_mag])
                if region_masks is not None:
                    # full [expert][region] magnitude matrix: own = diag[k][k], foreign = diag[k][other]
                    mat = []
                    for mk in per_voxel_mag:
                        erow = []
                        for rm in region_masks:
                            rm_b = rm.to(mk.device, mk.dtype)
                            denom = rm_b.sum().clamp(min=1.0)
                            erow.append(float(((mk * rm_b).sum() / denom).detach().cpu()))
                        mat.append(erow)
                    diag.expert_region_mag.append(mat)
                if config.auto_guidance and own_masks is not None and foreign_masks is not None \
                        and config.auto_rho_base is not None:
                    # auto per-expert guidance s_k from localization confidence (own/foreign increment ratio)
                    s_list = []
                    s_max, tau = config.auto_s_max, config.auto_tau_lo
                    for k, mk in enumerate(per_voxel_mag):
                        om = own_masks[k].to(mk.device, mk.dtype); fm = foreign_masks[k].to(mk.device, mk.dtype)
                        own_mag = (mk * om).sum() / om.sum().clamp(min=1.0)
                        for_mag = (mk * fm).sum() / fm.sum().clamp(min=1.0)
                        rho = float((own_mag / for_mag.clamp(min=1e-6)).detach().cpu())
                        c = (rho / max(config.auto_rho_base[k], 1e-6) - tau) / max(1.0 - tau, 1e-6)
                        c = min(1.0, max(0.0, c))
                        s_list.append(1.0 + (s_max - 1.0) * c)
                    diag.expert_region_mag.append(s_list)   # reuse trace slot to log per-step s_k
                    v_composed = v_uncond + sum(s * d for s, d in zip(s_list, deltas))
                elif config.cfg_scale_per_expert is not None:
                    if len(config.cfg_scale_per_expert) != len(deltas):
                        raise ValueError(
                            f"cfg_scale_per_expert has {len(config.cfg_scale_per_expert)} entries "
                            f"but there are {len(deltas)} experts"
                        )
                    v_composed = v_uncond + sum(
                        s * d for s, d in zip(config.cfg_scale_per_expert, deltas)
                    )
                else:
                    v_composed = v_uncond + config.cfg_scale * sum(deltas)
                fk_source = deltas
            else:  # naive_mean
                v_composed = sum(velocities) / N
                fk_source = velocities
        else:
            raise ValueError(f"Unknown composition: {config.composition}")

        # ----- soft-pull "expert-repaint": soft regional coupling toward each expert's own velocity -----
        if config.soft_pull_gamma > 0.0 and own_masks is not None and len(velocities) == len(own_masks):
            for k, vk in enumerate(velocities):
                Rk = own_masks[k].to(v_composed.device, v_composed.dtype)[None, None]  # (1,1,D,H,W) broadcast over K,C
                v_composed = v_composed + config.soft_pull_gamma * Rk * (vk - v_composed)

        # ----- FK weight increment -----
        dt_step = 1.0 / config.n_steps
        if config.use_fkc_weights and fk_source is not None and N >= 2:
            inc = _fk_weight_increment(fk_source, dt_step, beta=config.beta)
            log_w = log_w + inc
            log_w = log_w - log_w.mean()           # numerical stability
            diag.velocity_inner_product_curve.append(float(inc.mean().detach().cpu()))
        else:
            diag.velocity_inner_product_curve.append(0.0)

        # ----- ODE step (scheduler) -----
        if next_t is None or (isinstance(next_t, int) and next_t == 0 and step_idx == len(all_t) - 1):
            latents, _ = noise_scheduler.step(v_composed, t, latents)
        else:
            latents, _ = noise_scheduler.step(v_composed, t, latents, next_t)

        # ----- A1 annealed-Langevin corrector at the NEW level (additive; only when corrector_M>0) -----
        if config.corrector_M > 0 and config.composition == "cfg_poe" and uncond_expert is not None:
            nt = next_t if next_t is not None else 0
            s_norm = float(nt) / float(noise_scheduler.num_train_timesteps)
            if config.corrector_s_lo <= s_norm <= config.corrector_s_hi:
                for _ in range(config.corrector_M):
                    vs_c = []
                    for ex in experts:
                        d_c, m_c = ex.run_one_step(latents, nt, modality_tensor, include_modality)
                        vs_c.append(_unet_velocity(latents, nt, spacing_tensor, modality_tensor,
                                                   top_region_index_tensor, bottom_region_index_tensor,
                                                   d_c, m_c, diffusion_unet, include_modality, include_body_region))
                    du_c, mu_c = uncond_expert.run_one_step(latents, nt, modality_tensor, include_modality)
                    vu_c = _unet_velocity(latents, nt, spacing_tensor, modality_tensor,
                                          top_region_index_tensor, bottom_region_index_tensor,
                                          du_c, mu_c, diffusion_unet, include_modality, include_body_region)
                    v_c = vu_c + config.cfg_scale * sum(v - vu_c for v in vs_c)
                    eps_est = latents - (1.0 - s_norm) * v_c          # eps = x_s - (1-s)*v
                    score = -eps_est / s_norm                          # grad log p_s(x)
                    z = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=g)
                    if config.corrector_snr > 0:                       # Song-style SNR-adaptive step (balances drift vs noise)
                        eta_s = 2.0 * (config.corrector_snr * z.norm() / (score.norm() + 1e-8)) ** 2
                        latents = latents + eta_s * score + torch.sqrt(2.0 * eta_s) * z
                    else:                                              # fixed eta = c*s^2
                        eta = config.corrector_c * (s_norm ** 2)
                        latents = latents + 0.5 * eta * score + math.sqrt(eta) * z

        if config.fkc_eta > 0:
            latents = latents + config.fkc_eta * torch.randn_like(latents) * math.sqrt(dt_step)

        # ----- ESS + SMC resampling -----
        ess = float(_ess(log_w).detach().cpu())
        diag.ess_curve.append(ess)
        diag.logw_min_curve.append(float(log_w.min().detach().cpu()))
        diag.logw_max_curve.append(float(log_w.max().detach().cpu()))

        # Normalize "t" to [0,1] for the active resample window check
        t_norm = float(t) / float(all_t[0]) if all_t[0] != 0 else 0.0
        in_active_window = config.resample_t_lo <= t_norm <= config.resample_t_hi
        if (config.use_fkc_weights and config.composition != "naive_mean_offhandled"
                and in_active_window and ess < config.ess_threshold_frac * K_SMC):
            latents, log_w = _systematic_resample(latents, log_w)
            diag.resample_events.append(step_idx)

    return latents, log_w, diag


def _unet_velocity(
    latents, t, spacing_tensor, modality_tensor,
    top_region_index_tensor, bottom_region_index_tensor,
    down, mid, diffusion_unet, include_modality, include_body_region,
):
    """Wrap MAISI's UNet forward with explicit residuals from a ControlNet."""
    K_SMC = latents.shape[0]
    unet_inputs = {
        "x": latents,
        "timesteps": torch.full((K_SMC,), t, device=latents.device, dtype=latents.dtype),
        "spacing_tensor": spacing_tensor,
        "down_block_additional_residuals": down,
        "mid_block_additional_residual": mid,
    }
    if include_body_region:
        unet_inputs["top_region_index_tensor"] = top_region_index_tensor
        unet_inputs["bottom_region_index_tensor"] = bottom_region_index_tensor
    if include_modality:
        unet_inputs["class_labels"] = modality_tensor
    return diffusion_unet(**unet_inputs)


# =============================================================================
# SNIS final-particle selection
# =============================================================================


def snis_weights(log_w: torch.Tensor) -> torch.Tensor:
    """Self-normalized importance sampling weights from log_w."""
    return F.softmax(log_w, dim=0)


def snis_weighted_average(particles: torch.Tensor, log_w: torch.Tensor) -> torch.Tensor:
    """SNIS-weighted average of particles. particles: (K, ...). Returns (...)"""
    w = snis_weights(log_w)
    shape = [particles.shape[0]] + [1] * (particles.dim() - 1)
    return (particles * w.reshape(shape)).sum(dim=0)


# =============================================================================
# Spatial-aware FKC building blocks (SA-PFE-EMRF Stage S1.3)
# =============================================================================
#
# Three pieces:
#   * EndpointEstimator    -- z_t + (1-t)*v_t -> z_1_hat, with optional decode.
#   * SpatialEnergyEvaluator -- E_sp = E_obs + lambda_anat * E_anat on a soft mask.
#                            Multi-scale + ROI-block decomposition for tex Def 6.5.
#   * ROIBlockResampler    -- local-block weights + resample-move (Def 6.5 strategy 2+3).
#
# Designed to be plug-in: the existing sample_fkc_poe_n() function is unchanged;
# a new sample_sa_fkc_poe() composes the building blocks with the same expert
# infrastructure.


def rf_endpoint_estimate(z_t: torch.Tensor, t_norm: float, v_t: torch.Tensor) -> torch.Tensor:
    """Rectified-flow endpoint estimate z_1_hat = z_t + (1-t)*v_t.

    Args
    ----
      z_t    : (K_SMC, C, D, H, W) current latent
      t_norm : float in [0,1], with t=1 = noise, t=0 = data (RF convention).
               Note MAISI's scheduler often uses absolute timesteps; the caller
               must convert to normalized [0,1] (= raw_t / max_t).
      v_t    : same shape as z_t, current velocity estimate.

    Returns z_1_hat of same shape.
    """
    return z_t + (1.0 - float(t_norm)) * v_t


@dataclass
class SpatialFKCConfig:
    """Knobs for the spatial-aware FKC weight increment + resampling."""

    # Energy-term weights (applied multiplicatively before exp).
    w_obs: float = 1.0
    w_anat: float = 1.0
    w_reg: float = 0.0  # registration energy, used at Stage 2 only

    # Inverse-temperature ramp; beta(step) = beta_lo + (beta_hi - beta_lo) * step/N.
    beta_lo: float = 0.0
    beta_hi: float = 1.0

    # ROI block decomposition; if n_blocks == 1, behaves like global FKC.
    n_blocks_per_axis: int = 1  # gives n_blocks_per_axis**3 ROI blocks total

    # Resample-move:
    ess_threshold_frac: float = 0.5
    do_resample: bool = True
    resample_t_lo: float = 0.0  # normalized-time window
    resample_t_hi: float = 0.8

    # Coarse-to-fine ROI weighting: at step fraction <= coarse_until, weight blocks
    # by their *uniform* mean (organ layout); afterwards weight by per-block fine
    # energy (boundary, local consistency).
    coarse_until: float = 0.5

    # If True, run anatomy_prior gradient w.r.t. M_soft and translate into a
    # latent-space velocity correction (Theorem 6.4).  Stage-1 default False.
    use_velocity_correction: bool = False
    gamma_t: float = 0.1


class SpatialEnergyEvaluator:
    """E_sp(M_soft, partial_obs) = w_obs*E_obs + w_anat*E_anat (+ w_reg*E_reg).

    The evaluator is stateless wrt particles; one instance per case.  Caller is
    responsible for keeping M_soft per particle aligned with the partial obs.

    M_soft : (B, K, D, H, W) soft mask in [0,1] -- typically the aggregator's
             posterior mean, optionally refined per particle.
    partial_obs        : (B, K, D, H, W) one-hot partial mask (zeros where
             not observed).  Loss is masked by presence_mask.
    presence_mask      : (B, K, D, H, W) per-channel observed flag.
    """

    def __init__(
        self,
        anatomy_prior,                 # src.anatomy_prior.AnatomyPrior
        w_obs: float = 1.0,
        w_anat: float = 1.0,
        eps: float = 1e-6,
    ):
        self.anatomy_prior = anatomy_prior
        self.w_obs = float(w_obs)
        self.w_anat = float(w_anat)
        self.eps = float(eps)

    @staticmethod
    def _soft_dice(pred: torch.Tensor, target: torch.Tensor, presence: torch.Tensor, eps: float) -> torch.Tensor:
        """1 - Dice over (D,H,W) per (B, K), then mean across observed channels.

        Returns (B,) scalar per batch element.
        """
        # Reduce per (B,K) over voxels.
        inter = (pred * target).flatten(2).sum(dim=2)
        s_pred = pred.flatten(2).sum(dim=2)
        s_targ = target.flatten(2).sum(dim=2)
        dice = (2.0 * inter + eps) / (s_pred + s_targ + eps)
        loss = 1.0 - dice  # (B, K)
        # Channel-level presence: any observation at all on this channel?
        ch_obs = presence.amax(dim=(2, 3, 4))  # (B, K) in {0,1}
        denom = torch.clamp(ch_obs.sum(dim=1), min=1.0)
        return (loss * ch_obs).sum(dim=1) / denom  # (B,)

    def e_obs(self, M_soft: torch.Tensor, partial_obs: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        """Spatial observation energy = soft-Dice on observed channels."""
        return self._soft_dice(M_soft, partial_obs, presence, self.eps)

    def e_anat(self, M_soft: torch.Tensor) -> torch.Tensor:
        """Forward AnatomyPrior; returns (B,) scalar per batch element."""
        return self.anatomy_prior(M_soft)

    def __call__(
        self,
        M_soft: torch.Tensor,
        partial_obs: torch.Tensor,
        presence: torch.Tensor,
        return_breakdown: bool = False,
    ):
        e_o = self.e_obs(M_soft, partial_obs, presence)
        e_a = self.e_anat(M_soft)
        total = self.w_obs * e_o + self.w_anat * e_a
        if return_breakdown:
            return total, {"e_obs": e_o.detach(), "e_anat": e_a.detach()}
        return total


class ROIBlockResampler:
    """Block-local SMC resampling (tex Def 6.5 strategy 2).

    Splits the latent volume into n_blocks_per_axis**3 ROI blocks.  Each block
    can be resampled independently, which reduces global weight collapse on
    high-dimensional 3D volumes.

    NOTE: per-block resampling rewires *particles*, so neighbouring blocks may
    end up taken from different parent particles -- this is the price for
    avoiding global collapse.  For Stage 1 we default n_blocks=1 (global SMC,
    same as the existing sampler).
    """

    def __init__(self, n_blocks_per_axis: int = 1):
        self.nb = max(1, int(n_blocks_per_axis))

    @staticmethod
    def _block_slices(size: int, nb: int) -> list[tuple[int, int]]:
        edges = [round(size * i / nb) for i in range(nb + 1)]
        return [(edges[i], edges[i + 1]) for i in range(nb)]

    def resample(self, particles: torch.Tensor, log_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """particles: (K_SMC, C, D, H, W); log_w: (K_SMC,)  OR  (K_SMC, n_blocks).

        If log_w is per-particle (1D), behaves like global systematic resampling.
        If log_w is per-block (2D), each block is resampled with its own weights.
        """
        if self.nb == 1 or log_w.dim() == 1:
            return _systematic_resample(particles, log_w if log_w.dim() == 1 else log_w.mean(dim=1))
        K_SMC = particles.shape[0]
        _, C, D, H, W = particles.shape
        slices_d = self._block_slices(D, self.nb)
        slices_h = self._block_slices(H, self.nb)
        slices_w = self._block_slices(W, self.nb)
        out = particles.clone()
        b = 0
        for sd in slices_d:
            for sh in slices_h:
                for sw in slices_w:
                    w_block = log_w[:, b]
                    new_part, _ = _systematic_resample(particles[:, :, sd[0]:sd[1], sh[0]:sh[1], sw[0]:sw[1]], w_block)
                    out[:, :, sd[0]:sd[1], sh[0]:sh[1], sw[0]:sw[1]] = new_part
                    b += 1
        return out, torch.zeros_like(log_w)


def spatial_fkc_weight_increment(
    M_soft_per_particle: torch.Tensor,
    partial_obs: torch.Tensor,
    presence: torch.Tensor,
    evaluator: SpatialEnergyEvaluator,
    delta_beta: float,
) -> torch.Tensor:
    """Per-particle weight increment d_log_w_i = -delta_beta * E_sp(M_soft_i, obs).

    Args
    ----
      M_soft_per_particle : (K_SMC, K, D, H, W) per-particle soft mask estimate
      partial_obs         : (1, K, D, H, W) shared partial obs (broadcast)
      presence            : (1, K, D, H, W) shared presence mask
      evaluator           : SpatialEnergyEvaluator
      delta_beta          : Delta_beta_n step in the tempering schedule

    Returns (K_SMC,) tensor of log-weight increments.
    """
    K_SMC = M_soft_per_particle.shape[0]
    obs = partial_obs.expand_as(M_soft_per_particle)
    pres = presence.expand_as(M_soft_per_particle)
    e = evaluator(M_soft_per_particle, obs, pres)  # (K_SMC,)
    return -float(delta_beta) * e


@torch.no_grad()
def sample_sa_fkc_poe(
    experts: List[ControlNetExpert],
    diffusion_unet: torch.nn.Module,
    noise_scheduler,
    scale_factor: float,
    spacing_tensor: torch.Tensor,
    modality_tensor: torch.Tensor,
    latent_shape: Tuple[int, int, int, int],
    config: SamplerConfig,
    spatial_config: SpatialFKCConfig,
    M_soft_per_particle: torch.Tensor,         # (K_SMC, K, D, H, W) per-particle mask
    partial_obs: torch.Tensor,                 # (1, K, D, H, W)
    presence: torch.Tensor,                    # (1, K, D, H, W)
    spatial_evaluator: "SpatialEnergyEvaluator",
    top_region_index_tensor: Optional[torch.Tensor] = None,
    bottom_region_index_tensor: Optional[torch.Tensor] = None,
    include_modality: bool = True,
    include_body_region: bool = False,
    uncond_controlnet_cond: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, SamplerDiagnostics]:
    """Spatial-aware FKC-PoE sampler (tex Algorithm 3, Stage-1 form).

    Differences from ``sample_fkc_poe_n``:
      * Each particle carries its own posterior mask sample (M_soft_per_particle)
        and its own ControlNet condition (already encoded into the supplied
        ``experts``; caller is responsible for binding expert.controlnet_cond
        to a (K_SMC, n_classes, D, H, W) tensor whose batch axis is the
        particle axis).
      * Weight update uses spatial-energy potential rather than velocity inner
        product:  Δlog w_i = -Δβ_n · E_sp(M^{(i)}, obs).
      * Tempering ramp Δβ_n = (β_hi - β_lo)/N applied linearly across steps.
      * Resampling uses the ROI-block resampler (defaults to global).

    Stage-1 simplification: E_sp is computed once from the supplied per-particle
    soft mask and does not see the running latent.  Full Algorithm 3 (per-step
    decode of Z_1_hat to a predicted mask, then energy on the predicted mask) is
    a Stage-2 follow-up; this is the "no use_velocity_correction" branch.
    """
    device = next(diffusion_unet.parameters()).device
    dtype = next(diffusion_unet.parameters()).dtype
    K_SMC = config.K_SMC

    if config.composition == "cfg_poe" and uncond_controlnet_cond is None:
        raise ValueError("cfg_poe composition requires uncond_controlnet_cond.")

    # Pre-compute per-particle spatial energy (tex E_sp), shape (K_SMC,)
    e_sp = spatial_evaluator(
        M_soft_per_particle,
        partial_obs.expand_as(M_soft_per_particle),
        presence.expand_as(M_soft_per_particle),
    )  # (K_SMC,)

    g = torch.Generator(device=device); g.manual_seed(config.seed)
    latents = torch.randn(K_SMC, *latent_shape, device=device, dtype=dtype, generator=g)
    log_w = torch.zeros(K_SMC, device=device)
    diag = SamplerDiagnostics()

    if hasattr(noise_scheduler, "set_timesteps"):
        c, x, y, z = latent_shape
        latent_numel = int(x * y * z)
        try:
            noise_scheduler.set_timesteps(config.n_steps, input_img_size_numel=latent_numel)
        except TypeError:
            noise_scheduler.set_timesteps(config.n_steps)
    all_t = noise_scheduler.timesteps.to(device).tolist()
    next_ts = list(all_t[1:]) + [0]

    uncond_expert = None
    if config.composition == "cfg_poe":
        uncond_expert = ControlNetExpert(
            name="uncond",
            controlnet=experts[0].controlnet,
            controlnet_cond=uncond_controlnet_cond,
        )

    block_resampler = ROIBlockResampler(spatial_config.n_blocks_per_axis)
    n_steps = max(len(all_t), 1)

    for step_idx, (t, next_t) in enumerate(zip(all_t, next_ts)):
        # ---- velocity composition (mirrors sample_fkc_poe_n) ----
        all_down: List[List[torch.Tensor]] = []
        all_mid: List[torch.Tensor] = []
        for expert in experts:
            down, mid = expert.run_one_step(latents, t, modality_tensor, include_modality)
            all_down.append(down)
            all_mid.append(mid)

        if config.composition == "residual_sum":
            down_sum = [sum(level) for level in zip(*all_down)]
            mid_sum = sum(all_mid)
            v_composed = _unet_velocity(
                latents, t, spacing_tensor, modality_tensor,
                top_region_index_tensor, bottom_region_index_tensor,
                down_sum, mid_sum, diffusion_unet, include_modality, include_body_region,
            )
        elif config.composition in ("cfg_poe", "naive_mean"):
            velocities: List[torch.Tensor] = []
            for ei in range(len(experts)):
                v = _unet_velocity(
                    latents, t, spacing_tensor, modality_tensor,
                    top_region_index_tensor, bottom_region_index_tensor,
                    all_down[ei], all_mid[ei], diffusion_unet,
                    include_modality, include_body_region,
                )
                velocities.append(v)
            if config.composition == "cfg_poe":
                down_u, mid_u = uncond_expert.run_one_step(latents, t, modality_tensor, include_modality)
                v_uncond = _unet_velocity(
                    latents, t, spacing_tensor, modality_tensor,
                    top_region_index_tensor, bottom_region_index_tensor,
                    down_u, mid_u, diffusion_unet, include_modality, include_body_region,
                )
                v_composed = v_uncond + sum(v - v_uncond for v in velocities)
            else:
                v_composed = sum(velocities) / len(experts)
        else:
            raise ValueError(f"Unknown composition: {config.composition}")

        # ---- spatial FKC weight update (tex Def 6.5) ----
        # Linear tempering: Δβ_n = (β_hi - β_lo) / N each step.
        delta_beta = (spatial_config.beta_hi - spatial_config.beta_lo) / n_steps
        log_w = log_w - delta_beta * e_sp
        log_w = log_w - log_w.mean()  # numerical stability

        # ---- ODE step ----
        if next_t is None or (isinstance(next_t, int) and next_t == 0 and step_idx == len(all_t) - 1):
            latents, _ = noise_scheduler.step(v_composed, t, latents)
        else:
            latents, _ = noise_scheduler.step(v_composed, t, latents, next_t)

        # ---- ESS + resample (active window) ----
        ess = float(_ess(log_w).detach().cpu())
        diag.ess_curve.append(ess)
        diag.logw_min_curve.append(float(log_w.min().detach().cpu()))
        diag.logw_max_curve.append(float(log_w.max().detach().cpu()))
        diag.velocity_inner_product_curve.append(float(e_sp.mean().detach().cpu()))

        t_norm = float(t) / float(all_t[0]) if all_t[0] != 0 else 0.0
        in_window = spatial_config.resample_t_lo <= t_norm <= spatial_config.resample_t_hi
        if (spatial_config.do_resample and in_window
                and ess < spatial_config.ess_threshold_frac * K_SMC and K_SMC > 1):
            latents, log_w = block_resampler.resample(latents, log_w)
            diag.resample_events.append(step_idx)

    return latents, log_w, diag


# =============================================================================
# Shielded Local Expert Composition (MVP) — "屏蔽式局部专家组合"
# =============================================================================
#
# Motivation (2026-06-14): the contamination-clean cfg_poe result showed that
# composition HURTS the resident region (cfg_poe abd 0.338 < single_A 0.378).
# Diagnosis: through the shared UNet's receptive field, each expert "sees" the
# other expert's private zone in x_t, goes OOD, and emits bad guidance.
#
# This sampler implements 5 defences so an expert only ever works inside its own
# visible world and commits a *local conditional increment* to the base flow:
#   1. Private-zone shield   : x_t^A = (1-P_B) x_t + P_B x_t^base  (A never sees B's private zone)
#   2. Null-subtracted incr  : d_A = E_A(x^A,C_A) - E_A(x^A,∅)
#   3. Ownership gate + veto : g_A ← g_A (1-P_B); increment only inside A's turf
#   4. RMS budget            : ‖g_A d_A‖ ≤ ρ ‖v0‖   (expert can't out-shout base)
#   5. Time-sync projection  : v = v0 + u_perp + β u_parallel (mostly steer, don't fast-forward)
#
# Final:  v_joint = v0 + P_sync[ λ_A g_A (E_A(x^A,C_A)-E_A(x^A,∅))
#                              + λ_B g_B (E_B(x^B,C_B)-E_B(x^B,∅)) ]
# A base-only trajectory x_t^base is maintained alongside, evolving under v0.


@dataclass
class ShieldConfig:
    lambda_expert: float = 0.7      # λ_k — expert conditional strength
    rho: float = 0.5                # RMS budget: expert increment ≤ rho * RMS(v0)
    beta_sync: float = 0.1          # fraction of base-parallel component kept (0=pure steer)
    gate_floor: float = 1.0         # effective gate = floor + (1-floor)*g_k, then × foreign-veto.
                                    #   1.0 = pure foreign-zone veto (full own-region strength)
                                    #   0.0 = pure soft ownership (collapses guidance — diagnosed bad)
    use_shield: bool = True         # module 1: private-zone shield via base trajectory
    use_null_sub: bool = True       # module 2: null-subtracted increment (else raw expert velocity)
    use_gate: bool = True           # module 3: ownership gate + (optional) hard veto
    use_hard_veto: bool = True      # if False: soft ownership only (gk = own), no hard foreign zeroing
    norm_own: bool = False          # if True: renormalize gates so sum_k g_k -> 1 where any expert owns
                                    #   (guarantees full guidance in owned zones; avoids the gate_floor=0 collapse)
    use_budget: bool = True         # module 4: RMS budget
    use_time_sync: bool = True      # module 5: base time-sync projection
    # ---- uncertainty-aware precision-weighted PoE (spatial confidence composition) ----
    use_uncertainty: bool = False   # if True: w_k(x) = conf_k(x)^p / sum_j conf_j(x)^p, conf_k = ||d_k(x)|| (per-voxel
                                    #   guidance magnitude = how much expert k's condition moves the flow HERE = confidence).
                                    #   Generalizes naive PoE (p=0, uniform) and the ownership gate (geometric prior) into a
                                    #   MEASURED, spatial precision weighting. OOD/foreign experts emit ~0 increment -> auto-defer.
    unc_power: float = 2.0          # sharpness of precision weighting: 0 = naive uniform, large = winner-take-all
    unc_self_norm: bool = True      # normalize each expert's confidence map by its own spatial mean BEFORE the cross-
                                    #   expert competition. CRITICAL: removes the scale bias (large abd organs emit big
                                    #   increments, small vertebrae small ones) that would otherwise let abd dominate
                                    #   everywhere and collapse spine. Turns ||d_k|| into RELATIVE self-confidence.
    unc_smooth: int = 3             # avg-pool3d kernel to denoise the per-voxel confidence map (1 = off)
    unc_support_mix: float = 0.5    # robustifier vs "confidently-wrong OOD": tau_k *= (1-mix) + mix*support_k
                                    #   mix=0 = pure measured confidence; mix=1 = confidence zeroed outside expert's mask
    # ---- T2: regime-adaptive MC-variance composition (ABSOLUTE epistemic uncertainty) ----
    use_mc_uncertainty: bool = False  # u = sum_k c_k(x)*d_k  (ADDITIVE). c_k = outlier gate on per-voxel MC instability:
                                    #   ~1 at normal uncertainty (in-domain => reduces to naive additive PoE), ->0 ONLY where
                                    #   expert k's epistemic uncertainty is a spatial OUTLIER (genuinely OOD, e.g. AMOS abd
                                    #   expert in the spine zone). Distinguishes in-dist-quiet (keep) from OOD-wrong (suppress).
    mc_samples: int = 2             # J latent perturbations to estimate per-voxel velocity instability (epistemic proxy)
    mc_eps: float = 0.08            # latent perturbation scale (relative to randn)
    mc_kappa: float = 1.0           # outlier threshold in z-score units (c_k drops where instability_z > kappa)
    mc_sharp: float = 2.0           # sigmoid sharpness of the outlier gate
    seed: int = 1234
    n_steps: int = 30


def _base_velocity(uncond_expert, x, t, spacing_tensor, modality_tensor,
                   top_region_index_tensor, bottom_region_index_tensor,
                   diffusion_unet, include_modality, include_body_region):
    """Neutral base flow v0(x,t): UNet + ControlNet-on-zero-cond (the cfg_dropout-trained
    unconditional). This is the principled 'base manifold' velocity."""
    down_u, mid_u = uncond_expert.run_one_step(x, t, modality_tensor, include_modality)
    return _unet_velocity(x, t, spacing_tensor, modality_tensor,
                          top_region_index_tensor, bottom_region_index_tensor,
                          down_u, mid_u, diffusion_unet, include_modality, include_body_region)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean().sqrt()


@torch.no_grad()
def sample_shielded_local(
    experts: List[ControlNetExpert],          # exactly the partial-label experts [A, B, ...]
    diffusion_unet: torch.nn.Module,
    noise_scheduler,
    scale_factor: float,
    spacing_tensor: torch.Tensor,             # (1, 3)
    modality_tensor: torch.Tensor,            # (1,)
    latent_shape: Tuple[int, int, int, int],  # (C, D, H, W)
    uncond_controlnet_cond: torch.Tensor,     # zero one-hot mask for base + null-sub
    owner_gates: List[torch.Tensor],          # g_k: list of (1,1,D,H,W) soft ownership in [0,1] at latent res
    private_masks: List[torch.Tensor],        # P_k: list of (1,1,D,H,W) private-zone in [0,1] at latent res
    shield_cfg: ShieldConfig,
    top_region_index_tensor: Optional[torch.Tensor] = None,
    bottom_region_index_tensor: Optional[torch.Tensor] = None,
    include_modality: bool = True,
    include_body_region: bool = False,
    verbose: bool = False,
) -> Tuple[torch.Tensor, SamplerDiagnostics]:
    """Shielded Local Expert Composition (MVP). K_SMC=1 (no particles).

    owner_gates[k], private_masks[k] are latent-resolution (D,H,W) masks for expert k.
    For expert k, the foreign private zone shielded out = union of P_j for j != k.

    Returns (latents (1,C,D,H,W), diagnostics).
    """
    device = next(diffusion_unet.parameters()).device
    dtype = next(diffusion_unet.parameters()).dtype
    N = len(experts)
    assert N == len(owner_gates) == len(private_masks), "experts/gates/masks length mismatch"
    eps = 1e-6

    g = torch.Generator(device=device); g.manual_seed(shield_cfg.seed)
    x = torch.randn(1, *latent_shape, device=device, dtype=dtype, generator=g)
    xb = x.clone()  # base-only trajectory, same initial noise
    diag = SamplerDiagnostics()

    gates = [m.to(device=device, dtype=dtype) for m in owner_gates]
    privs = [m.to(device=device, dtype=dtype) for m in private_masks]
    if getattr(shield_cfg, "norm_own", False) and len(gates) > 1:
        gsum = sum(gates)                                   # renormalize so sum_k g_k -> 1 where any expert owns
        gates = [g / (gsum + 1e-3) for g in gates]          # (no-man stays ~0; owned zones get full guidance)
    # foreign private zone for expert k = max_{j!=k} P_j  (voxels owned by some other expert)
    foreign = []
    for k in range(N):
        if N == 1:
            foreign.append(torch.zeros_like(privs[0]))
        else:
            f = privs[0] * 0
            for j in range(N):
                if j != k:
                    f = torch.maximum(f, privs[j])
            foreign.append(f)

    uncond_expert = ControlNetExpert(name="uncond", controlnet=experts[0].controlnet,
                                     controlnet_cond=uncond_controlnet_cond)

    if hasattr(noise_scheduler, "set_timesteps"):
        c, xx, yy, zz = latent_shape
        try:
            noise_scheduler.set_timesteps(shield_cfg.n_steps, input_img_size_numel=int(xx * yy * zz))
        except TypeError:
            noise_scheduler.set_timesteps(shield_cfg.n_steps)
    all_t = noise_scheduler.timesteps.to(device).tolist()
    next_ts = list(all_t[1:]) + [0]

    def bvel(state, t):
        return _base_velocity(uncond_expert, state, t, spacing_tensor, modality_tensor,
                              top_region_index_tensor, bottom_region_index_tensor,
                              diffusion_unet, include_modality, include_body_region)

    for step_idx, (t, next_t) in enumerate(zip(all_t, next_ts)):
        v0 = bvel(x, t)                 # base velocity on joint trajectory
        v0b = bvel(xb, t)               # base velocity on base-only trajectory

        u = torch.zeros_like(x)
        d_list = []  # uncertainty path: collect raw increments, combine by precision after the loop
        inst_list = []  # MC path: per-voxel epistemic instability for each expert
        for k, expert in enumerate(experts):
            # ---- module 1: shield foreign private zone with base trajectory ----
            if shield_cfg.use_shield and N > 1:
                x_k = (1.0 - foreign[k]) * x + foreign[k] * xb
            else:
                x_k = x

            # ---- expert conditional velocity on the shielded input ----
            down_c, mid_c = expert.run_one_step(x_k, t, modality_tensor, include_modality)
            v_c = _unet_velocity(x_k, t, spacing_tensor, modality_tensor,
                                 top_region_index_tensor, bottom_region_index_tensor,
                                 down_c, mid_c, diffusion_unet, include_modality, include_body_region)
            # ---- module 2: null-subtracted increment ----
            if shield_cfg.use_null_sub:
                down_n, mid_n = uncond_expert.run_one_step(x_k, t, modality_tensor, include_modality)
                v_n = _unet_velocity(x_k, t, spacing_tensor, modality_tensor,
                                     top_region_index_tensor, bottom_region_index_tensor,
                                     down_n, mid_n, diffusion_unet, include_modality, include_body_region)
                d_k = v_c - v_n
            else:
                d_k = v_c - v0

            # ---- uncertainty-aware PoE: defer this increment to the precision-weighted combine ----
            if getattr(shield_cfg, "use_uncertainty", False):
                d_list.append(d_k)
                continue

            # ---- T2: estimate expert k's per-voxel epistemic instability (MC latent perturbation) ----
            if getattr(shield_cfg, "use_mc_uncertainty", False):
                vs = []
                for _j in range(max(2, shield_cfg.mc_samples)):
                    xj = x + shield_cfg.mc_eps * torch.randn(x.shape, device=device, dtype=dtype, generator=g)
                    dcj, mcj = expert.run_one_step(xj, t, modality_tensor, include_modality)
                    vcj = _unet_velocity(xj, t, spacing_tensor, modality_tensor,
                                         top_region_index_tensor, bottom_region_index_tensor,
                                         dcj, mcj, diffusion_unet, include_modality, include_body_region)
                    vs.append(vcj)
                vstack = torch.stack(vs, dim=0)                       # (J,1,C,D,H,W)
                inst = vstack.std(dim=0).pow(2).sum(dim=1, keepdim=True).sqrt()  # (1,1,D,H,W) instability
                inst_list.append(inst)
                d_list.append(d_k)
                continue

            # ---- module 3: ownership gate + hard foreign veto ----
            if shield_cfg.use_gate:
                fl = shield_cfg.gate_floor
                own = fl + (1.0 - fl) * gates[k]               # in [floor, 1]; floor=1 → all-ones
                veto = (1.0 - foreign[k]) if (N > 1 and shield_cfg.use_hard_veto) else 1.0  # optional hard veto
                gk = own * veto
                d_k = gk * d_k

            # ---- module 4: RMS budget ----
            if shield_cfg.use_budget:
                scale = torch.clamp(shield_cfg.rho * _rms(v0) / (_rms(d_k) + eps), max=1.0)
                d_k = d_k * scale

            u = u + shield_cfg.lambda_expert * d_k

        # ---- uncertainty-aware precision-weighted composition (spatial confidence) ----
        if getattr(shield_cfg, "use_uncertainty", False):
            import torch.nn.functional as F
            mags = []
            for d_k in d_list:
                m = d_k.pow(2).sum(dim=1, keepdim=True).sqrt()       # (1,1,D,H,W) per-voxel increment magnitude = confidence
                ks = int(getattr(shield_cfg, "unc_smooth", 3))
                if ks > 1:
                    m = F.avg_pool3d(m, kernel_size=ks, stride=1, padding=ks // 2)  # denoise the confidence field
                if getattr(shield_cfg, "unc_self_norm", True):
                    m = m / (m.mean() + eps)                          # per-expert relative confidence (kills scale bias)
                mags.append(m)
            p = float(getattr(shield_cfg, "unc_power", 2.0))
            mix = float(getattr(shield_cfg, "unc_support_mix", 0.5))
            taus = []
            for k, m in enumerate(mags):
                tau = m.clamp_min(0).pow(p)                           # precision ∝ confidence^p
                if mix > 0 and k < len(gates):
                    tau = tau * ((1.0 - mix) + mix * gates[k].clamp(0, 1))  # robustify vs confidently-wrong OOD
                taus.append(tau)
            tsum = sum(taus) + eps
            for k, d_k in enumerate(d_list):
                w_k = taus[k] / tsum                                  # convex per-voxel precision weights (sum_k w_k = 1)
                u = u + shield_cfg.lambda_expert * w_k * d_k

        # ---- T2: regime-adaptive additive composition with MC-variance outlier gate ----
        if getattr(shield_cfg, "use_mc_uncertainty", False):
            import torch.nn.functional as F
            cs = []
            for inst in inst_list:
                inst_s = F.avg_pool3d(inst, kernel_size=3, stride=1, padding=1)   # denoise instability field
                mu = inst_s.mean(); sd = inst_s.std() + eps
                z = (inst_s - mu) / sd                                # spatial z-score WITHIN expert (absolute outlier-ness)
                c = torch.sigmoid(-shield_cfg.mc_sharp * (z - shield_cfg.mc_kappa))  # ~1 normally, ->0 at OOD outliers
                cs.append(c)
            for k, d_k in enumerate(d_list):
                u = u + shield_cfg.lambda_expert * cs[k] * d_k        # ADDITIVE: in-domain c~1 => naive; OOD => suppress
            if verbose and (step_idx % 6 == 0):
                for k in range(N):
                    pin = float(((cs[k] * privs[k]).sum() / (privs[k].sum() + eps)).detach().cpu())
                    pfo = float(((cs[k] * foreign[k]).sum() / (foreign[k].sum() + eps)).detach().cpu())
                    print(f"   [mc step{step_idx} exp{k}] c_own={pin:.2f} c_foreign={pfo:.2f}", flush=True)

        # ---- module 5: time-sync projection (mostly steer, limit fast-forward) ----
        if shield_cfg.use_time_sync:
            denom = (v0 * v0).sum() + eps
            par_coef = (u * v0).sum() / denom
            u_par = par_coef * v0
            u_perp = u - u_par
            v_joint = v0 + u_perp + shield_cfg.beta_sync * u_par
        else:
            v_joint = v0 + u

        if verbose and (step_idx % 6 == 0 or step_idx == len(all_t) - 1):
            par_frac = float((_rms(u_par) / (_rms(u) + eps)).detach().cpu()) if shield_cfg.use_time_sync else float("nan")
            print(f"  [step {step_idx:2d} t={float(t):.3f}] RMS v0={float(_rms(v0)):.3f} u={float(_rms(u)):.4f} "
                  f"vj={float(_rms(v_joint)):.3f} u_par_frac={par_frac:.2f} "
                  f"x[min/max]={float(x.min()):.2f}/{float(x.max()):.2f}", flush=True)

        # ---- ODE step (joint + base-only) ----
        if next_t is None or (isinstance(next_t, int) and next_t == 0 and step_idx == len(all_t) - 1):
            x, _ = noise_scheduler.step(v_joint, t, x)
            xb, _ = noise_scheduler.step(v0b, t, xb)
        else:
            x, _ = noise_scheduler.step(v_joint, t, x, next_t)
            xb, _ = noise_scheduler.step(v0b, t, xb, next_t)

        diag.velocity_inner_product_curve.append(float(_rms(u).detach().cpu()))
        diag.ess_curve.append(1.0)

    return x, diag


@torch.no_grad()
def sample_region_inpaint(
    expert: "ControlNetExpert",
    diffusion_unet: torch.nn.Module,
    noise_scheduler,
    scale_factor,
    spacing_tensor: torch.Tensor,
    modality_tensor: torch.Tensor,
    latent_shape: Tuple[int, int, int, int],
    z_cons: torch.Tensor,                 # (1, C, D, H, W) consensus latent (Stage-1 PoE output)
    region_mask: torch.Tensor,            # (D,H,W) latent mask, 1 = inpaint here (expert's own dilated region)
    tau_re: float = 0.6,                  # re-noise level in [0,1]
    n_steps: int = 30,
    cfg_scale: float = 1.5,
    uncond_controlnet_cond: Optional[torch.Tensor] = None,
    seed: int = 1234,
    resample_steps: int = 1,              # RePaint time-travel depth U (1 = no resampling / SDEdit-like)
    eta: float = 0.0,                     # stochastic-reverse Langevin strength (resampling needs eta>0)
    known_fix: bool = True,               # True = RePaint hard known-region preservation (x0-space); False = pure SDEdit
    top_region_index_tensor: Optional[torch.Tensor] = None,
    bottom_region_index_tensor: Optional[torch.Tensor] = None,
    include_modality: bool = True,
    include_body_region: bool = False,
) -> torch.Tensor:
    """Stage-2 of consensus-inpaint: RePaint-style region inpainting of ONE expert over its own region,
    starting from the Stage-1 PoE consensus latent. The complement of `region_mask` is clamped to the
    forward-noised consensus at every step (RePaint known-region replacement), so the expert re-solves
    only its home region (in-distribution, well-conditioned) while staying harmonized to the consensus."""
    device = next(diffusion_unet.parameters()).device
    dtype = next(diffusion_unet.parameters()).dtype
    g = torch.Generator(device=device); g.manual_seed(seed)
    R = region_mask.to(device, dtype)
    while R.dim() < 5:
        R = R.unsqueeze(0)                # -> (1,1,D,H,W), broadcasts over channels
    c, x, y, z = latent_shape
    if hasattr(noise_scheduler, "set_timesteps"):
        try:
            noise_scheduler.set_timesteps(n_steps, input_img_size_numel=int(x * y * z))
        except TypeError:
            noise_scheduler.set_timesteps(n_steps)
    all_t = noise_scheduler.timesteps.to(device).tolist()
    Tmax = float(all_t[0]) if all_t[0] != 0 else 1.0
    next_ts = list(all_t[1:]) + [0]
    proc = [(t, nt) for t, nt in zip(all_t, next_ts) if (float(t) / Tmax) <= tau_re + 1e-6]
    if not proc:
        proc = list(zip(all_t, next_ts))
    uncond_expert = (ControlNetExpert("uncond", expert.controlnet, uncond_controlnet_cond)
                     if uncond_controlnet_cond is not None else None)

    def fwd_noise(tau, eps):
        return (1.0 - tau) * z_cons + tau * eps   # RF interpolant x_tau = (1-tau) x0 + tau eps

    def vel(x, t):
        down, mid = expert.run_one_step(x, t, modality_tensor, include_modality)
        v_c = _unet_velocity(x, t, spacing_tensor, modality_tensor, top_region_index_tensor,
                             bottom_region_index_tensor, down, mid, diffusion_unet, include_modality, include_body_region)
        if uncond_expert is not None and cfg_scale != 1.0:
            du, mu = uncond_expert.run_one_step(x, t, modality_tensor, include_modality)
            v_u = _unet_velocity(x, t, spacing_tensor, modality_tensor, top_region_index_tensor,
                                 bottom_region_index_tensor, du, mu, diffusion_unet, include_modality, include_body_region)
            return v_u + cfg_scale * (v_c - v_u)
        return v_c

    tau_start = float(proc[0][0]) / Tmax
    lat = fwd_noise(tau_start, torch.randn(z_cons.shape, device=device, dtype=dtype, generator=g))
    for (t, next_t) in proc:
        tau = float(t) / Tmax
        tau_n = (float(next_t) / Tmax) if next_t != 0 else 0.0
        U = max(1, resample_steps)
        for u in range(U):
            v = vel(lat, t)
            x0 = lat + tau * v                       # RF x0-estimate: x_tau + tau*v = x0
            if known_fix:
                x0 = R * x0 + (1.0 - R) * z_cons      # KNOWN (complement) region's clean estimate IS the consensus
            if u < U - 1 and tau > 0:                 # time-travel: re-noise harmonized x0 back to current level, redo
                lat = (1.0 - tau) * x0 + tau * torch.randn(z_cons.shape, device=device, dtype=dtype, generator=g)
            else:                                     # advance to next level
                lat = (1.0 - tau_n) * x0 + tau_n * torch.randn(z_cons.shape, device=device, dtype=dtype, generator=g)
                if eta > 0.0 and tau_n > 0.0:
                    lat = lat + eta * math.sqrt(max(tau - tau_n, 0.0)) * torch.randn(z_cons.shape, device=device, dtype=dtype, generator=g)
    return lat


__all__ = [
    "ControlNetExpert",
    "SamplerConfig",
    "SamplerDiagnostics",
    "sample_fkc_poe_n",
    "sample_region_inpaint",
    "snis_weights",
    "snis_weighted_average",
    # spatial-aware additions:
    "SpatialFKCConfig",
    "SpatialEnergyEvaluator",
    "ROIBlockResampler",
    "rf_endpoint_estimate",
    "spatial_fkc_weight_increment",
    "sample_sa_fkc_poe",
    # shielded local composition (MVP):
    "ShieldConfig",
    "sample_shielded_local",
]
