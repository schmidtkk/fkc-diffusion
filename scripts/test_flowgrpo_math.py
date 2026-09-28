#!/usr/bin/env python3
"""CPU unit tests for Flow-GRPO Gaussian and PPO math."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.flow_grpo_sampler import (  # noqa: E402
    gaussian_logprob,
    group_relative_advantage,
    ppo_clipped_loss,
    step_logprob,
)


class TinyScheduler:
    num_train_timesteps = 100

    def step(self, model_output, timestep, sample, next_timestep=None):
        dt = (float(timestep) - float(next_timestep)) / self.num_train_timesteps
        return sample + model_output * dt, sample


def const_velocity(x_t, t, cond, spacing, modality, unet, controlnet, scheduler):
    del t, cond, spacing, modality, unet, controlnet, scheduler
    return torch.full_like(x_t, 2.0)


def test_step_logprob_matches_hand_gaussian():
    scheduler = TinyScheduler()
    x_t = torch.zeros(2, 1, 2, 2, 2)
    t, next_t, sigma = 50.0, 40.0, 0.3
    mean = x_t + 2.0 * ((t - next_t) / scheduler.num_train_timesteps)
    x_next = mean + 0.1
    got = step_logprob(None, None, scheduler, x_t, x_next, t, next_t, sigma, None, None, None, const_velocity)
    var = sigma ** 2 * ((t - next_t) / scheduler.num_train_timesteps)
    want = -0.5 * ((((x_next - mean) ** 2) / var + math.log(2.0 * math.pi * var)).flatten(1).sum(dim=1))
    assert torch.allclose(got, want, atol=1e-6), (got, want)
    print("PASS step_logprob equals hand-computed Gaussian log-density")


def test_ratio_one_loss_is_negative_mean_advantage():
    logp_old = torch.tensor([-1.0, -2.0, -3.0])
    logp_new = logp_old.clone()
    adv = torch.tensor([1.0, -0.5, 0.25])
    loss, ratio = ppo_clipped_loss(logp_new, logp_old, adv, 0.2)
    assert torch.allclose(ratio, torch.ones_like(ratio))
    assert torch.allclose(loss, -adv.mean())
    print("PASS theta==theta_old gives ratio=1 and loss=-mean(A)")


def test_group_advantage_zero_mean_unit_std():
    rewards = torch.tensor([1.0, 2.0, 4.0, 7.0])
    adv = group_relative_advantage(rewards)
    assert abs(float(adv.mean())) < 1e-7
    assert abs(float(adv.std(unbiased=False)) - 1.0) < 1e-6
    print("PASS group advantage has zero mean and unit std")


def test_advantage_masking():
    rewards = torch.tensor([1.0, 2.0, 4.0, 7.0])
    adv = group_relative_advantage(rewards, adv_mask=True)
    mean = rewards.mean()
    assert torch.all(adv[rewards < mean] == 0)
    assert torch.all(adv[rewards >= mean] >= 0)
    print("PASS advantage masking zeros below-mean samples")


def test_ppo_clip_caps_ratio():
    logp_old = torch.tensor([0.0])
    logp_new = torch.tensor([math.log(2.0)])
    adv = torch.tensor([1.0])
    loss, ratio = ppo_clipped_loss(logp_new, logp_old, adv, 0.2)
    assert torch.allclose(ratio, torch.tensor([2.0]))
    assert torch.allclose(loss, torch.tensor(-1.2), atol=1e-6)
    print("PASS PPO clip caps positive-advantage ratio at 1+eps")


if __name__ == "__main__":
    test_step_logprob_matches_hand_gaussian()
    test_ratio_one_loss_is_negative_mean_advantage()
    test_group_advantage_zero_mean_unit_std()
    test_advantage_masking()
    test_ppo_clip_caps_ratio()
