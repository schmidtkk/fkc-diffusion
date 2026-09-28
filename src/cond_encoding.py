"""Shared conditioning encoding for from-scratch ControlNet experts.

Since the contamination-clean experts do NOT load MAISI's pretrained ControlNet
(which mandated the native 8-bit binarize_labels encoding), we are free to use a
one-hot per-organ encoding. One-hot is easier for the ControlNet to learn: each
organ gets a dedicated channel, with no bit-pattern entanglement between adjacent
IDs (e.g. vertebrae 33 vs 34 share 7 of 8 bits under binarize_labels).

CRITICAL: training and evaluation MUST import `to_one_hot` from this single module
so the channel map can never drift between train and inference (a silent-garbage bug).

Channel layout (43 channels, shared by ALL experts + Oracle):
  - channels 0..41 : the 42-organ union vocabulary, MAISI IDs sorted ascending
  - channel 42     : body silhouette (MAISI ID 200)
  - background (ID 0) -> all-zero across channels (also the cfg uncond)

Each expert only ever activates the channels in its own vocabulary; the rest stay
zero (their cond-embedding conv weights simply never receive gradient — harmless).
This keeps a single consistent semantic layout so cfg_poe composition is clean.
"""
from __future__ import annotations
import torch

# 42-organ union (15 abdominal + 27 spine) from data/label_partition.json
_UNION_IDS = [
    1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17,          # abdominal (15)
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,  # spine ...
    48, 49, 50, 51, 52, 53, 54, 55, 56, 97, 121, 127,            # ... (27)
]
BODY_ID = 200

# MAISI ID -> channel index
CHANNEL_MAP: dict[int, int] = {mid: ch for ch, mid in enumerate(_UNION_IDS)}
CHANNEL_MAP[BODY_ID] = len(_UNION_IDS)  # body -> channel 42

N_COND_CHANNELS = len(CHANNEL_MAP)  # 43


def to_one_hot(lbl_t: torch.Tensor) -> torch.Tensor:
    """Encode an integer label volume as a one-hot conditioning tensor.

    Args:
        lbl_t: (B, 1, D, H, W) integer label volume with MAISI IDs.
    Returns:
        (B, N_COND_CHANNELS, D, H, W) float32 one-hot tensor.
    """
    assert lbl_t.dim() == 5 and lbl_t.shape[1] == 1, f"expected (B,1,D,H,W), got {tuple(lbl_t.shape)}"
    B = lbl_t.shape[0]
    spatial = lbl_t.shape[2:]
    out = torch.zeros((B, N_COND_CHANNELS, *spatial), dtype=torch.float32, device=lbl_t.device)
    lbl_sq = lbl_t.squeeze(1)
    for maisi_id, ch in CHANNEL_MAP.items():
        out[:, ch] = (lbl_sq == maisi_id).to(torch.float32)
    return out


N_INPAINT_COND_CHANNELS = N_COND_CHANNELS + 2  # 45 = 43 label one-hot + 1 masked-CT + 1 mask


def to_inpaint_cond(lbl_t: torch.Tensor, masked_ct: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Conditioning for a TRAINED inpaint expert: [label one-hot (43) | masked-CT (1) | mask R (1)] -> (B,45,D,H,W).

    Single source of truth shared by train + inference (mirrors to_one_hot) so the 45-channel layout never drifts.
    Args:
        lbl_t:     (B,1,D,H,W) integer MAISI-ID label volume.
        masked_ct: (B,1,D,H,W) CT with the inpaint region zeroed/air-filled, already normalized to the model's
                   input convention (same normalization used to encode CTs to latents).
        mask:      (B,1,D,H,W) binary inpaint region R (1 = to be (re)generated).
    Returns:
        (B,45,D,H,W) float32.
    """
    assert lbl_t.dim() == 5 and lbl_t.shape[1] == 1
    assert masked_ct.shape == lbl_t.shape and mask.shape == lbl_t.shape, "masked_ct/mask must be (B,1,D,H,W)"
    onehot = to_one_hot(lbl_t)                                                         # (B,43,...)
    extra = torch.cat([masked_ct.to(torch.float32), mask.to(torch.float32)], dim=1)    # (B,2,...)
    return torch.cat([onehot, extra.to(onehot.device)], dim=1)                          # (B,45,...)
