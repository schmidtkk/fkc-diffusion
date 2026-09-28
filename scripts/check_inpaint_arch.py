#!/usr/bin/env python
"""Forward-only architecture check for the 45-ch inpaint ControlNet (NO training/backward — setsid-safe).
Confirms: (1) controlnet from config_network_inpaint45 instantiates with cond-embed in_channels=45,
(2) it accepts a 45-ch cond + 4-ch latent and returns down/mid residuals, (3) frozen UNet consumes them."""
import sys
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "maisi")); sys.path.insert(0, str(REPO))
from scripts.utils import define_instance  # noqa
from scripts.diff_model_setting import load_config  # noqa
from src.cond_encoding import to_inpaint_cond, N_INPAINT_COND_CHANNELS  # noqa

dev = torch.device("cuda:0")
cfg = load_config(str(REPO / "configs/env_path2_expert_A_amos22.json"),
                  str(REPO / "configs/config_expert_aligned_finetune.json"),
                  str(REPO / "configs/config_network_inpaint45.json"))
cn = define_instance(cfg, "controlnet_def").to(dev).eval()
# report the cond-embedding first conv in_channels
first = None
for m in cn.modules():
    if isinstance(m, torch.nn.Conv3d):
        first = m; break
print("controlnet first conv in/out:", first.in_channels, first.out_channels)
ce_in = [m.in_channels for m in cn.modules() if isinstance(m, torch.nn.Conv3d) and m.in_channels == 45]
print("found a Conv3d with in_channels==45:", len(ce_in) > 0)

unet = define_instance(cfg, "diffusion_unet_def").to(dev).eval()
sd = torch.load(cfg.trained_diffusion_path, map_location=dev, weights_only=False)
unet.load_state_dict(sd["unet_state_dict"], strict=False)

# tiny forward: latent (1,4,16,16,16), cond (1,45,64,64,64)
B, D = 1, 16
x = torch.randn(B, 4, D, D, D, device=dev)
lbl = torch.zeros(B, 1, 4*D, 4*D, 4*D, device=dev); lbl[0,0,20:40,20:40,20:40] = 1
ct = torch.rand(B, 1, 4*D, 4*D, 4*D, device=dev)
msk = torch.zeros(B, 1, 4*D, 4*D, 4*D, device=dev); msk[0,0,20:40,20:40,20:40] = 1
cond = to_inpaint_cond(lbl.long(), ct, msk).to(dev)
print("cond shape:", tuple(cond.shape))
t = torch.full((B,), 500.0, device=dev)
mod = torch.ones((B,), device=dev, dtype=torch.long)
sp = torch.tensor([[1.5,1.5,1.5]], device=dev) * 100.0
with torch.no_grad():
    down, mid = cn(x=x, timesteps=t, controlnet_cond=cond, class_labels=mod)
    print("controlnet OK -> #down:", len(down), "mid:", tuple(mid.shape))
    out = unet(x=x, timesteps=t, spacing_tensor=sp, class_labels=mod,
               down_block_additional_residuals=down, mid_block_additional_residual=mid)
    print("frozen UNet consumed residuals -> out:", tuple(out.shape))
print("ARCH CHECK PASSED: 45-ch inpaint ControlNet + frozen UNet forward works.")
