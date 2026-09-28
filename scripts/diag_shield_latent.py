#!/usr/bin/env python
"""Decisive latent-level diagnostic: localize why shielded composition is weak.

Compares FINAL latents (no segmentation) on 1 case:
  L_sa  = single_A via residual_sum
  L_sh1 = shielded with [A] only (N=1)  -> should EQUAL L_sa (floor=1,lam=1,no damping)
  L_sh2 = shielded with [A,B]           -> B's effect = L_sh2 - L_sh1
Also reports, in A's private zone (P_A) vs B's zone (P_B), the per-region latent
differences, to detect mask/latent spatial MISALIGNMENT.
"""
import sys, json
from pathlib import Path
import numpy as np, nibabel as nib, torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "maisi")); sys.path.insert(0, str(REPO))
from scripts.utils import define_instance
from scripts.diff_model_setting import load_config
from src.fkc_poe_sampler import (ControlNetExpert, SamplerConfig, sample_fkc_poe_n,
                                 ShieldConfig, sample_shielded_local)
from src.cond_encoding import to_one_hot
from src.shield_masks import build_shield_masks
DATA = REPO / "data/phase1_ts_iso15"
ABD = {1,3,4,5,6,7,8,9,10,11,12,13,14,15}; SP = set(range(33,57))
dev = torch.device("cuda:0")
cfg = load_config(str(REPO/"configs/env_path2_expert_A_amos22.json"),
                  str(REPO/"configs/config_expert_aligned_finetune.json"),
                  str(REPO/"configs/config_network_onehot43.json"))
unet = define_instance(cfg,"diffusion_unet_def").to(dev)
sd = torch.load(cfg.trained_diffusion_path, map_location=dev, weights_only=False)
unet.load_state_dict(sd["unet_state_dict"], strict=False)
sf = sd["scale_factor"]; sf = sf.to(dev) if hasattr(sf,"to") else torch.tensor(sf).to(dev)
unet.eval(); ns = define_instance(cfg,"noise_scheduler")
ctrlA = define_instance(cfg,"controlnet_def").to(dev)
ctrlA.load_state_dict(torch.load(str(REPO/"checkpoints/path2_expert_A_amos22_scratch_onehot/checkpoint_best.pt"),map_location=dev,weights_only=False)["controlnet_state_dict"],strict=True); ctrlA.eval()
ctrlB = define_instance(cfg,"controlnet_def").to(dev)
ctrlB.load_state_dict(torch.load(str(REPO/"checkpoints/path2_expert_B_ctspine_scratch_onehot/checkpoint_best.pt"),map_location=dev,weights_only=False)["controlnet_state_dict"],strict=True); ctrlB.eval()
def bc(l,v,b=200):
    o=np.full_like(l,b,dtype=np.int32); o[l==0]=0
    for s in v: o[l==s]=s
    return o
ent = json.load(open(DATA/"eval_dataset.json"))["training"][0]
lbl = nib.load(str(DATA/ent["label"])).get_fdata().astype(np.int32)
cA=bc(lbl,ABD); cB=bc(lbl,SP)
condA=to_one_hot(torch.from_numpy(cA).long().unsqueeze(0).unsqueeze(0).to(dev))
condB=to_one_hot(torch.from_numpy(cB).long().unsqueeze(0).unsqueeze(0).to(dev))
uncond=torch.zeros_like(condA)
C,D,H,W=4,64,64,64
gA,gB,PA,PB=build_shield_masks(cA,cB,ABD,SP,latent_dim=64,device=dev)
sp=torch.tensor([[1.5,1.5,1.5]],device=dev,dtype=torch.float32)*100.0
mod=torch.ones((1,),device=dev,dtype=torch.long)

def rms(x): return float(x.pow(2).mean().sqrt())

# 1. single_A residual_sum
eA=[ControlNetExpert("A",ctrlA,condA)]
scfg=SamplerConfig(K_SMC=1,n_steps=30,composition="residual_sum",use_fkc_weights=False,seed=1234)
L_sa,_,_=sample_fkc_poe_n(eA,unet,ns,sf,sp,mod,(C,D,H,W),scfg,uncond_controlnet_cond=None,include_modality=True,include_body_region=False)
# 1b. cfg single A (should equal residual_sum single A mathematically)
scfg2=SamplerConfig(K_SMC=1,n_steps=30,composition="cfg_poe",use_fkc_weights=False,seed=1234)
L_cfgA,_,_=sample_fkc_poe_n(eA,unet,ns,sf,sp,mod,(C,D,H,W),scfg2,uncond_controlnet_cond=uncond,include_modality=True,include_body_region=False)
# 2. shielded A-only (N=1)
shp=ShieldConfig(lambda_expert=1.0,gate_floor=1.0,use_shield=True,use_null_sub=True,use_gate=True,use_budget=False,use_time_sync=False,seed=1234,n_steps=30)
L_sh1,_=sample_shielded_local([ControlNetExpert("A",ctrlA,condA)],unet,ns,sf,sp,mod,(C,D,H,W),uncond,[gA],[PA],shp,include_modality=True,include_body_region=False)
# 3. shielded A+B
L_sh2,_=sample_shielded_local([ControlNetExpert("A",ctrlA,condA),ControlNetExpert("B",ctrlB,condB)],unet,ns,sf,sp,mod,(C,D,H,W),uncond,[gA,gB],[PA,PB],shp,include_modality=True,include_body_region=False)

print("\n==== latent comparisons (RMS) ====")
print(f"RMS(L_sa)              = {rms(L_sa):.4f}")
print(f"RMS(L_cfgA - L_sa)     = {rms(L_cfgA-L_sa):.5f}   (cfg vs residual_sum single A; should be ~0)")
print(f"RMS(L_sh1 - L_cfgA)    = {rms(L_sh1-L_cfgA):.5f}   (shielded-A-only vs cfg-A; should be ~0 if machinery correct)")
print(f"RMS(L_sh1 - L_sa)      = {rms(L_sh1-L_sa):.5f}")
print(f"RMS(L_sh2 - L_sh1)     = {rms(L_sh2-L_sh1):.5f}   (B's total effect on the latent)")
# per-region: how much does L_sh2 differ from L_sh1 INSIDE A's zone (P_A) vs B's zone (P_B)?
pa=(PA>0).float(); pb=(PB>0).float()
diff=(L_sh2-L_sh1).abs().mean(dim=1,keepdim=True)  # (1,1,D,H,W)
ina = float((diff*pa).sum()/(pa.sum()+1e-6)); inb=float((diff*pb).sum()/(pb.sum()+1e-6))
nm = float((diff*(1-pa)*(1-pb)).sum()/(((1-pa)*(1-pb)).sum()+1e-6))
print(f"\nB's effect by region (mean|Δlatent|):  A-zone(P_A)={ina:.4f}  B-zone(P_B)={inb:.4f}  no-mans-land={nm:.4f}")
print("  -> if A-zone >> 0, B is LEAKING into A's region (veto fails / mask misaligned)")
