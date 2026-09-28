#!/usr/bin/env python
"""1-case diagnostic for the shielded sampler: isolate which module collapses output."""
import sys, json
from pathlib import Path
import numpy as np, nibabel as nib, torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "maisi")); sys.path.insert(0, str(REPO))
from scripts.utils import define_instance
from scripts.diff_model_setting import load_config
from src.fkc_poe_sampler import ControlNetExpert, ShieldConfig, sample_shielded_local
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
unet.eval()
ns = define_instance(cfg,"noise_scheduler")
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
experts=[ControlNetExpert("A",ctrlA,condA),ControlNetExpert("B",ctrlB,condB)]

for name, kw in [
    ("FULL shielded(beta=0.1)", dict(use_shield=True,use_gate=True,use_budget=True,use_time_sync=True,beta_sync=0.1)),
    ("no_timesync",            dict(use_shield=True,use_gate=True,use_budget=True,use_time_sync=False)),
    ("timesync_beta=1.0",      dict(use_shield=True,use_gate=True,use_budget=True,use_time_sync=True,beta_sync=1.0)),
    ("no_gate",                dict(use_shield=True,use_gate=False,use_budget=True,use_time_sync=True,beta_sync=0.1)),
    ("no_budget_no_sync",      dict(use_shield=True,use_gate=True,use_budget=False,use_time_sync=False)),
]:
    print(f"\n===== {name} =====", flush=True)
    shp=ShieldConfig(lambda_expert=0.7,rho=0.5,seed=1234,n_steps=30,
                     **{**dict(use_shield=True,use_null_sub=True,use_gate=True,use_budget=True,use_time_sync=True,beta_sync=0.1), **kw})
    lat,_=sample_shielded_local(experts,unet,ns,sf,sp,mod,(C,D,H,W),uncond,[gA,gB],[PA,PB],shp,
                                include_modality=True,include_body_region=False,verbose=True)
    print(f"  final latent: nan={bool(torch.isnan(lat).any())} min={float(lat.min()):.2f} max={float(lat.max()):.2f} rms={float(lat.pow(2).mean().sqrt()):.3f}", flush=True)
