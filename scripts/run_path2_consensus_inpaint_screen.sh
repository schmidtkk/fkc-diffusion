#!/usr/bin/env bash
# Screen the user's two-stage CONSENSUS-INPAINT composition on OOD Path-2 (n=20).
# Stage-1 PoE consensus -> Stage-2 each expert RePaint-inpaints its own dilated region.
# Predict: lifts spine (B's home region, decoupled from A's dilution + anchored to TS-like consensus),
# preserves abd. Compare vs naive(cfg_poe), hand-asym, auto(0.8/1.5) on the SAME first-20 cases.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
PY=${PY:-python}
P2A=checkpoints/path2_expert_A_amos22_scratch_onehot/checkpoint_best.pt
P2B=checkpoints/path2_expert_B_ctspine_scratch_onehot/checkpoint_best.pt
GEN=outputs/eval_path2_cfg_poe_scratch_onehot
ORACLE_CTS=outputs/eval_path1_oracle_on_path2_set_scratch_onehot
GPU=${GPU:-0}
N=${N:-20}
DONE=/tmp/DONE_P2_CONSINP
rm -f "$DONE"

echo "==== [1/3] generate Path-2 consensus_inpaint (tau=0.6, cfg=1.5, dilate=3), $N cases ===="
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_path2_shielded.py \
  --ckpt-a "$P2A" --ckpt-b "$P2B" --out-dir "$GEN" \
  --methods consensus_inpaint --inpaint-tau 0.6 --inpaint-cfg 1.5 --inpaint-dilate 3 \
  --save-label consensus_inp --case-start 0 --n-cases $N --tag path2_consinp \
  > /tmp/p2_consinp_gen.log 2>&1
echo "  gen exit=$? ; consensus_inp count: $(ls $GEN/case_eval_*_consensus_inp.nii.gz 2>/dev/null | wc -l)"

echo "==== [2/3] union-proxy score (first $N cases) ===="
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/score_composition_union_proxy.py \
  --synth-dir "$GEN" --oracle-dir "$ORACLE_CTS" \
  --methods cfg_poe,asym_a15_b10,auto_cfg,consensus_inp,oracle --n-cases $N --tag path2_consinp \
  > /tmp/p2_consinp_score.log 2>&1
echo "  score exit=$?"

echo "==== [3/3] table + Wilcoxon (consensus_inp vs naive / vs auto / vs hand-asym) ===="
$PY - <<'PYEOF'
import json, numpy as np
from scipy.stats import wilcoxon
J="reports/verify_path1/score_composition_unionproxy_path2_consinp.json"
m=json.load(open(J))["methods"]
def vec(me,r): return {c['i']:c[r] for c in m[me]['per_case'] if c.get(r) is not None}
print(f"{'method':16s} {'abd':>7s} {'spine':>7s} {'mean':>7s}")
for kk in ['cfg_poe','auto_cfg','asym_a15_b10','consensus_inp','oracle']:
    if kk in m: print(f"{kk:16s} {m[kk]['GOLD5']:7.3f} {m[kk]['LUMBAR5']:7.3f} {(m[kk]['GOLD5']+m[kk]['LUMBAR5'])/2:7.3f}")
def cmp(cand, base, blab):
    print(f"-- {cand} vs {blab} --")
    for reg in ['abd','spine']:
        b=vec(base,reg); c=vec(cand,reg); ks=sorted(set(b)&set(c))
        if len(ks)<2: print(f"   {reg}: n<2"); continue
        bv=np.array([b[k] for k in ks]); cv=np.array([c[k] for k in ks]); d=cv-bv
        print(f"   {reg:6s} n={len(ks)} {blab}={bv.mean():.3f} {cand}={cv.mean():.3f} d={d.mean():+.3f} p={wilcoxon(cv,bv).pvalue:.4f}")
for base,lab in [('cfg_poe','naive'),('auto_cfg','auto'),('asym_a15_b10','hand-asym')]:
    cmp('consensus_inp', base, lab)
PYEOF
touch "$DONE"; echo "ALL DONE -> $DONE"
