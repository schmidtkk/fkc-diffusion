#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
PY=${PY:-python}
P2A=checkpoints/path2_expert_A_amos22_scratch_onehot/checkpoint_best.pt
P2B=checkpoints/path2_expert_B_ctspine_scratch_onehot/checkpoint_best.pt
IPA=checkpoints/path2_expert_A_inpaint/checkpoint_latest.pt
IPB=checkpoints/path2_expert_B_inpaint/checkpoint_latest.pt
P2DIR=outputs/eval_path2_cfg_poe_scratch_onehot
ORACLE=outputs/eval_path1_oracle_on_path2_set_scratch_onehot
rm -f /tmp/DONE_TRAINED_INPAINT_POW
echo "=== GEN trained_inpaint cases 12-24 (GPU 1) ==="
CUDA_VISIBLE_DEVICES=1 $PY -u scripts/eval_path2_shielded.py \
  --ckpt-a "$P2A" --ckpt-b "$P2B" --ckpt-a-inpaint "$IPA" --ckpt-b-inpaint "$IPB" \
  --out-dir "$P2DIR" --inpaint-cfg 1.5 --inpaint-dilate 3 \
  --methods trained_inpaint --save-label trained_inpaint --case-start 12 --n-cases 13 --tag trained_inpaint_pow \
  > /tmp/trained_inpaint_pow_gen.log 2>&1
echo "  gen exit=$? ; trained_inpaint CTs now: $(ls $P2DIR/case_eval_*_trained_inpaint.nii.gz 2>/dev/null | wc -l)"
echo "=== SCORE n=25 ==="
CUDA_VISIBLE_DEVICES=1 $PY scripts/score_composition_union_proxy.py \
  --synth-dir "$P2DIR" --oracle-dir "$ORACLE" \
  --methods trained_inpaint,auto_v2a,asym_a15_b10,cfg_poe,oracle \
  --n-cases 25 --tag trained_inpaint_n25 > /tmp/trained_inpaint_pow_score.log 2>&1
echo "  score exit=$?"
echo "=== RESULT TABLE (Path-2 OOD trained inpaint, n=25) ==="
$PY -c "
import json
d=json.load(open('reports/verify_path1/score_composition_unionproxy_trained_inpaint_n25.json'))
print(f\"{'method':<16}{'abd':>7}{'spine':>8}{'mean':>8}\")
for m,v in d['methods'].items():
    a=v['GOLD5']; s=v['LUMBAR5']; print(f'{m:<16}{a:>7.3f}{s:>8.3f}{(a+s)/2:>8.3f}')
import numpy as np
from scipy.stats import wilcoxon
def vec(me,r): return {c['i']:c[r] for c in d['methods'][me]['per_case'] if c.get(r) is not None}
for base in ['asym_a15_b10','cfg_poe']:
  for reg in ['abd','spine']:
    A=vec('trained_inpaint',reg); B=vec(base,reg); ks=sorted(set(A)&set(B))
    av=np.array([A[k] for k in ks]); bv=np.array([B[k] for k in ks])
    try: p=wilcoxon(av,bv).pvalue
    except Exception: p=float('nan')
    print(f'  trained_inpaint vs {base:13} [{reg:5}]: d={(av-bv).mean():+.3f} p={p:.4f} (n={len(ks)})')
" 2>&1
touch /tmp/DONE_TRAINED_INPAINT_POW
echo "=== TRAINED INPAINT POWER DONE ==="
