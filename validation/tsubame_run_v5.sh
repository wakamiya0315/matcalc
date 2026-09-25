#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -o /gs/fs/tga-ishikawalab/wakamiya/temp/matcalc-v5-full/log/
#$ -e /gs/fs/tga-ishikawalab/wakamiya/temp/matcalc-v5-full/log/
# V5: one benchmark on its full dataset with one code path. CODE (upstream | fork-ase | fork-torchsim), B and
# EXTRA (extra run_one.py options, e.g. "--dtype float32") come from qsub -v. TSDIR is the fork checkout.
set -uo pipefail
V=/gs/fs/tga-ishikawalab/wakamiya/temp/matcalc-v5-full
VENV=/gs/fs/tga-ishikawalab/wakamiya/apps/matcalc-ts-cu126-py312
GE=/gs/fs/tga-ishikawalab/wakamiya/git_edit
cd "$V" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SRC=$TSDIR/src; [ "$CODE" = upstream ] && SRC=$GE/matcalc/src
TAG=${TAG:-$CODE}
echo "host: $(hostname)  start: $(date)  $CODE $B $TAG  code: $(git -C ${SRC%/src} log --oneline -1)"
nvidia-smi -L
PYTHONPATH=$SRC "$VENV/bin/python" run_one.py "$CODE" "$B" --out "out/${B}_${TAG}.csv" --checkpoint "out/${B}_${TAG}.ckpt.json" --workers ${WORKERS:-1} ${CUEQ:+--cueq} \
  > "log/${B}_${TAG}.out" 2> "log/${B}_${TAG}.err"
echo "exit=$?  end: $(date)"
tail -1 "log/${B}_${TAG}.out"
