#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -t 1-4
#$ -o /gs/fs/tga-ishikawalab/wakamiya/temp/matcalc-discovery-shards/log/
#$ -e /gs/fs/tga-ishikawalab/wakamiya/temp/matcalc-discovery-shards/log/
# The whole Discovery benchmark with MACE-MP-0 in 4 shards, one gpu_h slice per array task (about 55 min each; the
# unsplit run takes 3.4 h on one slice). TSDIR is the fork checkout:
#   mkdir -p /gs/fs/tga-ishikawalab/wakamiya/temp/matcalc-discovery-shards/{log,out}
#   qsub -g tga-ishikawalab -l h_rt=01:15:00 -v TSDIR=<fork checkout> tsubame_discovery_shards.sh
# then join the shards (a few minutes, e.g. on iqrsh), in the work directory:
#   PYTHONPATH=<fork checkout>/src python <fork checkout>/validation/merge_shards.py \
#       out/discovery_0of4.csv out/discovery_1of4.csv out/discovery_2of4.csv out/discovery_3of4.csv --out out/discovery.csv
set -uo pipefail
D=/gs/fs/tga-ishikawalab/wakamiya/temp/matcalc-discovery-shards
VENV=/gs/fs/tga-ishikawalab/wakamiya/apps/matcalc-ts-cu126-py312
N=4
K=$((SGE_TASK_ID - 1))
cd "$D" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host: $(hostname)  start: $(date)  shard $K/$N  code: $(git -C "$TSDIR" log --oneline -1)"
nvidia-smi -L
PYTHONPATH=$TSDIR/src "$VENV/bin/python" "$TSDIR/validation/run_one.py" fork-torchsim discovery --model medium \
  --workers 3 --shard "$K/$N" --out "out/discovery_${K}of${N}.csv" --checkpoint "out/discovery_${K}of${N}.ckpt.json.gz" \
  > "log/discovery_${K}of${N}.out" 2> "log/discovery_${K}of${N}.err"
echo "exit=$?  end: $(date)"
