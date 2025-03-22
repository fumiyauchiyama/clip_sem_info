#!/bin/sh
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -P gag51404

cd /groups/gag51404/fumiyau/repos/clip_sem_info

source /etc/profile.d/modules.sh

CMD="pip install wandb && cd /groups/gag51404/fumiyau/repos/clip_sem_info/src && \
torchrun --nproc_per_node 8 -m open_clip_train.main \
  --train-data '/groups/gag51404/fumiyau/data/cc3m/cc3m_train/{00000..00331}.tar' \
  --train-num-samples 3318333  \
  --dataset-type webdataset \
  --model ViT-B-32 \
  --siglip \
  --report-to wandb \
  --wandb-project-name clip-sem-info \
  --batch-size 3000 \
  --precision amp_bf16 \
  --workers 8 \
  --imagenet-val /groups/gag51404/fumiyau/data/imagenet_1k/val \
  --grad-clip-norm 1.0 \
  --warmup 2000 \
  --lr 1e-3 \
  --seed 0 \
  --epochs 64"

export SINGULARITY_BINDPATH="/groups/gag51404/fumiyau"
singularity exec --nv envs/singularity/clip_sem_info.sif  /bin/bash -c "$CMD"