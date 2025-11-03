#!/bin/sh
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=2:00:00
#PBS -P gag51404

cd /groups/gag51404/fumiyau/repos/clip_sem_info

source /etc/profile.d/modules.sh

CMD="pip install wandb && cd /groups/gag51404/fumiyau/repos/clip_sem_info/src && \
torchrun --nproc_per_node 8 -m open_clip_train.main \
  --train-data '/groups/gag51404/fumiyau/data/cc12m/cc12m/{00000..01242}.tar' \
  --train-num-samples 12423374  \
  --dataset-type webdataset \
  --model ViT-B-32 \
  --report-to wandb \
  --wandb-project-name clip-sem-info \
  --batch-size 1000 \
  --precision amp \
  --workers 8 \
  --epochs 3 \
  --imagenet-val /groups/gag51404/fumiyau/data/imagenet_1k/val"

export SINGULARITY_BINDPATH="/groups/gag51404/fumiyau"
singularity exec --nv envs/singularity/clip_sem_info.sif  /bin/bash -c "$CMD"