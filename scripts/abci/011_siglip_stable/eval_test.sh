#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -P gag51404

cd /groups/gag51404/fumiyau/repos/clip_sem_info

source /etc/profile.d/modules.sh

CMD="cd /groups/gag51404/fumiyau/repos/clip_sem_info/src && \
python clip_sem_info/main.py"

export SINGULARITY_HF_TOKEN="xxxxxxxx"
export SINGULARITY_BINDPATH="/groups/gag51404/fumiyau"
singularity exec --nv envs/singularity/clip_sem_info.sif  /bin/bash -c "$CMD"