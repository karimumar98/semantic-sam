#!/bin/bash


export CURRENT_NODE=$(hostname)
echo "Initializing node "$CURRENT_NODE

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURM_NTASKS_PER_NODE="$SLURM_NTASKS_PER_NODE

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

export RANK=$((SLURM_PROCID))
echo "RANK="$RANK

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
export PYTHONPATH=$PYTHONPATH:/cluster/project/zhang/umarka/clip_detector/utils/coco_caption

export WANDB_DIR=/cluster/scratch/umarka/wandb

nvidia-smi -L

module load python_gpu/3.10.4 cuda/12.1.1
source /cluster/project/zhang/umarka/clip_detector/semantic-segment-anything/.ssam/bin/activate 

python -u train.py -c /cluster/project/zhang/umarka/clip_detector/workspace/configs/coco_data.yaml 
