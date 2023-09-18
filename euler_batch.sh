#!/bin/bash

# #SBATCH --nodes=4
# #SBATCH --ntasks 32
# #SBATCH --ntasks-per-node=8
# #SBATCH --time=48:00:00
# #SBATCH --gpus-per-node=8
# #SBATCH --mem-per-cpu=32g 
# #SBATCH --output=output/multinode_%A.log
# #SBATCH --gres=gpumem:20g 
# #SBATCH --gpus=32


# srun --gres=gpumem:30g train_multi_node.sh
srun --gres=gpumem:20g train_multi_node.sh
# srun --gres=gpumem:20g pre_train_multi_node.sh
