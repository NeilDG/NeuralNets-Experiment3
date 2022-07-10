#!/bin/bash
#SBATCH -J V7_3
#SBATCH --partition=gpu
#SBATCH --qos=12c-1h_2gpu
#SBATCH --cpus-per-task=6
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --output=script_3.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

ITERATION=7
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda/3-2021.11
module load cuda/10.1_cudnn-7.6.5

source activate NeilGAN_V2

srun python iid_train.py \
--iteration=$ITERATION \
--server_config=1 \
--batch_size=128 \
--patch_size=64 \
--plot_enabled=0 \
--version_name="iid_networkv8.01" --min_epochs=40 --net_config=4 --num_blocks=4 \
--da_enabled=0 --da_version_name="embedding_v5.00_5" \
--img_to_load=-1 --load_previous=0

conda deactivate