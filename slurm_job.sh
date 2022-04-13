#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH -p v100_dev_q
#SBATCH -n 1
#SBATCH -A ece6524-spring2022 

echo Allocation granted
module purge
module load cuda/10.2.89

source ~/anaconda3/bin/activate
conda init
conda activate cascades

cd $SLURM_SUBMIT_DIR
cd /home/gaurangajitk/DL/image-captioning

conda activate cascades
python train.py
exit;