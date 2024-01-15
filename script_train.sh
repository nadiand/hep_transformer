#!/bin/bash
#SBATCH --job-name=train          # Job name
#SBATCH --ntasks=1                    # Run on a single GPU
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END

cd /projects/0/nisei0750/nadia/repo/hep_transformer/

module purge
module load 2022 Python/3.10.4-GCCcore-11.3.0

pip install torch==2.1.0
pip install scikit-learn
pip install pandas

python training_f16.py