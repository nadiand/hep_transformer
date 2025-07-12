#!/bin/bash
#SBATCH --job-name=train              # Job name
#SBATCH --ntasks=1                    # Run on a single GPU
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END

cd /projects/0/nisei0750/nadia/repo/hep_transformer/

module purge
module load 2023
module load CUDA/12.1.1

python -m venv trackformers
source trackformers/bin/activate

pip install torch==2.7
pip install scikit-learn
pip install pandas
pip install matplotlib

pip install mambapy
pip install lightning_utilities
pip install hdbscan

nvcc --version
#pip freeze > requirements.txt

python training_flash.py
#python test.py