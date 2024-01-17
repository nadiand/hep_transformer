#!/bin/bash
#SBATCH --job-name=domain_decomposition          # Job name
#SBATCH --ntasks=1                    # Run on a single GPU
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END

cd /projects/0/nisei0750/nadia/repo/hep_transformer/

module purge
module load 2022 Python/3.10.4-GCCcore-11.3.0

pip install torch==1.13.1
pip install scikit-learn
pip install pandas

for i in {21000..21010} ; do
    python domain_decomposition.py -e ${i}
done
