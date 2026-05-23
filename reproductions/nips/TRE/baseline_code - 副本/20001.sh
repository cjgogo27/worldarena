#!/bin/bash
#SBATCH --account=research-ceg-tp
#SBATCH --job-name=UAV_GV
#SBATCH --partition=compute
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --nodes=1
#SBATCH --ntasks=3                         # Run on a single CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G                           # Job memory request
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output=UAV-GV/test3.log
python3 Intermodal_ALNS_0625_400.py & rm -r /scratch/yimengzhang/cjyang/codes_ALNS-1/Figures/experiment606250006
wait
