#!/bin/bash
#SBATCH -o clean.out
#SBATCH --job-name=ieclean
# Limit running time to 5 minutes.
#SBATCH -t 0:05:00  # time requested in hour:minute:second
# Request 8GB of RAM
#SBATCH --mem=8Gs

hostname
source env/bin/activate
echo 'env started'
time python3 clean.py
echo 'cleaning done'