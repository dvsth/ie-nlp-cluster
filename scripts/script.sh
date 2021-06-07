#!/bin/bash
#SBATCH --job-name=ieneurips
# Limit running time to 5 minutes.
#SBATCH -t 0:05:00  # time requested in hour:minute:second
# Request 8GB of RAM
#SBATCH --mem=10Gs

echo 'script loaded'
python3 /usr/xtmp/ds447/ie-neurips/nlp.py