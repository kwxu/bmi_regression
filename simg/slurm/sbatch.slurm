#!/bin/bash
#SBATCH --account=vuiis_masi_gpu_acc
#SBATCH --partition=pascal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --mem=12G

SIMG_PATH=/scratch/xuk9/singularity/thorax_combine_20201022.img
BIN_EXTERNAL=/home/xuk9/src/bmi_regression/simg/bin
PROJ_EXTERNAL=/scratch/xuk9/nifti/SPORE/CAC_class
SRC_EXTERNAL=/home/xuk9/src/bmi_regression

set -o xtrace
singularity exec --nv -B ${BIN_EXTERNAL}:/bin_external -B ${SRC_EXTERNAL}:/src_external -B ${PROJ_EXTERNAL}:/proj_root ${SIMG_PATH} $1 $2 $3
set +o xtrace