#!/bin/bash

# make the script stop when error (non-true exit code) is occured
set -e

############################################################
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
# <<< conda initialize <<<
############################################################

SCRIPT=`realpath -s $0`
export PIPEDIR=`dirname $SCRIPT`

CPU="8"  # number of CPUs to use
MEM="64" # max memory (in GB)

# Inputs:
IN_FASTA="$1"     # input.fa file
IN_C6D="$2"       # c6d numpy file (n_templates, L, L, 4)
WDIR=`realpath -s $3`  # working folder

# Optional: confidence parameter (default 0.7)
CONFIDENCE="${4:-0.7}"

mkdir -p $WDIR/log

conda activate RoseTTAFold

############################################################
# 1. Create minimal MSA from input fasta
############################################################
if [ ! -s $WDIR/t000_.msa0.a3m ]
then
    echo "Creating minimal MSA from input FASTA"
    # Convert FASTA to a3m format (minimal MSA with just the query sequence)
    cat $IN_FASTA | sed 's/>/>query\n/' > $WDIR/t000_.msa0.a3m
fi

############################################################
# 2. Run custom prediction with c6d templates
############################################################
if [ ! -s $WDIR/t000_.npz ]
then
    echo "Running custom prediction with c6d templates"
    python $PIPEDIR/network/predict_custom.py \
        -m $PIPEDIR/weights \
        -i $WDIR/t000_.msa0.a3m \
        -o $WDIR/t000_ \
        --c6d $IN_C6D \
        --confidence $CONFIDENCE \
        1> $WDIR/log/network.stdout 2> $WDIR/log/network.stderr
fi

echo "Done"
