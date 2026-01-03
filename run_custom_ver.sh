#!/bin/bash

# stop on error
set -e

############################################################
# >>> conda initialize >>>
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
# <<< conda initialize <<<
############################################################

SCRIPT=`realpath -s $0`
export PIPEDIR=`dirname $SCRIPT`

CPU="8"   # number of CPUs to use
MEM="64"  # max memory (in GB)

# Inputs:
IN="$1"              # input.fasta
WDIR=`realpath -s $2`  # working folder

mkdir -p $WDIR/log

conda activate RoseTTAFold

############################################################
# 1. generate MSAs
############################################################
if [ ! -s $WDIR/t000_.msa0.a3m ]
then
  echo "Running HHblits"
  $PIPEDIR/input_prep/make_msa.sh $IN $WDIR $CPU $MEM > $WDIR/log/make_msa.stdout 2> $WDIR/log/make_msa.stderr
fi

############################################################
# 2. predict secondary structure for HHsearch run
############################################################
if [ ! -s $WDIR/t000_.ss2 ]
then
  echo "Running PSIPRED"
  $PIPEDIR/input_prep/make_ss.sh $WDIR/t000_.msa0.a3m $WDIR/t000_.ss2 > $WDIR/log/make_ss.stdout 2> $WDIR/log/make_ss.stderr
fi

############################################################
# 3. search for templates (optional but kept same as e2e script)
############################################################
DB="$PIPEDIR/pdb100_2021Mar03/pdb100_2021Mar03"

if [ ! -s $WDIR/t000_.hhr ]
then
  echo "Running hhsearch"
  HH="hhsearch -b 50 -B 500 -z 50 -Z 500 -mact 0.05 -cpu $CPU -maxmem $MEM -aliw 100000 -e 100 -p 5.0 -d $DB"
  cat $WDIR/t000_.ss2 $WDIR/t000_.msa0.a3m > $WDIR/t000_.msa0.ss2.a3m
  $HH -i $WDIR/t000_.msa0.ss2.a3m -o $WDIR/t000_.hhr -atab $WDIR/t000_.atab -v 0 > $WDIR/log/hhsearch.stdout 2> $WDIR/log/hhsearch.stderr
fi

############################################################
# 4. custom prediction (writes outprefix.npz only)
############################################################
if [ ! -s $WDIR/t000_.custom.npz ]
then
  echo "Running custom prediction"
  python $PIPEDIR/network/predict_custom.py \
    -m $PIPEDIR/weights \
    -i $WDIR/t000_.msa0.a3m \
    -o $WDIR/t000_.custom \
    --hhr $WDIR/t000_.hhr \
    --atab $WDIR/t000_.atab \
    --db $DB 1> $WDIR/log/network.stdout 2> $WDIR/log/network.stderr
    --custom-t2d /path/to/t2d.npy
fi

echo "Done"
