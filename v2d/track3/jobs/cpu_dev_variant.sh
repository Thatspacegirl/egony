#!/bin/bash
# DEV (public): assemble + official score of one FoundationPose variant root for all 20 public episodes.
#   bash cpu_dev_variant.sh NAME FPOSE_ROOT        -> devscore/pred_1009c_NAME/score.txt (per episode + total + CD-O)
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3
NAME=$1; FR=$2
PRED=/mnt/secondary/v2d/t3/devscore/pred_1009c_$NAME; rm -rf $PRED
PRED=$PRED ASM_EXTRA="--clamp_static" PER_EP=1 CDO=1 TAILN=22 FPOSE_DIR_ROOT=$FR bash $T/t3_dev_eval.sh \
   0 2 11 12 13 18 21 22 23 24 26 27 31 39 40 41 42 43 44 45
