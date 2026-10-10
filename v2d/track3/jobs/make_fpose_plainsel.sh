#!/bin/bash
# Build fpose_plainsel/<split>/<episode>/<obj>.npz (symlinks) = the PLAIN FoundationPose variant with the frozen mesh
# policy: SAM 3D pots from fpose_sel/, public boards 23/24 from fpose_board/, everything else from fpose/ (v1; the
# evaluation v1 run already used "board final").  Every npz records the mesh it tracked (key `mesh`).
R=/mnt/secondary/v2d/t3; O=$R/fpose_plainsel
rm -rf $O
for sp in public evaluation; do
  for d in $R/fpose/$sp/episode_*; do
    e=$(basename $d); mkdir -p $O/$sp/$e
    for f in $d/*.npz; do
      o=$(basename $f)
      src=$f
      [ -f $R/fpose_board/$sp/$e/$o ] && src=$R/fpose_board/$sp/$e/$o
      [ -f $R/fpose_sel/$sp/$e/$o ] && src=$R/fpose_sel/$sp/$e/$o
      ln -s $src $O/$sp/$e/$o
    done
  done
done
find $O -name "*.npz" | wc -l
