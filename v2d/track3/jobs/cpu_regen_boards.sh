#!/bin/bash
# 2026-10-09 post-crash: regenerate the board meshes (written 01:0x-01:52 in the crash window) with the same code and
# compare sha256 with checks/recompute_1009/before_evalB_chain.json; old copies in checks/recompute_1009/board_before.
T=/home/asubuntudesktop/TestingGrounds/egony/v2d/track3; L=/mnt/secondary/v2d/t3/logs
for x in "public 23 wooden_piece_1" "public 23 wooden_piece_2" "public 24 wooden_piece_1" "public 24 wooden_piece_2" "evaluation 25 wooden_piece_1" "evaluation 25 wooden_piece_2"; do
  set -- $x
  OMP_NUM_THREADS=2 nice -n 15 /mnt/secondary/v2d/envs/t3-fpose/bin/python -I $T/t3_board_mesh.py --split $1 --episode $2 --obj $3 2>&1 | grep -v static_prefix | tail -3 | sed "s/^/$1 $2 $3: /"
done
python3 $T/t3_hashdirs.py /mnt/secondary/v2d/t3 /mnt/secondary/v2d/t3/checks/recompute_1009/after_board.json meshes/board
python3 - <<'PY'
import json
a=json.load(open('/mnt/secondary/v2d/t3/checks/recompute_1009/before_evalB_chain.json'))
b=json.load(open('/mnt/secondary/v2d/t3/checks/recompute_1009/after_board.json'))
for k in sorted(b):
    print('SAME' if a.get(k)==b[k] else 'DIFF', k)
PY
echo REGEN_DONE
