#!/bin/bash

getID(){
  key=$(basename $1)
  case $key in
    wsj_tt.txt) set_id=0;;
    chime5_ct_dev.txt) set_id=1;;
    chime5_U01_dev.txt) set_id=2;;
    mixer6_CH02_tt.txt) set_id=3;;
    mixer6_CH09_tt.txt) set_id=4;;
    *)
      echo "ERROR: Can't get data paths frome filelist. Check filelists/path.sh"
      ;;
  esac
}

mix_spec_dirs=( \
  /export/a15/mmaciej2/enhancement_separation/rsh_net/mask_generation/fixed_shift/specs/2speakers/wav8k/min/tt/mix/ \
  /export/a15/mmaciej2/enhancement_separation/rsh_net/mask_generation/chime5/specs/2speakers/wav8k/min/close_talking_dev_new/mix/ \
  /export/a15/mmaciej2/enhancement_separation/rsh_net/mask_generation/chime5/specs/2speakers/wav8k/min/U01_dev/mix/ \
  /export/a15/mmaciej2/enhancement_separation/rsh_net/mask_generation/mixer6/specs/2speakers/wav8k/min/CH02_tt/mix/ \
  /export/a15/mmaciej2/enhancement_separation/rsh_net/mask_generation/mixer6/specs/2speakers/wav8k/min/CH09_tt/mix/ \
)

wav_dirs=( \
  /export/a15/mmaciej2/data/WSJ-mix/data/2speakers/wav8k/min/tt/ \
  /export/a15/mmaciej2/data/CHiME5-mix/data/2speakers/wav8k/min/close_talking_dev_new/ \
  /export/a15/mmaciej2/data/CHiME5-mix/data/2speakers/wav8k/min/U01_dev/ \
  /export/a15/mmaciej2/data/mixer6-mix/data/2speakers/wav8k/min/CH02_tt/ \
  /export/a15/mmaciej2/data/mixer6-mix/data/2speakers/wav8k/min/CH09_tt/ \
)
