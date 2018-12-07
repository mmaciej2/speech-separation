#!/bin/bash

getID(){
  key=$(basename $1)
  case $key in
    mixer6_CH02_tt.txt) set_id=0;;
    mixer6_CH09_tt.txt) set_id=1;;
    mixer6_CH02_tt_5.txt) set_id=0;;
    mixer6_CH09_tt_5.txt) set_id=1;;
    *)
      echo "ERROR: Can't get data paths frome filelist. Check filelists/path.sh"
      ;;
  esac
}

mix_spec_dirs=( \
  /export/a15/mmaciej2/enhancement_separation/rsh_net/mask_generation/mixer6/specs/2speakers/wav8k/min/CH02_tt/mix/ \
  /export/a15/mmaciej2/enhancement_separation/rsh_net/mask_generation/mixer6/specs/2speakers/wav8k/min/CH09_tt/mix/ \
)

wav_dirs=( \
  /export/a15/mmaciej2/data/mixer6-mix/data/2speakers/wav8k/min/CH02_tt/ \
  /export/a15/mmaciej2/data/mixer6-mix/data/2speakers/wav8k/min/CH09_tt/ \
)
