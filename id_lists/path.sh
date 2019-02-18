#!/bin/bash

getID(){
  key=$1
  case $key in
    mixer6_CH02_cv) set_id=0;;
    mixer6_CH02_tr) set_id=1;;
    mixer6_CH02_tt) set_id=2;;
    mixer6_CH02_tr_100k) set_id=3;;
    wsj_cv) set_id=4;;
    wsj_tr) set_id=5;;
    wsj_tt) set_id=6;;
    wsj_cv3) set_id=7;;
    wsj_tr3) set_id=8;;
    wsj_tt3) set_id=9;;
    *)
      echo "ERROR: Can't get data paths frome filelist. Check id_lists/path.sh"
      ;;
  esac
}

wav_dirs=( \
  /expscratch/mmaciejewski/mixer6-mix/data/2speakers/wav8k/min/CH02_cv/ \
  /expscratch/mmaciejewski/mixer6-mix/data/2speakers/wav8k/min/CH02_tr/ \
  /expscratch/mmaciejewski/mixer6-mix/data/2speakers/wav8k/min/CH02_tt/ \
  /expscratch/mmaciejewski/mixer6-mix/data/2speakers/wav8k/min/CH02_tr_100k/ \
  /expscratch/mmaciejewski/wsj0-mix/data/2speakers/wav8k/min/cv \
  /expscratch/mmaciejewski/wsj0-mix/data/2speakers/wav8k/min/tr \
  /expscratch/mmaciejewski/wsj0-mix/data/2speakers/wav8k/min/tt \
  /expscratch/mmaciejewski/wsj0-mix/data/3speakers/wav8k/min/cv \
  /expscratch/mmaciejewski/wsj0-mix/data/3speakers/wav8k/min/tr \
  /expscratch/mmaciejewski/wsj0-mix/data/3speakers/wav8k/min/tt \
)
