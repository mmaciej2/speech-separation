#!/bin/bash

getID(){
  key=$1
  case $key in
    wsj_cv) set_id=0;;
    wsj_tr) set_id=1;;
    wsj_tt) set_id=2;;
    chime5_ct_dev) set_id=3;;
    chime5_ct_train) set_id=4;;
    chime5_U01_dev) set_id=5;;
    chime5_U01_train) set_id=6;;
    mixer6_CH02_cv) set_id=7;;
    mixer6_CH02_tr) set_id=8;;
    mixer6_CH02_tr_100k) set_id=9;;
    mixer6_CH02_tt) set_id=10;;
    mixer6_CH09_cv) set_id=11;;
    mixer6_CH09_tr) set_id=12;;
    mixer6_CH09_tr_100k) set_id=13;;
    mixer6_CH09_tt) set_id=14;;
    *)
      echo "ERROR: Can't get data paths frome filelist. Check id_lists/path.sh"
      ;;
  esac
}

wav_dirs=( \
  /export/a15/mmaciej2/data/WSJ-mix/data/2speakers/wav8k/min/cv/ \
  /export/a15/mmaciej2/data/WSJ-mix/data/2speakers/wav8k/min/tr/ \
  /export/a15/mmaciej2/data/WSJ-mix/data/2speakers/wav8k/min/tt/ \
  /export/a15/mmaciej2/data/CHiME5-mix/data/2speakers/wav8k/min/close_talking_dev_new/ \
  /export/a15/mmaciej2/data/CHiME5-mix/data/2speakers/wav8k/min/close_talking_train/ \
  /export/a15/mmaciej2/data/CHiME5-mix/data/2speakers/wav8k/min/U01_dev/ \
  /export/a15/mmaciej2/data/CHiME5-mix/data/2speakers/wav8k/min/U01_train/ \
  /export/a15/mmaciej2/data/mixer6-mix/data/2speakers/wav8k/min/CH02_cv/ \
  /export/a15/mmaciej2/data/mixer6-mix/data/2speakers/wav8k/min/CH02_tr/ \
  /export/a15/mmaciej2/data/mixer6-mix/data/2speakers/wav8k/min/CH02_tr_100k/ \
  /export/a15/mmaciej2/data/mixer6-mix/data/2speakers/wav8k/min/CH02_tt/ \
  /export/a15/mmaciej2/data/mixer6-mix/data/2speakers/wav8k/min/CH09_cv/ \
  /export/a15/mmaciej2/data/mixer6-mix/data/2speakers/wav8k/min/CH09_tr/ \
  /export/a15/mmaciej2/data/mixer6-mix/data/2speakers/wav8k/min/CH09_tr_100k/ \
  /export/a15/mmaciej2/data/mixer6-mix/data/2speakers/wav8k/min/CH09_tt/ \
)
