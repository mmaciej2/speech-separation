#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage:"
  echo "$0 <dataset>"
  exit 1;
fi

dset=$1

mkdir -p data/$dset

case $dset in
  combo*)
    sets="wsj_tr chime5_ct_train chime5_U01_train mixer6_CH02_tr mixer6_CH09_tr"
    for source_set in $sets; do
      if [ ! -d data/$source_set ]; then
        echo "The combo dataset uses selections from the following sets:"
        echo "  $sets"
        echo "You must create those datasets first"
        exit 1;
      fi
    done
    rm -f data/$dset/wav.scp
    for source_set in $sets; do
      while read line; do
        grep -e "$line" data/$source_set/wav.scp >> data/$dset/wav.scp
      done < id_lists/${dset}.txt
    done
    ;;
  *)
    source id_lists/path.sh
    getID $dset
    wavdir="${wav_dirs[set_id]}/mix/"
    awk -v wavdir="$wavdir" '{print $0" "wavdir $0".wav"}' id_lists/${dset}.txt > data/$dset/wav.scp
    ;;
esac

awk '{print $1" 2"}' data/$dset/wav.scp > data/$dset/reco2num_spk
