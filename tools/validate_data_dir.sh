#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage:"
  echo "$0 <data-dir>"
  exit 1;
fi

dir=$1

if [ ! -f $dir/wav.scp ]; then
  echo "No wav.scp file"
  exit 1;
fi

feats=$(ls $dir/feats*.scp | awk -F'/' '{print $NF}')

for file in $feats utt2num_spk utt2spk; do
  if [ -f $dir/$file ]; then
    nl=$(diff <(cut -d' ' -f1 $dir/wav.scp) <(cut -d' ' -f1 $dir/$file) | wc -l)
    if [ "$nl" -ne "0" ]; then
      echo "$file does not match wav.scp"
      exit 1;
    fi
  fi
done
