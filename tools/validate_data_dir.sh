#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage:"
  echo "$0 <data-dir>"
  exit 1;
fi

dir=$1

if [ ! -f $dir/segments ]; then
  echo "Using wav.scp for utterance list"
  utt_file="$dir/wav.scp"
  if [ ! -f $dir/wav.scp ]; then
    echo "ERROR: No wav.scp file"
    exit 1;
  fi
else
  echo "Using segments for utterance list"
  utt_file="$dir/segments"
  if [ ! -f $dir/wav.scp ]; then
    echo "WARNING: no wav.scp file"
  else
    nl=$(diff <( cut -d' ' -f1 $dir/wav.scp | sort -u ) <( cut -d' ' -f2 $dir/segments | sort -u ) | wc -l)
    if [ "$nl" -ne "0" ]; then
      echo "segments does not match wav.scp"
      exit 1;
    fi
  fi
fi

if [ -f $dir/feats*.scp ]; then
  feats=$(ls $dir/feats*.scp | awk -F'/' '{print $NF}')
fi

for file in $feats utt2num_spk utt2spk; do
  if [ -f $dir/$file ]; then
    nl=$(diff <(cut -d' ' -f1 $utt_file) <(cut -d' ' -f1 $dir/$file) | wc -l)
    if [ "$nl" -ne "0" ]; then
      echo "$file does not match wav.scp"
      exit 1;
    fi
  fi
done

echo "Data directory $dir is OK."
