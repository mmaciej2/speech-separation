#!/bin/bash

set -e

if [ $# -le 2 ]; then
  echo "Usage:"
  echo "$0 <data-dir> <train|test> <feat-storage-dir> [opts]"
  echo "optional arguments:"
  echo "  --nj (num jobs)          <1>"
  echo "  --mj (max concur. jobs)  <20>"
  exit 1;
fi

data_dir=$1
dtype=$2
feat_dir=$3
shift 3
nj=1
mj=20

# Parse optional arguments
while true; do
  [ -z "${1:-}" ] && break;
  case "$1" in
    --*) name=$(echo "$1" | sed 's/--//g' | sed 's/-/_/g')
      printf -v $name "$2"
      shift 2
      ;;
    *) echo "ERROR: malformed arguemnts"
      exit 1
      ;;
  esac
done

if [ $nj -gt 1 ]; then
  tools/split_data_dir.sh $data_dir $nj
  qsub $cpu_cmd -sync y -t 1-$nj -tc $mj -j y -o $data_dir/split$nj/extract_feats.log.$TASK_ID \
    steps/extract_feats.py $data_dir/split$nj $dtype $feat_dir
  rm -f $data_dir/{feats_${dtype}.scp,utt2num_spk}
  for i in $(seq 1 $nj); do
    cat $data_dir/split$nj/feats_${dtype}.scp.$i >> $data_dir/feats_${dtype}.scp
    cat $data_dir/split$nj/utt2num_spk.$i >> $data_dir/utt2num_spk
  done
else
  $run_cmd steps/extract_feats.py $data_dir $dtype $feat_dir
fi
