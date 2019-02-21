#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Usage:"
  echo "$0 <data-dir> <num-split>"
  exit 1;
fi

dir=$1
ns=$2

rm -rf $dir/split$ns
mkdir -p $dir/split$ns

nl=`wc -l $dir/wav.scp`

for file in wav.scp; do
  awk -v dir="$dir/split$ns" -v file="$file" -v ns="$ns" -v nl="$nl" '{print($0) > dir"/"file"."(int((NR-1)*ns/nl)+1)}' $dir/$file
done

for file in segments; do
  awk -v dir="$dir/split$ns" -v file="$file" -v ns="$ns" -v nl="$nl" '{if ($2 != prev) {prev = $2; nw += 1}; print($0) > dir"/"file"."(int((nw-1)*ns/nl)+1)}' $dir/$file
done
