#!/bin/bash

set -e

if [ $# -le 0 ]; then
  echo "Usage:"
  echo "$0 <data-dir> [opts]"
  echo "optional arguments:"
  echo "  --nj (num jobs)         <1>"
  echo "  --mj (max concur. jobs) <20>"
  echo "  --hard-mask             <false>"
  echo "  --fft-dim               <512>"
  echo "  --step-size             <128>"
  echo "  --sample-rate           <8000>"
  exit 1;
fi

data_dir=$1
shift 1
nj=1
mj=20
hard_mask=false
fft_dim=512
step_size=128
sample_rate=8000

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

[ "$hard_mask" != true ] || mask_opt="--hard-mask"
if [ "$hard_mask" == true ]; then
  mask_opt="--hard-mask"
  eval_dir=oracle_hard_mask_eval
else
  mask_opt=""
  eval_dir=oracle_soft_mask_eval
fi

if [ $nj -gt 1 ]; then
  tools/validate_data_dir.sh $data_dir || exit 1;
  tools/split_data_dir.sh $data_dir $nj
  qsub $cpu_cmd -sync y -t 1-$nj -tc $mj -j y -o $data_dir/split$nj/evaluate_oracle.log.\$TASK_ID \
    steps/evaluate_oracle.py $data_dir/split$nj \
      $mask_opt \
      --fft-dim $fft_dim \
      --step-size $step_size \
      --sample-rate $sample_rate
  sleep 15
  rm -rf $data_dir/$eval_dir
  mkdir -p $data_dir/$eval_dir
  for score_type in session source; do
    for metric in SDR SIR SAR; do
      for i in $(seq 1 $nj); do
        cat $data_dir/split$nj/$eval_dir/${score_type}_${metric}s.txt.$i >> $data_dir/$eval_dir/${score_type}_${metric}s.tmp
      done
      sort -u $data_dir/$eval_dir/${score_type}_${metric}s.tmp > $data_dir/$eval_dir/${score_type}_${metric}s.txt
      rm $data_dir/$eval_dir/${score_type}_${metric}s.tmp
    done
  done
else
  $run_cmd steps/evaluate_oracle.py $data_dir \
    $mask_opt \
    --fft-dim $fft_dim \
    --step-size $step_size \
    --sample-rate $sample_rate
  sleep 15
fi

for metric in SDR SIR SAR; do
  cut -d' ' -f2- ${data_dir}/$eval_dir/source_${metric}s.txt | sed -e 's/ /\n/g' | \
    awk -v f_out="${data_dir}/$eval_dir/${metric}_stats.txt" \
      'BEGIN {max = -100; min = 100} {x += $0; y += $0^2; if($0 > max) {max = $0}; if($0 < min) {min = $0}} END {printf("Mean:\t%f\nStd:\t%f\nMax:\t%f\nMin:\t%f\n", x/NR, sqrt(y/NR-(x/NR)^2), max, min) > f_out}'
done
