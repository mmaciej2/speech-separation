#!/bin/bash

set -e
. ./cmd.sh

stage=1

model_dir=exp/TasNet2_wsj_tr
test_sets="wsj_tt"
email= # set this if you would like qsub email

featdir=`pwd`/feats
batch_size=30
intermediate_model_num=  # set this to the intermediate model to use it over final


test_data_dirs=""
exp_dirs=""
if [ -z "$intermediate_model_num" ]; then
  model=best
else
  model=$intermediate_model_num
fi
[ ! -f "$model_dir/conf" ] || model_config="$model_dir/conf"
[ -z "$email" ] || email_opt="-M $email"

source activate mm


# Data prep
if [ $stage -le 0 ]; then
  # Note: make sure the filelists/path.sh file contains appropriate information
  #  for each of your datasets
  echo "### Preparing data directories (stage 0) ###"

  for test_set in $test_sets; do
    local/prepare_data_dir.sh $test_set
  done
fi

# Generate masks
if [ $stage -le 1 ]; then
  echo "### Generating masks (stage 1) ###"

  for test_set in $test_sets; do
    test_data_dirs="$test_data_dirs data/$test_set"
    max_num_spk=$(awk 'BEGIN{max=0}{if($2>max){max=$2}}END{print max}' data/$test_set/reco2num_spk)
    exp_dir=$model_dir/output_$model/$test_set
    for i in $(seq 1 $max_num_spk); do
      mkdir -p $exp_dir/wav/s$i
    done
    cp data/$test_set/reco2num_spk data/$test_set/utt2num_spk
  done

  qsub -sync y -j y -o $model_dir/output_$model/eval_$(date +%Y%m%d_%H%M%S).log $email_opt $eval_cmd \
    steps/qsub_eval_e2e.sh \
    $model_dir \
    $test_data_dirs \
    --intermediate-model-num "$intermediate_model_num" \
    --model-config "$model_config" \
    --batch-size $batch_size
fi

# Evaluate estimated sources
if [ $stage -le 2 ]; then
  echo "### Evaluating estimated sources (stage 2) ###"

  for test_set in $test_sets; do
    exp_dir=$model_dir/output_$model/$test_set
    mkdir -p $exp_dir/results

    steps/evaluate_sources.py data/$test_set $exp_dir

    head -n 1 $exp_dir/results/SDR_stats.txt |\
      awk -v dset=$test_set '{printf("%s mean SDR: %0.2f\n", dset, $2)}'
  done
fi
