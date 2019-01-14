#!/bin/bash

set -e

stage=4

arch=uPIT
model_dir=exp/uPIT_mixer6_CH02_tr
test_sets="mixer6_CH02_tt mixer6_CH09_tt"
email= # set this if you would like qsub email

featdir=`pwd`/feats
batch_size=100
intermediate_model_num=  # set this to the intermediate model to use it over final


test_data_dirs=""
exp_dirs=""
if [ -z "$intermediate_model_num" ]; then
  model=final
else
  model=$intermediate_model_num
fi
[ -z "$email" ] || email_opt="-M $email"


# Data prep
if [ $stage -le 0 ]; then
  # Note: make sure the filelists/path.sh file contains appropriate information
  #  for each of your datasets
  echo "### Preparing data directories (stage 0) ###"

  for test_set in $test_sets; do
    local/prepare_data_dir.sh $test_set
  done
fi

# Extract features
if [ $stage -le 1 ]; then
  echo "### Extracting features (stage 1) ###"

  for test_set in $test_sets; do
    steps/extract_feats.py data/$test_set "test" $featdir/${test_set}_test
  done
fi

# Generate masks
if [ $stage -le 2 ]; then
  echo "### Generating masks (stage 2) ###"

  for test_set in $test_sets; do
    test_data_dirs="$test_data_dirs data/$test_set"
    mkdir -p $model_dir/output_$model/$test_set/masks
  done

  qsub -sync y -j y -o $model_dir/output_$model/eval_\$JOB_ID.log $email_opt \
    steps/qsub_eval.sh \
    $arch \
    $model_dir \
    $test_data_dirs \
    --intermediate-model-num "$intermediate_model_num" \
    --batch-size $batch_size
fi

# Generate estimated source wav files
if [ $stage -le 3 ]; then
  echo "### Generating estimated source wav files (stage 3) ###"

  for test_set in $test_sets; do
    max_num_spk=$(awk 'BEGIN{max=0}{if($2>max){max=$2}}END{print max}' data/$test_set/utt2num_spk)
    exp_dir=$model_dir/output_$model/$test_set
    for i in $(seq 1 $max_num_spk); do
      mkdir -p $exp_dir/wav/s$i
    done

    steps/reconstruct_sources.py data/$test_set $exp_dir
  done
fi

# Evaluate estimated sources
if [ $stage -le 4 ]; then
  echo "### Evaluating estimated sources (stage 4) ###"

  for test_set in $test_sets; do
    exp_dir=$model_dir/output_$model/$test_set
    mkdir -p $exp_dir/results

    steps/evaluate_sources.py data/$test_set $exp_dir

    head -n 1 $exp_dir/results/SDR_stats.txt |\
      awk -v dset=$test_set '{printf("%s mean SDR: %0.2f\n", dset, $2)}'
  done
fi
