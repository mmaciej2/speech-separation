#!/bin/bash

set -e
. ./cmd.sh

stage=1

arch=TasNet3
train_set=wsj_tr
cv_set=wsj_cv
model_config= # optional config file for model
email=mmaciej2@jhu.edu # set this if you would like qsub email and to run the train command in the background

suffix=

copy_data_to_gpu=true
start_epoch=0
num_epochs=100
batch_size=15
n_debug=-1


train_data_dir=data/$train_set
[ -z "$cv_set" ] || cv_data_dir=data/$cv_set
if [ -z "$email" ]; then
  opt="-sync y"
else
  opt="-M $email"
fi

source activate mm


# Data prep
if [ $stage -le 0 ]; then
  # Note: make sure the filelists/path.sh file contains appropriate information
  #  for each of your datasets
  echo "### Preparing data directories (stage 0) ###"

  for dataset in $train_set $cv_set; do
    local/prepare_data_dir.sh $dataset
  done
fi

# Train model
if [ $stage -le 1 ]; then
  echo "### Training model (stage 1) ###"

  exp_dir=exp/${arch}_${train_set}${suffix}
  [ "$n_debug" -lt "0" ] || exp_dir=${exp_dir}_DBG
  mkdir -p $exp_dir
  cp archs/${arch}.py $exp_dir/arch.py
  [ -z "$model_config" ] || cp $model_config $exp_dir/conf

  qsub -j y -o $exp_dir/train_$(date +%Y%m%d_%H%M%S).log $opt $train_cmd \
    steps/qsub_train.sh \
    $arch \
    $exp_dir \
    $train_data_dir \
    --cv-datadir "$cv_data_dir" \
    --n-debug "$n_debug" \
    --model-config "$model_config" \
    --copy-data-to-gpu "$copy_data_to_gpu" \
    --start-epoch "$start_epoch" \
    --num-epochs "$num_epochs" \
    --batch-size "$batch_size"
fi
