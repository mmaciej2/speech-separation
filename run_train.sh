#!/bin/bash

set -e
. ./cmd.sh

stage=0

arch=RSHxv
train_set=mixer6_CH02_1-2spk_tr_40k
cv_set=mixer6_CH02_cv
model_config=conf/RSHxv.conf # optional config file for model
email= # set this if you would like qsub email and to run the train command in the background

featdir=/expscratch/mmaciejewski/enh_sep_feats/speech-separation
suffix=_l$(grep lambda $model_config | awk -F'=' '{print $NF}')
copy_data_to_gpu=true
start_epoch=0
num_epochs=200
batch_size=80


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

# Extract features
if [ $stage -le 1 ]; then
  echo "### Extracting features (stage 1) ###"

  for data_dir in $train_data_dir $cv_data_dir; do
    $run_cmd steps/extract_feats.py $data_dir "train" $featdir/$(basename $data_dir)_train
  done
fi

# Train model
if [ $stage -le 2 ]; then
  echo "### Training model (stage 2) ###"

  exp_dir=exp/${arch}_${train_set}${suffix}
  mkdir -p $exp_dir
  cp archs/${arch}.py $exp_dir/arch.py
  [ -z "$model_config" ] || cp $model_config $exp_dir/conf

  qsub -j y -o $exp_dir/train_\$JOB_ID.log $opt $train_cmd \
    steps/qsub_train.sh \
    $arch \
    $exp_dir \
    $train_data_dir \
    --cv-datadir "$cv_data_dir" \
    --model-config "$model_config" \
    --copy-data-to-gpu "$copy_data_to_gpu" \
    --start-epoch "$start_epoch" \
    --num-epochs "$num_epochs" \
    --batch-size "$batch_size"
fi
