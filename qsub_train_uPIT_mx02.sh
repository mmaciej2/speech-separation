#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M mmaciej2@jhu.edu
#$ -m eas
#$ -l gpu=1,hostname=(b1[123456789]*|c0*|b2*)&!c06*&!c09*,ram_free=8G,mem_free=8G
#$ -r no
set -e
device=`free-gpu`


arch=uPIT
dirout=upit_mx02
train_filelist=filelists/mixer6_CH02_tr.txt
cv_filelist=filelists/mixer6_CH02_cv.txt
copy_data_to_gpu=true


start_epoch=0
num_epochs=200


echo "args:"
echo "  arch: $arch"
echo "  dirout: $dirout"
echo "  train_filelist: $train_filelist"
echo "  cv_filelist: $cv_filelist"
echo "  copy_data_to_gpu: $copy_data_to_gpu"
echo ""
if [ "$copy_data_to_gpu" = true ]; then
  filename=$(basename $train_filelist)
  name="${filename%.*}"
  datadir=/export/${HOSTNAME}/mmaciej2/tmp_$name
  echo "Using temp data directory: $datadir"
fi
if [ -d "$datadir" ]; then
  echo "Temp data dir already exists, aborting. If a script is already running"
  echo "  on this machine, try again. Otherwise, this is likely stranded data"
  echo "  from a failed run and can be deleted"
  exit 1;
fi
echo "Working on machine $HOSTNAME"
mkdir -p $dirout/intermediate_models $dirout/plots /export/${HOSTNAME}/mmaciej2

python3 train_qsub.py $arch $device $train_filelist $dirout \
                      --cv-filelist "$cv_filelist" \
                      --train-copy-location "$datadir" \
                      --start-epoch $start_epoch \
                      --num-epochs $num_epochs

if [ "$copy_data_to_gpu" = true ]; then rm -r $datadir; fi
