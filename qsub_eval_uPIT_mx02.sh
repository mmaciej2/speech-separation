#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -M mmaciej2@jhu.edu
#$ -m eas
#$ -l gpu=1,hostname=!c06*&b1[1234589]*|c*|b2*,ram_free=8G,mem_free=8G,h_rt=72:00:00
#$ -r no
set -e
device=`free-gpu`
source filelists/path.sh


arch=uPIT
modeldir=upit_mx02
model=model-final
filelists="filelists/mixer6_CH02_tt.txt filelists/mixer6_CH09_tt.txt"


echo "Working on machine $HOSTNAME"

for filelist in $filelists; do
  filename=$(basename $filelist)
  name="${filename%.*}"
  echo "*** Working on $filename ***"

  getID $filename

  dirout=output-${model##*-}_$name
  gpu_dirout=/export/${HOSTNAME}/mmaciej2/$modeldir/$dirout
  mkdir -p $gpu_dirout/s1 $gpu_dirout/s2

  echo "Generating masks"
  python3 eval_qsub.py $arch $device $modeldir/$model $filelist $gpu_dirout

  echo "Generating wav files"
  python tools/eval_to_wav.py ${mix_spec_dirs[set_id]} $gpu_dirout $modeldir/$dirout/wav

  echo "Computing SDR"
  python tools/eval_sdr.py $modeldir/$dirout/wav ${wav_dirs[set_id]}

  echo ""
  rm -r $gpu_dirout
done
