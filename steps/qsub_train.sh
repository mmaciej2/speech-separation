#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -m eas
#$ -l gpu=1
#$ -r no
set -e
device=`free-gpu`


# This is root directory for copying data onto the local machine that
#   the GPU job is running on
gpu_data_dir=/export/${HOSTNAME}/mmaciej2

if [ $# -le 3 ]; then
  echo "Usage:"
  echo "$0 <arch> <dir_out> <train_datadir> [opts]"
  echo "optional arguments:"
  echo "  --cv-datadir"
  echo "  --model-config"
  echo "  --copy-data-to-gpu  <true>"
  echo "  --start-epoch       <0>"
  echo "  --num-epochs        <200>"
  echo "  --batch-size        <100>"
  exit 1;
fi

arch=$1
dirout=$2
train_datadir=$3
shift 3
copy_data_to_gpu=true
start_epoch=0
num_epochs=200
batch_size=100

echo "args:"
echo "  arch: $arch"
echo "  dirout: $dirout"

# Parse optional arguments
while true; do
  [ -z "${1:-}" ] && break;
  case "$1" in
    --*) name=$(echo "$1" | sed 's/--//g' | sed 's/-/_/g')
      printf -v $name "$2"
      echo "  $name: $2"
      shift 2
      ;;
    *) echo "ERROR: malformed arguemnts"
      exit 1
      ;;
  esac
done
echo ""

echo "Working on machine $HOSTNAME"
if [ "$copy_data_to_gpu" = true ]; then
  name=$(basename $train_datadir)
  mkdir -p $gpu_data_dir
  datadir=$gpu_data_dir/tmp_${name}_train
  echo "Using temp data directory: $datadir"
fi
if [ -d "$datadir" ]; then
  echo "Temp data dir already exists, aborting. If a script is already running"
  echo "  on this machine, try again. Otherwise, this is likely stranded data"
  echo "  from a failed run and can be deleted"
  exit 1;
fi
mkdir -p $dirout/intermediate_models $dirout/train_stats/plots
for file in $dirout/train_stats/{train,cv}_loss.txt; do
  touch $file
  awk -v ep=$start_epoch '{if($1<=ep){print $0}}' $file > ${file}.tmp
  mv ${file}.tmp $file
done

python3 steps/train_qsub.py $arch $device $train_datadir $dirout \
                            --cv-data-dir "$cv_datadir" \
                            --model-config "$model_config" \
                            --train-copy-location "$datadir" \
                            --start-epoch $start_epoch \
                            --num-epochs $num_epochs \
                            --batch-size $batch_size

if [ "$copy_data_to_gpu" = true ]; then rm -r $datadir; fi
