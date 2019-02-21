#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -m eas
#$ -l gpu=1
#$ -r no
set -e
source activate mm
module load cuda90/toolkit cuda90/blas cudnn
device=$CUDA_VISIBLE_DEVICES


if [ $# -le 1 ]; then
  echo "Usage:"
  echo "$0 <model_dir> <test_data_dir1> [<test_data_dir2> ...] [opts]"
  echo "optional arguments:"
  echo "  --model-config"
  echo "  --batch-size             <100>"
  echo "  --intermediate-model-num"
  exit 1;
fi

model_dir=$1
test_data_dirs=""
shift 1
batch_size=100

echo "args:"
echo "  model_dir: $model_dir"

# Parse remaining arguments
while true; do
  [ -z "${1:-}" ] && break;
  case "$1" in
    --*) name=$(echo "$1" | sed 's/--//g' | sed 's/-/_/g')
      printf -v $name "$2"
      echo "  $name: $2"
      shift 2
      ;;
    *) test_data_dirs="$test_data_dirs $1"
      echo "  test_data_dir: $1"
      shift 1
      ;;
  esac
done
echo ""


if [ -z "$intermediate_model_num" ]; then
  model=$model_dir/final.mdl
  base_dir_out=$model_dir/output_final
else
  model=$model_dir/intermediate_models/${intermediate_model_num}.mdl
  base_dir_out=$model_dir/output_$intermediate_model_num
fi


echo "Working on machine $HOSTNAME"

for data_dir in $test_data_dirs; do
  dir_out=$base_dir_out/$(basename $data_dir)/masks
  python3 steps/eval_qsub.py $model_dir/arch.py $device $model $data_dir $dir_out \
                             --model-config "$model_config" \
                             --batch-size $batch_size \
                             --model-config "$model_config"
done
