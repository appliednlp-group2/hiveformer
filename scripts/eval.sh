CMDNAME=`basename $0`
if [ $# -ne 6 ]; then
  echo "Usage: $CMDNAME dataset_dir task seed instruction headless device" 1>&2
  exit 1
fi
# sh scripts/eval.sh /root/hiveformer_dataset/multi_task_dataset turn_tap 0 instructions_fixed 0 cuda:2
dataset_dir=$1
task=$2
seed=$3
instruction=$4
headless=$5
device=$6

checkpoint_dir=./xp/hiveformer/$instruction/$task/$seed
instructions_path=$dataset_dir/instructions/$task/$instruction.pkl

python eval_fixed.py \
	--tasks $task \
	--checkpoint_dir $checkpoint_dir \
	--instructions $instructions_path \
	--num_demos 50 \
    --device $device \
	--num_words 53 \
	--instr_size 512 \
	--seed $seed \
	--steps 50 100 150 200 250 300 350 400 450 500 \
	$([ "$headless" = 1 ] && echo "--headless")

# 50 100 150 200 250 300 350 400 450 500 \