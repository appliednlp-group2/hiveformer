CMDNAME=`basename $0`
if [ $# -ne 6 ]; then
  echo "Usage: $CMDNAME dataset_dir tasks seed instruction headless device" 1>&2
  exit 1
fi
# sh scripts/eval.sh /root/hiveformer_dataset/multi_task_dataset turn_tap 0 instructions_fixed 0 cuda:2
dataset_dir=$1
tasks=$2
folder_name=$(echo "$tasks" | sed 's/,/-/g')
tasks=($(echo "$tasks" | tr ',' '\n'))
seed=$3
instruction=$4
headless=$5
device=$6

checkpoint_dir=./xp/hiveformer/$instruction/$folder_name/$seed
instructions_dir=$dataset_dir/instructions

python eval_fixed.py \
	--tasks ${tasks[@]} \
	--checkpoint_dir $checkpoint_dir \
	--instructions $instructions_dir \
	$([ "$instruction" = "instructions_fixed" ] && echo "--fixed") \
	--num_demos 50 \
    --device $device \
	--num_words 53 \
	--instr_size 512 \
	--seed $seed \
	--steps 50 100 150 200 250 300 350 400 450 500\
	$([ "$headless" = 1 ] && echo "--headless")
# ffmpeg -f x11grab -s 1400x900 -r 25 -i :0.0 slide_block_to_target.mp4
# 50 100 150 200 250 300 350 400 450 500 \
