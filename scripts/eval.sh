dataset_dir=../hiveformer_dataset/multi_task_dataset
task=turn_tap
seed=0
checkpoint_dir=./xp/hiveformer/$task/$seed
instructions_path=$dataset_dir/instructions/$task/instructions_fixed.pkl

python eval_fixed.py \
	--tasks $task \
	--checkpoint_dir $checkpoint_dir \
	--instructions $instructions_path \
	--num_demos 50 \
    --device "cuda:0" \
	--num_words 53 \
	--instr_size 512 \
	--seed $seed
