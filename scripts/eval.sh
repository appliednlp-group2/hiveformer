dataset_dir=../shota.takashiro/hiveformer_dataset/multi_task_dataset
task=close_door
task=put_rubbish_in_bin
checkpoint_dir=./xp/hiveformer/version6
instructions_dir=./dataset/instructions/$task/instructions.pkl
seed=0

python eval_fixed.py \
	--tasks $task \
	--checkpoint_dir $checkpoint_dir \
	--instructions $dataset_dir/instructions/$task/instructions.pkl \
	--num_demos 50 \
    --device "cuda:0" \
	--num_words 53 \
	--instr_size 512 \
	--seed $seed \
	--headless
