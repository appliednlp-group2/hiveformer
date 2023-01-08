checkpoint_dir=./xp/hiveformer/version3
task=put_rubbish_in_bin
instructions_path=./dataset2/instructions/$task/instructions_chatgpt3.pkl

python eval_fixed.py \
	--tasks $task \
	--dataset $dataset_dir/$seed \
	--num_workers 5  \
 	--instructions $instructions_path \
	--variations 0 \
    --device "cuda:1" \
	--checkpoint_dir $checkpoint_dir \
	--num_words 60 \
	--instr_size 512 \
	--chatgpt