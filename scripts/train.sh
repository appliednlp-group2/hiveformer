dataset_dir=./dataset2/packaged
task=put_rubbish_in_bin
seed=0

python train.py \
	--tasks $task \
	--dataset $dataset_dir/$seed \
	--num_workers 5  \
 	--instructions ./dataset2/instructions/$task/instructions_chatgpt2.pkl \
	--variations 0 \
    --device "cuda:1" \
	--train_iters 100000 \
	--instr_size 512 \
	--num_words 240