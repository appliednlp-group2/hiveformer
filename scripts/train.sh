dataset_dir=./dataset/packaged
task=put_rubbish_in_bin
seed=0

python train.py \
	--tasks $task \
	--dataset $dataset_dir/$seed \
	--num_workers 5  \
 	--instructions ./dataset/instructions/$task/instructions_fix.pkl \
	--variations 0 \
    --device "cuda:1" \
	--train_iters 100000 --add_pos_emb