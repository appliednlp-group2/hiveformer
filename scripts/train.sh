dataset_dir=./dataset/packaged
task=put_rubbish_in_bin
seed=0

python train.py \
	--tasks $task \
	--dataset $dataset_dir/$seed \
	--num_workers 2  \
 	--instructions ./dataset/instructions/$task/instructions.pkl \
	--variations 0 \
    --device "cuda:0" \
	--train_iters 10000