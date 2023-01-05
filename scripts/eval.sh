checkpoint_dir=./xp/hiveformer/version12/
instructions_dir=./dataset/instructions/$task/instructions.pkl

python eval.py \
	--checkpoint $checkpoint_dir \
	--instructions instructions.pkl \
	--num_episodes 100 \
    --device "cuda:1"