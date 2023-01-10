task=put_rubbish_in_bin
checkpoint_dir=./xp/hiveformer/version2/
instructions_dir=./dataset/instructions/$task/instructions_fix.pkl

python eval.py \
	--checkpoint $checkpoint_dir \
	--instructions $instructions_dir \
	--num_episodes 100 \
    --device "cuda:1" --add_pos_emb