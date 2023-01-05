file_name=instructions.pkl
task=put_rubbish_in_bin

mkdir ./dataset/instructions
python preprocess_instructions.py \
	--tasks $task \
	--output ./dataset/instructions/$task/$file_name \
	--annotations annotations.json \
    --device "cuda:1"
