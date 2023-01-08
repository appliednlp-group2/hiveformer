file_name=instructions_chatgpt3.pkl
task=put_rubbish_in_bin
input_dir=dataset2/packaged/0/$task+0

mkdir ./dataset2/instructions
python preprocess_instructions.py \
	--tasks $task \
	--output ./dataset2/instructions/$task/$file_name \
	--annotations annotations.json \
	--episodes 100 \
	--input_dir $input_dir \
	--encoder "clip" \
    --device "cuda:1" \
	--model_max_length 60
