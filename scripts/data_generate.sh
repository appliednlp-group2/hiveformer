data_dir=./dataset2/raw
output_dir=./dataset2/packaged
task=put_rubbish_in_bin
seed=0
episodes=100

# Generate samples
# python RLBench/tools/dataset_generator.py \
#   --save_path=$data_dir/$seed \
#   --tasks=$task \
#   --image_size=128,128 \
#   --renderer=opengl \
#   --episodes_per_task=$episodes \
#   --variations=1 \
#   --processes=1

python data_gen.py \
  --data_dir=$data_dir/$seed \
  --output=$output_dir/$seed \
  --max_variations=1 \
  --tasks=$task

# mkdir ./dataset/instructions
# python preprocess_instructions.py \
# 	--tasks $task \
# 	--output dataset/instructions/$task/instructions.pkl \
# 	--annotations annotations.json \
#     --device "cuda:0"
