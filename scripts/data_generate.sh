data_dir=./hiveformer_dataset/multi_task_dataset/raw
output_dir=./hiveformer_dataset/multi_task_dataset/packaged
instruction_dir=./hiveformer_dataset/multi_task_dataset/instructions
tasks=(basketball_in_hoop slide_block_to_target wipe_desk lamp_off close_drawer turn_tap take_usb_out_of_computer turn_oven_on close_microwave)
seed=0
episodes=100

# Generate samples
# python RLBench/tools/dataset_generator.py \
#   --save_path=$data_dir/$seed \
#   --tasks=$tasks \
#   --image_size=128,128 \
#   --renderer=opengl \
#   --episodes_per_task=$episodes \
#   --variations=1 \
#   --processes=8

# for task in ${tasks[@]}
# do
#   python data_gen.py \
#     --data_dir=$data_dir/$seed \
#     --output=$output_dir/$seed \
#     --max_variations=1 \
#     --tasks=$task
# done

mkdir $instruction_dir
python preprocess_instructions.py \
  --tasks ${tasks[@]} \
  --output $instruction_dir \
  --annotations annotations.json \
  --device "cuda:0"

