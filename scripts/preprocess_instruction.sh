instruction_dir=./hiveformer_dataset/multi_task_dataset/instructions
tasks=(basketball_in_hoop slide_block_to_target wipe_desk lamp_off close_drawer turn_tap take_usb_out_of_computer turn_oven_on close_microwave)


mkdir $instruction_dir
python preprocess_instructions.py \
  --tasks ${tasks[@]} \
  --output $instruction_dir \
  --annotations annotations.json \
  --device "cuda:0"
#   --fixed
