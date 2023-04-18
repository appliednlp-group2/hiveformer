# tasks=close_door,close_drawer,close_microwave
tasks=basketball_in_hoop,slide_block_to_target,wipe_desk,lamp_off,close_drawer,turn_tap,take_usb_out_of_computer,turn_oven_on,close_microwave
cuda=cuda:2
# instruction=instructions
instruction=instructions_fixed
headless=1
folder_name=$(echo "$tasks" | sed 's/,/-/g')
for seed in 0 1 2
do 
    nohup bash scripts/eval.sh /root/hiveformer_dataset/multi_task_dataset $tasks $seed $instruction $headless $cuda > xp/hiveformer/$instruction/$folder_name/$seed/eval.log &
done