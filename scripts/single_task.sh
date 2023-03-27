# tasks=(basketball_in_hoop slide_block_to_target wipe_desk lamp_off close_drawer turn_tap take_usb_out_of_computer turn_oven_on close_microwave)

task=slide_block_to_target
cuda=(cuda:0 cuda:0 cuda:1)
for seed in 0 1 2
do 
    mkdir xp/hiveformer/$task/$seed
    nohup sh scripts/train.sh /root/hiveformer_dataset/multi_task_dataset $task $seed 0 ${cuda[$seed]} > xp/hiveformer/$task/$seed/log.txt &
done
# sh scripts/train.sh /root/hiveformer_dataset/multi_task_dataset put_rubbish_in_bin 0 0 cuda:0 