#! /bin/bash

tasks=close_door,close_drawer,close_microwave
cuda=(cuda:0 cuda:1 cuda:2)
instruction=instructions
for seed in 0 1 2
do 
    folder_name=$(echo "$tasks" | sed 's/,/-/g')
    mkdir -p xp/hiveformer/$instruction/$folder_name/$seed
    nohup bash scripts/train_multi.sh \
        -d /root/hiveformer_dataset/multi_task_dataset \
        -t $tasks \
        -s $seed \
        -v 0 \
        -i $instruction \
        -e ${cuda[$seed]} \
        > xp/hiveformer/$instruction/$folder_name/$seed/log.txt &
done
# sh scripts/train.sh /root/hiveformer_dataset/multi_task_dataset put_rubbish_in_bin 0 0 cuda:0 
