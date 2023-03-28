task=turn_tap
cuda=cuda:2
instruction=instructions_fixed
headless=1
for seed in 0 1 2
do 
    nohup bash scripts/eval.sh /root/hiveformer_dataset/multi_task_dataset $task $seed $instruction $headless $cuda > xp/hiveformer/$instruction/$task/$seed/eval.txt &
done