task=put_rubbish_in_bin
data_dir=./dataset2/raw/0/$task/variation0/episodes
output_dir=./dataset2/packaged/0/$task+0
episodes=99

for i in `seq 93 $episodes`
do
    description_path=$data_dir/episode$i/description.txt
    output_path=$output_dir/instruction$i.txt
    python notebook/chatGPT.py $description_path $output_path
done
