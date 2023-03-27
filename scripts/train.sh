CMDNAME=`basename $0`
if [ $# -ne 5 ]; then
  echo "Usage: $CMDNAME dataset_dir task seed variations device" 1>&2
  exit 1
fi
# sh scripts/train.sh /root/hiveformer_dataset/multi_task_dataset put_rubbish_in_bin 0 0 cuda:0 
dataset_dir=$1
task=$2
seed=$3
variations=$4
device=$5

CMDNAME=`basename $0`

echo $dataset_dir, $task, $seed, $variations, $device

CUDA_LAUNCH_BLOCKING=1 python train.py \
	--tasks $task \
	--dataset $dataset_dir/packaged/0 \
	--num_workers 2  \
 	--instructions $dataset_dir/instructions/$task/instructions.pkl \
	--variations $variations \
  	--device $device \
	--train_iters 100000 \
	--seed $seed