CMDNAME=`basename $0`
if [ $# -ne 5 ]; then
  echo "Usage: $CMDNAME dataset_dir task seed variations device" 1>&2
  exit 1
fi
# sh scripts/train.sh ../shota.takashiro/dataset2 put_rubbish_in_bin 0 0 cuda:0
dataset_dir=$1
task=$2
seed=$3
variations=$4
device=$5

CMDNAME=`basename $0`

echo $dataset_dir, $task, $seed, $variations, $device

python train.py \
	--tasks $task \
	--dataset $dataset_dir/packaged/$seed \
	--num_workers 8  \
 	--instructions $dataset_dir/instructions/$task/instructions.pkl \
	--variations $variations \
  	--device $device \
	--train_iters 100000