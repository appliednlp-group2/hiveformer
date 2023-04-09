#! /bin/bash

CMDNAME=`basename $0`
if [ $# -ne 12 ]; then
  echo "Usage: $CMDNAME -d dataset_dir -t tasks -s seed -v variations -i instruction -e device" 1>&2
  exit 1
fi

while getopts d:t:s:v:i:e: OPT
do
  case $OPT in
    d) 
        dataset_dir=$OPTARG
        echo "[-d] ($OPTARG)が指定された"
    ;;
    t) 
        tasks=$OPTARG
        tasks=($(echo "$tasks" | tr ',' '\n'))
        echo "[-t] ($OPTARG)が指定された"
    ;;
    s)
        seed=$OPTARG
        echo "[-s] ($OPTARG)が指定された"
    ;;
    v) 
        variations=$OPTARG
        echo "[-v] ($OPTARG)が指定された"
    ;;
    i) 
        instruction=$OPTARG
        echo "[-i] ($OPTARG)が指定された"
    ;;
    e) 
        device=$OPTARG
        echo "[-e] ($OPTARG)が指定された"
    ;;
     *) echo "該当なし（OPT=$OPT）";;
  esac
done

: <<'COMMENTOUT'
bash scripts/train_multi.sh \
    -d /root/hiveformer_dataset/multi_task_dataset \
    -t close_door,close_drawer,close_microwave \
    -s 0 \
    -v 0 \
    -i instructions \
    -e cuda:0
COMMENTOUT

echo $dataset_dir, ${tasks[@]}, $seed, $variations, $instruction, $device

CUDA_LAUNCH_BLOCKING=1 python train.py \
	--tasks ${tasks[@]} \
	--dataset $dataset_dir/packaged/0 \
	--num_workers 2  \
 	--instructions $dataset_dir/instructions/$instruction.pkl \
	--variations $variations \
  	--device $device \
	--train_iters 100000 \
	--seed $seed