#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

# mkdir /mnt/vepfs/Perception/perception-users
# mkdir /mnt/vepfs/Perception/perception-public
# if ! [ -d "/vepfs" ]; then
#   ln -s /qcraft-vepfs-01 /vepfs
# fi
# if ! [ -d "/mnt/vepfs/Perception/perception-users" ]; then
#   ln -s /qcraft-vepfs-01/Perception/perception-users /mnt/vepfs/Perception/perception-users
# fi
# if ! [ -d "/mnt/vepfs/Perception/perception-public" ]; then
#   ln -s /qcraft-vepfs-01/Perception/perception-public /mnt/vepfs/Perception/perception-public
# fi
# if ! [ -d "/tos/qcraftlabeldata" ]; then
#   ln -s /tos/qcraft/qcraftlabeldata /tos/qcraftlabeldata
# fi

source /mnt/vepfs/Perception/perception-users/hongliang/condanew/anaconda3/bin/activate /mnt/vepfs/Perception/perception-users/hongliang/condanew/anaconda3/envs/monoconv



PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
