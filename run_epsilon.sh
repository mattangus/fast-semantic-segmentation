#!/bin/bash
echo --config_path "$1"
echo --eval_dir "$2"
echo --trained_checkpoint "$3"
echo output "$2"/mahal_eps_"*"

read -r -p "Are you sure? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
    echo "ok then!"
else
    exit
fi

for ITEM in 0.00002,0.00004,0.00006,0.00008 0.00010,0.00012,0.00014,0.00016 0.00018,0.00020,0.00022,0.00024 0.00026,0.00028,0.00030,0.00032 0.00034,0.00036,0.00038,0.00040
do

    OLDIFS=$IFS
    IFS=','
    read eps1 eps2 eps3 eps4 <<< "${ITEM}"
    IFS=$OLDIFS

    echo "CUDA_VISIBLE_DEVICES=0,1 python3 -u error_detect.py --config_path ""$1"" --eval_dir ""$2"" --trained_checkpoint ""$3"" --use_train --global_cov --epsilon ""$eps1"" &> ""$2""/mahal_eps_"$eps1".log &"
    CUDA_VISIBLE_DEVICES=0,1 python3 -u error_detect.py --config_path "$1" --eval_dir "$2" --trained_checkpoint "$3" --use_train --global_cov --epsilon "$eps1" &> "$2"/mahal_eps_"$eps1".log &

    echo "CUDA_VISIBLE_DEVICES=2,3 python3 -u error_detect.py --config_path ""$1"" --eval_dir ""$2"" --trained_checkpoint ""$3"" --use_train --global_cov --epsilon ""$eps2"" &> ""$2""/mahal_eps_"$eps2".log &"
    CUDA_VISIBLE_DEVICES=2,3 python3 -u error_detect.py --config_path "$1" --eval_dir "$2" --trained_checkpoint "$3" --use_train --global_cov --epsilon "$eps2" &> "$2"/mahal_eps_"$eps2".log &

    echo "CUDA_VISIBLE_DEVICES=4,5 python3 -u error_detect.py --config_path ""$1"" --eval_dir ""$2"" --trained_checkpoint ""$3"" --use_train --global_cov --epsilon ""$eps3"" &> ""$2""/mahal_eps_"$eps3".log &"
    CUDA_VISIBLE_DEVICES=4,5 python3 -u error_detect.py --config_path "$1" --eval_dir "$2" --trained_checkpoint "$3" --use_train --global_cov --epsilon "$eps3" &> "$2"/mahal_eps_"$eps3".log &

    echo "CUDA_VISIBLE_DEVICES=6,7 python3 -u error_detect.py --config_path ""$1"" --eval_dir ""$2"" --trained_checkpoint ""$3"" --use_train --global_cov --epsilon ""$eps4"" &> ""$2""/mahal_eps_"$eps4".log &"
    CUDA_VISIBLE_DEVICES=6,7 python3 -u error_detect.py --config_path "$1" --eval_dir "$2" --trained_checkpoint "$3" --use_train --global_cov --epsilon "$eps4" &> "$2"/mahal_eps_"$eps4".log &

    echo "waiting for $eps1 - $eps4"
    wait
    echo "done waiting"

done
