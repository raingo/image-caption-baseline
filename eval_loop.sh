#!/bin/bash
# vim ft=sh

gpu=$1
export CUDA_VISIBLE_DEVICES=$gpu

function eval {
model_dir=$1

model_name=`basename $model_dir`
model_path=$model_dir/model.ckpt
flock -n $model_path python cnn2lm.py ./data/mm/test.tsv.tf/ $model_dir/vocab $model_path ./data/mm/test.$model_name

}
while :
do
  eval ./data/coco/
  eval ./data/mm/
  sleep 1
done
