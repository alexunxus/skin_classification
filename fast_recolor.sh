#!/bin/bash

target_folder="result/inference_CSV"
#target_folder="/mnt/dataset/demo/vgh_subtype/v200426_post/whole_slide/"

#for sub_folder in "$(ls -d ${target_folder}/)"*
for sub_folder in "${target_folder}/"*
do
echo ${sub_folder##*/}
python3 recolor.py \
	--mask_dir ${target_folder} \
	--slide_name "${sub_folder##*/}"
done



#python recolor.py --mask_dir result/inference_CSV/ --slide_name 2019-10-30\ 01.59.42. --data_dir result/inference_CSV/

