#!/bin/bash

for base_name in /data/infant_brain_seg_reg/*
#do
#echo "$(basename $(file))"
#done
do
	for mo_dir in $base_name/*
	do
	#echo "$mo_dir"
	file_name=$mo_dir/intensity_mask_out.nii.gz
	echo "$file_name"
	/usr/local/fsl/bin/fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -o $mo_dir/intensity_mask_out $mo_dir/intensity_mask_out
	echo "$file_name"
	done
done 
