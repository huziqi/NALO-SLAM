import numpy as np
import os
import random

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root)
        print(dirs)
        
    files.sort()
    fullpath=[root+i for i in files]
    return fullpath

if __name__ == '__main__':
    img_dir="/home/hzq/dso_run/07/images/"
    gt_dir="/home/hzq/dso_run/07/new_masks/"
    img_path=file_name(img_dir)
    gt_path=file_name(gt_dir)
    path=[img_path, gt_dir]
    print(path)

    with open("/home/hzq/bts/train_test_inputs/samples07.txt","w") as f:
        temp_str=[img_path[i]+" "+gt_path[i]+"\n" for i in range(0,len(img_path))]
        for i in range(0,len(temp_str)):
            f.write(temp_str[i])