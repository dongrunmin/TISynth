from argparse import ArgumentParser
from typing import List
from PIL import Image
from tqdm import tqdm

import shutil
import os
import numpy as np

filtered_mask_path = None
syn_img_path = None
out_syn_img_path = None
out_syn_mask_path = None

cls2id = {
    'traffic land': 0,
    'inland water': 1,
    'residential land': 2,
    'cropland': 3,
    'agriculture construction': 4,
    'blank': 5,
    'industrial land': 6,
    'park': 7,
    'greenbelt': 8,
    'public management and service': 9,
    'commercial land': 10,
    'public construction': 11,
    'special': 12,
    'forest': 13,
    'storage': 14,
    'wetland': 15,
    'grass': 16,
}
id2cls = {
    v: k for k, v in cls2id.items()
}
ignore_index = 255

# Specify the category names that have poor classification results.
inaccurate = ['agriculture construction', 'special', 'public construction', 'commercial land', 'grass']
minor_set = [cls2id[cls_name] for cls_name in inaccurate]

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--filtered-mask-path', type=str, required=True)
    parser.add_argument('--syn-img-path', type=str, required=True)  
    parser.add_argument('--out-dir', type=str, required=True)
    
    args = parser.parse_args()

    filtered_mask_path = args.filtered_mask_path
    syn_img_path = args.syn_img_path
    out_syn_img_path = os.path.join(args.out_dir, 'resampled_img')
    out_syn_mask_path = os.path.join(args.out_dir, 'resampled_label')
    os.makedirs(out_syn_img_path, exist_ok=True)
    os.makedirs(out_syn_mask_path, exist_ok=True)

    for mask_name in tqdm(os.listdir(filtered_mask_path)):
        if os.path.exists(os.path.join(out_syn_mask_path, mask_name)):
            continue
        mask = np.array(Image.open(os.path.join(filtered_mask_path, mask_name)), dtype=np.uint8)
        flag = np.any(np.isin(minor_set, mask))
        if not flag:
            ratio = np.mean(mask != ignore_index)
            flag = ratio > 0.8
        
        if flag:
            shutil.copy(
                os.path.join(filtered_mask_path, mask_name),
                os.path.join(out_syn_mask_path, mask_name)
                )
            
            img_name = mask_name.replace('.png', '.jpg')
            shutil.copy(
                os.path.join(syn_img_path, img_name),
                os.path.join(out_syn_img_path, img_name)
            )
        