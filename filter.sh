#!/bin/bash
python post_process/filter_gid.py \
    demo/mmseg/base_config.py \
    mmseg_model_ckpt_path \
    --real-img-path FUSU/img_train \
    --real-mask-path FUSU/label_train \
    --syn-img-path synth_dataset \
    --syn-mask-path FUSU/label_train \
    --filtered-mask-path filtered_labels \
    --batch_size 8

