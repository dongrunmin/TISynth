#!/bin/bash
python post_process/resample.py \
        --filtered-mask-path filtered_labels \
        --syn-img-path synth_dataset \
        --out-dir resampled_dataset

