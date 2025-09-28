python main.py \
    --base configs/v2-ssl_vector_GID_4m.yaml \
    -t \
    --pretrain_path TISynth_models/controlnet1.5.ckpt \
    --config_model models/cldm_ssl_v15_aia_v0_augmentation.yaml \
    -n ... \
    --gpus ... \
    --data_root ... \
    --train_txt_file .. \
    --val_txt_file ...