python batch_infer.py --batch_size 1 \
        --config models/cldm_ssl_v15_aia_v0_augmentation.yaml \
        --ckpt TISynth_models/GID_model.ckpt \
        --dataset GID26K \
        --ddim_steps 50 \
        --seed 1 \
        --outdir synth_dataset/ \
        --txt_file demo/GID_demo/demo.json \
        --data_root demo/GID_demo