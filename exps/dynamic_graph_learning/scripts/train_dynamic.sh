cd .. 
source activate PartAssembly
CUDA_VISIBLE_DEVICES=1 python ./train_dynamic.py  \
    --exp_suffix '_all' \
    --model_version 'model_dynamic_mlp' \
    --category 'Chair' \
    --train_data_fn 'Chair.train.npy' \
    --val_data_fn 'Chair.val.npy' \
    --loss_weight_trans_l2 1.0 \
    --loss_weight_rot_l2 0.0 \
    --loss_weight_rot_cd 10 \
    --loss_weight_shape_cd 1.0 \
    --device cuda:0 \
    --num_epoch_every_visu 1000 \
    --level 3 \
    --overwrite \
    --lr 1e-3 \
    --batch_size 16 \
    --num_workers 8 \
    --num_batch_every_visu 0 \
