cd .. 
CUDA_VISIBLE_DEVICES=3 python ./test_dynamic.py  \
    --exp_suffix '' \
    --model_version 'model_dynamic' \
    --category 'Table' \
    --train_data_fn 'Table.train.npy' \
    --val_data_fn "Table.val.npy" \
    --device cuda:0 \
    --model_dir "path_to_the_checkpoints"\
    --level 3 \
    --batch_size 4 \
    --num_batch_every_visu 0 
