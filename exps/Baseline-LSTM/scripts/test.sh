cd .. 
source activate PartAssembly

CUDA_VISIBLE_DEVICES=1 python ./test.py  \
    --exp_suffix '' \
    --model_version 'model' \
    --category 'Chair' \
    --train_data_fn 'Chair.train.npy' \
    --val_data_fn "val_filelist.npy" \
    --device cuda:0 \
    --model_dir "" \
    --level 3 \
    --batch_size 2 \
    --num_batch_every_visu 0 
