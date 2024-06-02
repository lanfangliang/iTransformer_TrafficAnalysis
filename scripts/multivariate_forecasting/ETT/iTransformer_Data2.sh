export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path Data2.csv \
  --model_id data2_61_1 \
  --model $model_name \
  --data Data2 \
  --features M \
  --seq_len 61 \
  --pred_len 1 \
  --e_layers 2 \
  --enc_in 2 \
  --dec_in 2 \
  --c_out 2 \
  --des 'Exp' \
  --d_model 4 \
  --d_ff 4 \
  --batch_size 1 \
  --freq t \
  --itr 1
