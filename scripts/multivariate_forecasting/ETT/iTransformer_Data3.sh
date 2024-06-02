export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path Zone1_data.csv \
  --model_id Zone1_data_13_1 \
  --model $model_name \
  --data Data1 \
  --features M \
  --seq_len 13 \
  --pred_len 1 \
  --e_layers 2 \
  --enc_in 196 \
  --dec_in 196 \
  --c_out 196 \
  --des 'Exp' \
  --d_model 196 \
  --d_ff 196 \
  --batch_size 1 \
  --freq m \
  --itr 1
