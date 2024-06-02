export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path Data1.csv \
  --model_id Data1_13_1 \
  --model $model_name \
  --data Data1 \
  --features M \
  --seq_len 13 \
  --pred_len 1 \
  --e_layers 2 \
  --enc_in 3417 \
  --dec_in 3417 \
  --c_out 3417 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 1 \
  --freq m \
  --itr 1
