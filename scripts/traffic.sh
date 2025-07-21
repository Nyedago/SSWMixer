if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=Model

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom
d_model=256
multi_levels=2
seq_len=96
dropout=0.05
learning_rate=0.0005
patch_len=$seq_len
factor=5
J=3


for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 24 \
    --enc_in 862 \
    --J $J \
    --train_epochs 30 \
    --patience 5 \
    --multi_levels $multi_levels \
    --d_model $d_model \
    --dropout $dropout \
    --patch_len $patch_len \
    --factor $factor \
    --use_multi_gpu 1 \
    --itr 1 --batch_size 1 --learning_rate $learning_rate
done