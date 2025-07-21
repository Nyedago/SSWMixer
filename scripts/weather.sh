if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=Model

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
d_model=672
multi_levels=3
seq_len=672
dropout=0.1
learning_rate=0.0005
patch_len=96
factor=5
J=4

for multi_levels in 3 4 5
do
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
    --enc_in 21 \
    --J $J \
    --train_epochs 30 \
    --patience 10 \
    --multi_levels $multi_levels \
    --d_model $d_model \
    --dropout $dropout \
    --patch_len $patch_len \
    --factor $factor \
    --itr 1 --batch_size 64 --learning_rate $learning_rate
done
done