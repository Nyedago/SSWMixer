if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=Model

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom
d_model=128
multi_levels=3
seq_len=672
dropout=0.05
patch_len=$seq_len
factor=5
J=3

for multi_levels in 3 4 5
do
for dropout in 0.05 0.1 0.15
do
for learning_rate in 0.0004 0.0005 0.0006
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
    --enc_in 321 \
    --J $J \
    --train_epochs 30 \
    --patience 5 \
    --multi_levels $multi_levels \
    --d_model $d_model \
    --dropout $dropout \
    --patch_len $patch_len \
    --factor $factor \
    --itr 1 --batch_size 256 --learning_rate $learning_rate
done
done
done
done