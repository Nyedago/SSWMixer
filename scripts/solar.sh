if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=Model

root_path_name=./dataset/
data_path_name=solar_AL.txt
model_id_name=Solar
data_name=Solar
down_sampling_layers=3
down_sampling_window=2

seq_len=720
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
    --period_len 4 \
    --enc_in 137 \
    --train_epochs 30 \
    --patience 5 \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_method avg \
    --down_sampling_window $down_sampling_window \
    --itr 1 --batch_size 256 --learning_rate 0.02
done