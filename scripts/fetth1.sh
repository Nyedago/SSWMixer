if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=Model
root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
d_models=(128 512 512 512)
patch_level=5
seq_lens=(96 96 96 96)
pred_lens=(96)
dropouts=(0.1 0.05 0.1 0.05)
factor=(3 5 5 5)
J=(4 4 3 3)
learning_rates=(0.003700176 0.00052 0.00053 0.00054)

for i in "${!pred_lens[@]}"; do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'${seq_lens[$i]}'_'${pred_lens[$i]} \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len ${seq_lens[$i]} \
    --pred_len ${pred_lens[$i]} \
    --enc_in 7 \
    --J ${J[$i]} \
    --train_epochs 30 \
    --patience 10 \
    --patch_level $patch_level \
    --d_model ${d_models[$i]} \
    --dropout ${dropouts[$i]} \
    --patch_len ${seq_lens[$i]} \
    --factor ${factor[$i]} \
    --itr 1 --batch_size 256 --learning_rate ${learning_rates[$i]}
done
