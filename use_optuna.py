import argparse
import os
import torch
from exp.exp_main import Exp_Main
from collections import defaultdict
import random
import numpy as np
import optuna
import datetime

parser = argparse.ArgumentParser(description='WavMixer & other models for Time Series Forecasting')

# basic config
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='WavMixer', help='model name')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# SparseTSF
parser.add_argument('--period_len', type=int, default=24, help='period length')
parser.add_argument('--model_type', default='linear', help='model type: linear/mlp')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=96, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# TimeMixer
parser.add_argument('--down_sampling_layers', type=int, default=2, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')

# Model
parser.add_argument('--wavelet', type = str, default = 'db2', help = 'wavelet type for wavelet transform')
parser.add_argument('--J', type = int, default = 2, help = 'decomposition level for wavelet transform')
parser.add_argument('--factor', type = int, default = 5, help = 'factor for wavelet transform')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
parser.add_argument('--patch_levels', type=int, default=3, help='patch levels')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='learned',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', default=False, help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# Optuna参数
# ----------------------
parser.add_argument('--n_trials', type=int, default=50, help='优化试验次数')
parser.add_argument('--timeout', type=int, default=1800, help='优化超时时间（秒）')
parser.add_argument('--save_best_params', type=str, default='best_params.json', help='最佳参数保存路径')
parser.add_argument('--study_name', type=str, default='wavmixer_optimization', help='Optuna研究名称')
parser.add_argument('--storage', type=str, default=None, help='数据库存储地址（如SQLite路径）')


# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', type=int, help='use multiple gpus', default=0)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# random seed
fix_seed_list = range(2025, 2035)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)
Exp = Exp_Main
class Tuner:
    """时间序列预测模型超参数优化器（指定超参数版本）"""
    def __init__(self, args, ran_seed: int, n_jobs: int = 1):
        self.fixed_seed = ran_seed
        self.n_jobs = n_jobs
        self.result_dic = defaultdict(list)
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.study = None
        self.args = args

    def _objective(self, trial: optuna.Trial, args: argparse.Namespace) -> float:
        """定义目标函数，仅优化指定的超参数"""
        # ----------------------- 指定超参数搜索空间 -----------------------
        # learning_rate（对数均匀分布）
        self.args.learning_rate = trial.suggest_loguniform('learning_rate', low=0.0001, high=0.005)
        # dropout（分类选择）
        self.args.dropout = trial.suggest_categorical('dropout', [0.0, 0.05, 0.1, 0.15, 0.2])
        # J（小波分解层数，整数范围）
        self.args.J = trial.suggest_int('J', low=1, high=4)  # 对应原代码中的level
        # dmodel（模型维度，分类选择）
        self.args.d_model = trial.suggest_categorical('d_model', [128, 256, 512])
        # factor（小波因子，整数范围）
        self.args.factor = trial.suggest_int('factor', low=2, high=4)
        # patch_levels（补丁层级，整数范围）
        self.args.patch_levels = trial.suggest_int('patch_levels', low=1, high=5)
        
        
        # ----------------------- 生成试验标识 -----------------------
        trial_setting = (
            f"{self.args.model}_{self.args.data}_"
            f"lr{self.args.learning_rate:.0e}_dp{self.args.dropout}_"
            f"J{self.args.J}_dm{self.args.d_model}_f{self.args.factor}_pl{self.args.patch_levels}_"
            f"sd{self.fixed_seed}"
        )
        ii=0
        random.seed(fix_seed_list[ii])
        torch.manual_seed(fix_seed_list[ii])
        np.random.seed(fix_seed_list[ii])
        exp = Exp_Main(self.args)
        print(f">>>>>>> 试验 {trial.number} 开始训练: {trial_setting} >>>>>>>>>")
        exp.train(trial_setting)  # 训练（可添加Optuna进度报告）
        print(f">>>>>>> 试验 {trial.number} 开始测试 <<<<<<<<<<")
        return exp.min_test_loss  # 返回最小测试损失（假设由exp.test()计算）

    def tune(self, args: argparse.Namespace, n_trials: int = 50) -> None:
        """执行超参数优化"""
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.fixed_seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )
        wrapped_obj = lambda trial: self._objective(trial, args)
        
        print(f"开始优化(试验次数: {n_trials}, 并行数: {self.n_jobs})")
        self.study.optimize(wrapped_obj, n_trials=n_trials, n_jobs=self.n_jobs)
        print("\n优化完成!最佳损失值:", self.study.best_value)
        self._save_results(args)

    def _save_results(self, args: argparse.Namespace) -> None:
        """保存最佳参数和结果"""
        if not self.study:
            return
        
        best_trial = self.study.best_trial
        self.result_dic.update({
            "data": [args.data],
            "model": [args.model],
            "pred_len": [args.pred_len],
            "best_loss": [best_trial.value],
            "trial_number": [best_trial.number]
        })
        self.result_dic.update(best_trial.params)  # 写入优化的超参数
        
        # 生成文件名
        file_prefix = f"{args.model}_{args.data}_J{args.J}_pl{args.patch_levels}"
        file_name = f"./hyperParameterSearchOutput/{file_prefix}_best_params_{self.current_time}.csv"
        
        # 保存到CSV
        try:
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            pd.DataFrame(self.result_dic).to_csv(file_name, index=False)
            print(f"\n结果已保存至: {file_name}")
            print("最佳参数:")
            for k, v in best_trial.params.items():
                print(f"- {k}: {v}")
        except Exception as e:
            print(f"保存失败: {str(e)}")


### 使用示例
if __name__ == "__main__":
   
    # 初始化优化器（固定随机种子，并行数=CPU核心数）
    tuner = Tuner(args, ran_seed=2025, n_jobs=1)
    tuner.tune(args, n_trials=30)  # 执行30次优化试验