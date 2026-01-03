import pandas as pd
import numpy as np
import torch
import sys
from math import sqrt
from torch import optim
import torch.nn as nn
import argparse
import os
import warnings

# Suppress FutureWarning from pandas concat
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils.set_seed import set_seed
from utils.util import plot_forecast
import matplotlib.pyplot as plt

from Dataset.custom_dataset import get_dataloader, load_manufacturing_data
from Chronos import ChronosForecaster

pd.set_option('display.max_columns', 50)



def get_argument_parser():
    parser = argparse.ArgumentParser()
    # Pretrain dataset directory and files
    parser.add_argument('--data_dir', type=str, default='./Dataset', help='data directory')
    parser.add_argument('--pretrain_files', nargs='+', default=['ai4i2020.csv', 'IoT.csv', 'Steel_industry.csv'], help='pretraining dataset files')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='SAMYANG_dataset', help='dataset name')
    parser.add_argument('--target', type=str, default='SATURATOR_ML_SUPPLY_F_PV.Value', help='label') 
    parser.add_argument('--minute_interval', type=int, default=15, help='dataset collection interval')
    
    # Dataloader
    parser.add_argument('--seq_len', type=int, default=512, help='sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='label length')
    parser.add_argument('--pred_len', type=int, default=6, help='prediction length')
    #parser.add_argument('--step_size', type=int, default=1, help='data slicing step size')
    parser.add_argument('--model_name', type=str, default='Chronos', help='model name')

    # Train
    parser.add_argument('--ft_learning_rate', type=float, default=1e-6, help='learning_rate for fine-tuning')
    parser.add_argument('--pt_learning_rate', type=float, default=5e-6, help='learning_rate for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_steps', type=int, default=400, help='number of fine-tune steps')
    parser.add_argument('--pretrain_steps', type=int, default=400, help='number of pretrain steps')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--device_num', type=str, default='0', help='set gpu number')
    #parser.add_argument('--patience', type=int, default=100, help='train patience')
    #parser.add_argument('--num_workers', type=int, default=0, help='num_workers')

    # Chronos specific
    parser.add_argument('--use_chronos', default = False, action='store_true', help='run Chronos-2 inference instead of TCN training')
    parser.add_argument('--use_covariates',default = False,  action='store_true', help='use covariates for Chronos-2')
    parser.add_argument('--use_cross_learning', default = False, action='store_true', help='use cross learning for Chronos-2')
    parser.add_argument('--cov_corr_threshold', type=float, default=0.5, help='min abs correlation to keep a covariate')
    parser.add_argument('--fine_tune', default = False, action='store_true', help='fine-tune Chronos-2 model')
    parser.add_argument('--continual_pretrain', default = False, action='store_true', help='continual pretrain Chronos-2 model')
    parser.add_argument('--use_das', default = False, action='store_true', help='use DAS for continual pretraining')
    parser.add_argument('--evaluate_naive', default = False, action='store_true', help='use naive method')

    #args = parser.parse_args(args=[])
    args = parser.parse_args()

    return args



# %%
if __name__ == "__main__":

    args = get_argument_parser()
    
    print("\n" + "="*60)
    print(f"{'Argument':<30} | {'Value'}")
    print("-" * 60)
    for arg, value in vars(args).items():
        print(f"{arg:<30} | {value}")
    print("="*60 + "\n")

    data = pd.read_csv('Dataset/{}.csv'.format(args.dataset), parse_dates=['TimeStamp'])
    print(f"Original shape: {data.shape} from Dataset/{args.dataset}.csv")

    cutoff_date = pd.Timestamp('2021-04-30')
    data = data[data['TimeStamp'] > cutoff_date].reset_index(drop= True)

    # Filter by date
    data = data[data['TimeStamp'] > cutoff_date].reset_index(drop=True)
    total_samples = len(data)

    # Count Outliers vs Normal
    n_outliers = int(data['outlier'].sum())
    n_normal = total_samples - n_outliers
    target_column = args.target
    target_idx = data.columns.get_loc(target_column)

    print(f"전체 샘플: {total_samples:,}개 (2021-04-30 이후)")
    print(f"Outlier: {n_outliers:,}개 ({n_outliers/total_samples*100:.2f}%)")
    print(f"Normal: {n_normal:,}개 ({n_normal/total_samples*100:.2f}%)")


    print(f"Target: {target_column} (index: {target_idx})")
    print(f"Date range: {data['TimeStamp'].min()} to {data['TimeStamp'].max()}")

    set_seed(args.seed)

    data_loader = {}
    data_loader['train'], y_scaler = get_dataloader(data, args, shuffle = True, flag = 'train',drop_last = True)
    data_loader['val'] = get_dataloader(data, args, shuffle = True, flag = 'val', drop_last = True)
    data_loader['test'] = get_dataloader(data, args, shuffle = False, flag = 'test', drop_last = False)

    n_train = len(data_loader['train'].dataset)
    n_val = len(data_loader['val'].dataset)
    n_test = len(data_loader['test'].dataset)
    total_seq = n_train + n_val + n_test

    print(f"\nTrain: {n_train:,} 시퀀스 ({n_train/total_seq*100:.2f}%)")
    print(f"Validation: {n_val:,} 시퀀스 ({n_val/total_seq*100:.2f}%)")
    print(f"Test: {n_test:,} 시퀀스 ({n_test/total_seq*100:.2f}%)")
    print(f"총 유효 시퀀스: {total_seq:,}개")


    # If use_chronos flag is set, run Chronos-2 inference on test loader
    forecaster = ChronosForecaster(args)
    
    metrics_df = None
    pred_df = None
    y_true_df = None
    context_df = None

    if args.evaluate_naive:
        print("Running naive method...")
        # Implement naive method logic here
        # For example, you might want to generate predictions using a simple baseline
        # and then calculate metrics accordingly.
        metrics_df_mean, metrics_df_last = forecaster.evaluate_naive(data_loader['test'])
        print("Naive method completed.")

    if args.use_chronos and args.continual_pretrain:
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        # Load manufacturing datasets for continual pretraining
        manufacturing_datasets = load_manufacturing_data(
            args.data_dir,
            args.pretrain_files
        )
        
        # 1. Continual Pretraining
        if args.use_das:
            print("Using DAS for Continual Pretraining...")
        else:
            print("Using Standard Continual Pretraining...")

        forecaster.continual_pretrain(manufacturing_datasets)
        
        # 2. Fine-tuning on target dataset
        forecaster.fine_tune(data_loader)
        
        # 3. Inference
        metrics_df, pred_df, y_true_df, context_df = forecaster.run(data_loader)
        
        print("Chronos-2 inference with continual pretraining + fine-tuning completed.")

    elif args.use_chronos and args.fine_tune and not args.continual_pretrain:
        forecaster.fine_tune(data_loader)
        print("Chronos-2 fine-tuning completed.")
        metrics_df, pred_df, y_true_df, context_df = forecaster.run(data_loader)
    
    elif args.use_chronos:
        metrics_df, pred_df, y_true_df, context_df = forecaster.run(data_loader)
        print("Chronos-2 zero-shot inference completed.")


    # Determine experiment ID based on args
    exp_id = "unknown"
    if args.use_chronos:
        if args.continual_pretrain:
            if args.use_das:
                if args.use_cross_learning:
                    exp_id = "8. DAS Continual Pretrained + Fine-tuning + Cross Learning"
                else:
                    exp_id = "8. DAS Continual Pretrained + Fine-tuning"
            else:
                if args.use_cross_learning:
                    exp_id = "7. Continual Pretrained Model + Fine-tuning + Cross Learning"
                else:
                    exp_id = "6. Continual Pretrained Model + Fine-tuning"
        elif args.fine_tune:
            if args.use_cross_learning:
                exp_id = "5. Fine-tuning With Cross Learning"
            else:
                exp_id = "4. Fine-tuning No Cross Learning"
        else:
            if args.use_covariates:
                if args.use_cross_learning:
                    exp_id = "3. With Covariates, With Cross Learning"
                else:
                    exp_id = "2. With Covariates, No Cross Learning"
            else:
                exp_id = "1. No Covariates, No Cross Learning"
    elif args.evaluate_naive:
        exp_id = "0. Naive"
    
    # Plotting logic for MSE quartiles
    if metrics_df is not None and pred_df is not None and y_true_df is not None:
        print("\nGenerating plots for MSE quartiles (0%, 25%, 50%, 75%, 100%)...")
        
        # Keep a list of quantiles for plotting, but export ALL test ids to CSV.
        quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        # All test-set sequence ids sorted by MSE (for CSV export)
        sorted_metrics = metrics_df.sort_values(by='mse').reset_index(drop=True)
        selected_ids_all = []
        n = len(sorted_metrics)
        if n > 0:
            for idx, row in sorted_metrics.iterrows():
                q = idx / (n - 1) if n > 1 else 0.0
                selected_ids_all.append((q, row['id'], row['mse']))

        # Quantile-based selection for plotting only
        selected_ids_plot = []
        if n > 0:
            for q in quantiles:
                idx = int((n - 1) * q)
                row = sorted_metrics.iloc[idx]
                selected_ids_plot.append((q, row['id'], row['mse']))

        # Export predictions for ALL test sequences to CSV
        print("Exporting all test-sequence predictions to CSV...")
        export_list = []
        for q, sid, mse in selected_ids_all:
            # Get predictions and true values for this ID
            sub_pred = pred_df[pred_df['id'] == sid].copy()
            sub_true = y_true_df[y_true_df['id'] == sid].copy()
            
            # Merge on id and TimeStamp
            merged_sub = pd.merge(sub_true, sub_pred, on=['id', 'TimeStamp'], how='inner')
            
            # Rename quantile columns for clarity if they exist
            rename_map = {}
            if '0.1' in merged_sub.columns: rename_map['0.1'] = 'pred_lower_0.1'
            if '0.9' in merged_sub.columns: rename_map['0.9'] = 'pred_upper_0.9'
            if '0.5' in merged_sub.columns: rename_map['0.5'] = 'pred_median_0.5'
            
            if rename_map:
                merged_sub.rename(columns=rename_map, inplace=True)
            
            # Add info about which quantile of MSE this represents
            merged_sub['mse_quantile_rank'] = q
            merged_sub['mse_value'] = mse
            
            # Add context data if available
            if context_df is not None:
                sub_context = context_df[context_df['id'] == sid].copy()
                if not sub_context.empty:
                    # Keep only essential columns from context (TimeStamp, target, id)
                    # This ensures other columns (like covariates) are blank/NaN in the context rows
                    keep_cols = ['id', 'TimeStamp', args.target]
                    sub_context = sub_context[[c for c in keep_cols if c in sub_context.columns]]
                    
                    # Add missing columns from merged_sub with NaN
                    for col in merged_sub.columns:
                        if col not in sub_context.columns:
                            sub_context[col] = np.nan
                            
                    # Ensure mse info is present in context rows too
                    sub_context['mse_quantile_rank'] = q
                    sub_context['mse_value'] = mse
                    
                    # Reorder to match merged_sub
                    sub_context = sub_context[merged_sub.columns]
                    
                    # Concatenate context and prediction
                    merged_sub = pd.concat([sub_context, merged_sub], ignore_index=True)
                    merged_sub.sort_values('TimeStamp', inplace=True)
            
            export_list.append(merged_sub)
            
        if export_list:
            export_df = pd.concat(export_list, ignore_index=True)
            safe_exp_id = exp_id.replace(" ", "_").replace(".", "").replace(",", "").replace("+", "plus")
            csv_path = f'result/EX{safe_exp_id}_quantile_predictions.csv'
            export_df.to_csv(csv_path, index=False)
            print(f"Saved quantile predictions to {csv_path}")

        fig, axes = plt.subplots(5, 1, figsize=(15, 25))
        if len(quantiles) == 1:
            axes = [axes] # Handle single plot case if needed
        
        # Add experiment title to the figure
        fig.suptitle(f"Experiment: {exp_id} with mean MSE = {metrics_df['mse'].mean():.2f}", fontsize=16, y=0.99)

        for (q, sid, mse), ax in zip(selected_ids_plot, axes):
            plot_forecast(
                context_df=pd.DataFrame(), # Context not used in current util.py implementation
                pred_df=pred_df,
                test_df=y_true_df,
                target_column=args.target,
                timeseries_id=sid,
                prediction_length=args.pred_len,
                ax=ax,
                title_suffix=f"(MSE Quantile: {int(q*100)}%, MSE: {mse:.4f})"
            )
        
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make room for suptitle
        
        # Create a safe filename from exp_id
        safe_exp_id = exp_id.replace(" ", "_").replace(".", "").replace(",", "").replace("+", "plus")
        filename = f'result/EX{safe_exp_id}.png'
        plt.savefig(filename)
        print(f"Saved plot to {filename}")

        # Save results to log file
        log_path = "result/result_log.txt"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        with open(log_path, "a") as f:
            f.write(f"\n{'='*20} Experiment Log: {pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=9)} (UTC+9) {'='*20}\n")
            f.write(f"Experiment ID: {exp_id}\n")
            f.write(f"Settings:\n")
            for arg, value in vars(args).items():
                f.write(f"  {arg}: {value}\n")
            
            if metrics_df is not None:
                mean_metrics = metrics_df[['mse', 'rmse', 'mae']].mean()
                f.write(f"\nMetrics (Average):\n")
                f.write(f"  MSE: {mean_metrics['mse']:.4f}\n")
                f.write(f"  RMSE: {mean_metrics['rmse']:.4f}\n")
                f.write(f"  MAE: {mean_metrics['mae']:.4f}\n")
            else:
                f.write("\nMetrics: None (Inference not run or failed)\n")
            
            #naive_metrics_df = None
            #naive_metrics_df = forecaster.evaluate_naive(data_loader['test'])


            #if naive_metrics_df is not None:
            #    mean_naive = naive_metrics_df[['mse', 'rmse', 'mae']].mean()
            #    f.write(f"\nNaive Forecast Metrics (Average):\n")
            #    f.write(f"  MSE: {mean_naive['mse']:.4f}\n")
            #    f.write(f"  RMSE: {mean_naive['rmse']:.4f}\n")
            #    f.write(f"  MAE: {mean_naive['mae']:.4f}\n")

            f.write(f"{'='*60}\n")
        
        print(f"Appended experiment results to {log_path}")
