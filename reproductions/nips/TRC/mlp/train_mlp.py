#!/usr/bin/env python3
"""
MLP（多层感知机）行程时间预测
=================================

功能：
1. 读取插值后的数据
2. 创建特征（不泄露测试集信息）
3. 训练 MLP 模型，学习 flow→travel time 的关系
4. 评估模型（包括准确率指标）

数据集划分：
  训练集: 2008-02-02 ~ 2008-02-06
  验证集: 2008-02-07
  测试集: 2008-02-08
"""

import sys
sys.stdout.reconfigure(line_buffering=True)   # 实时输出，无需 python -u
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# MLP 模型定义
# ============================================================

class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ============================================================
# 预测器
# ============================================================

class MLPTravelTimePredictor:
    """MLP 行程时间预测器（无数据泄露）"""

    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[设备] 使用: {self.device}")

    # ----------------------------------------------------------
    # 1. 加载数据
    # ----------------------------------------------------------

    def load_interpolated_data(self):
        print("=" * 80)
        print("1. 加载插值后的数据")
        print("=" * 80)

        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])

        print(f"数据量: {len(self.df):,} 条记录")
        print(f"日期范围: {self.df['date'].min()} ~ {self.df['date'].max()}")
        print(f"插值记录: {self.df['is_interpolated'].sum():,} "
              f"({self.df['is_interpolated'].sum()/len(self.df)*100:.1f}%)")
        print(f"原始记录: {(~self.df['is_interpolated']).sum():,}")

    # ----------------------------------------------------------
    # 2. 特征工程（与 XGBoost 版本完全相同）
    # ----------------------------------------------------------

    def create_features_no_leak(self, train_dates, val_dates, test_dates):
        print("\n" + "=" * 80)
        print("2. 创建特征（无数据泄露）")
        print("=" * 80)

        df = self.df.copy()

        train_df = df[df['date'].isin(pd.to_datetime(train_dates))].copy()
        val_df   = df[df['date'].isin(pd.to_datetime(val_dates))].copy()
        test_df  = df[df['date'].isin(pd.to_datetime(test_dates))].copy()

        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_df):,} 条 ({train_dates})")
        print(f"  验证集: {len(val_df):,} 条 ({val_dates})")
        print(f"  测试集: {len(test_df):,} 条 ({test_dates})")

        # 只用训练集计算统计值
        print(f"\n计算历史统计（仅使用训练集）...")
        train_global_mean = train_df['avg_time'].mean()
        train_global_std  = train_df['avg_time'].std()

        hour_stats = (train_df.groupby('hour')['avg_time']
                      .agg(['mean', 'std'])
                      .reset_index())
        hour_stats.columns = ['hour', 'hour_mean', 'hour_std']
        hour_stats['hour_std'].fillna(0, inplace=True)

        train_df['flow_bin'] = pd.cut(
            train_df['flow'], bins=[0, 1, 3, 5, 10, 100],
            labels=['0-1', '1-3', '3-5', '5-10', '10+'])
        flow_stats = (train_df.groupby('flow_bin')['avg_time']
                      .agg(['mean', 'std'])
                      .reset_index())
        flow_stats.columns = ['flow_bin', 'flow_bin_mean', 'flow_bin_std']

        print(f"  训练集全局均值: {train_global_mean:.2f} 分钟")
        print(f"  训练集全局标准差: {train_global_std:.2f} 分钟")

        for name, dataset in [('train', train_df), ('val', val_df), ('test', test_df)]:
            print(f"\n创建{name}集特征...")

            # 时间特征
            dataset['hour_norm']      = dataset['hour'] / 23.0
            dataset['minute_norm']    = dataset['minute'] / 59.0
            dataset['time_slot_norm'] = dataset['time_slot'] / 95.0
            dataset['day_of_week']    = dataset['date'].dt.dayofweek / 6.0
            dataset['is_weekend']     = (dataset['date'].dt.dayofweek >= 5).astype(int)
            dataset['is_morning_peak'] = ((dataset['hour'] >= 7) & (dataset['hour'] <= 9)).astype(int)
            dataset['is_evening_peak'] = ((dataset['hour'] >= 17) & (dataset['hour'] <= 19)).astype(int)
            dataset['is_peak']        = (dataset['is_morning_peak'] | dataset['is_evening_peak']).astype(int)

            # Flow 特征
            dataset['flow_raw']     = dataset['flow']
            dataset['flow_log']     = np.log1p(dataset['flow'])
            dataset['flow_sqrt']    = np.sqrt(dataset['flow'])
            dataset['flow_squared'] = dataset['flow'] ** 2
            dataset['flow_density_morning'] = dataset['flow'] * dataset['is_morning_peak']
            dataset['flow_density_evening'] = dataset['flow'] * dataset['is_evening_peak']

            # 历史统计特征（训练集计算）
            dataset = dataset.merge(hour_stats, on='hour', how='left')
            dataset['hour_mean'].fillna(train_global_mean, inplace=True)
            dataset['hour_std'].fillna(train_global_std, inplace=True)

            dataset['flow_bin'] = pd.cut(dataset['flow'], bins=[0, 1, 3, 5, 10, 100],
                                          labels=['0-1', '1-3', '3-5', '5-10', '10+'])
            dataset = dataset.merge(flow_stats, on='flow_bin', how='left')
            dataset['flow_bin_mean'].fillna(train_global_mean, inplace=True)
            dataset['flow_bin_std'].fillna(0, inplace=True)

            # 时序特征（shift/rolling，仅使用过去数据）
            dataset = dataset.sort_values(
                ['origin', 'dest', 'date', 'time_slot']).reset_index(drop=True)
            for lag in [1, 2, 4, 8, 16]:
                dataset[f'lag_{lag}'] = (dataset
                    .groupby(['origin', 'dest'])['avg_time'].shift(lag))
            dataset['rolling_4h_mean'] = (dataset
                .groupby(['origin', 'dest'])['avg_time']
                .rolling(window=16, min_periods=1).mean().values)
            dataset['rolling_4h_std'] = (dataset
                .groupby(['origin', 'dest'])['avg_time']
                .rolling(window=16, min_periods=1).std().values)

            # 填充缺失值（仅数值列）
            for col in dataset.columns:
                if dataset[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    if 'lag_' in col or 'rolling_' in col:
                        dataset[col].fillna(train_global_mean, inplace=True)
                    else:
                        dataset[col].fillna(0, inplace=True)

            if name == 'train':
                train_df = dataset
            elif name == 'val':
                val_df = dataset
            else:
                test_df = dataset

        return train_df, val_df, test_df, train_global_mean

    # ----------------------------------------------------------
    # 3. 训练 MLP
    # ----------------------------------------------------------

    def train_model(self, X_train, y_train, X_val, y_val,
                    hidden_dims=(256, 128, 64), dropout=0.2,
                    epochs=100, batch_size=2048, lr=1e-3, patience=10):
        print("\n" + "=" * 80)
        print("3. 训练 MLP 模型")
        print("=" * 80)

        # 标准化（在训练集上 fit，同样变换验证/测试集）
        X_train_sc = self.scaler.fit_transform(X_train)
        X_val_sc   = self.scaler.transform(X_val)

        # 转为 Tensor
        def to_tensor(X, y=None):
            Xt = torch.FloatTensor(X).to(self.device)
            if y is not None:
                yt = torch.FloatTensor(y.values).to(self.device)
                return Xt, yt
            return Xt

        Xt, yt = to_tensor(X_train_sc, y_train)
        Xv, yv = to_tensor(X_val_sc,   y_val)

        loader = DataLoader(TensorDataset(Xt, yt),
                            batch_size=batch_size, shuffle=True)

        # 模型、优化器、损失
        model = MLP(X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-5)
        criterion = nn.HuberLoss(delta=10.0)   # 对异常值更鲁棒

        print(f"\nMLP 结构: {X_train.shape[1]} → "
              f"{' → '.join(str(h) for h in hidden_dims)} → 1")
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"批次大小: {batch_size}, 最大 epochs: {epochs}, 早停耐心: {patience}\n")

        best_val_loss = float('inf')
        best_state    = None
        no_improve    = 0
        train_losses  = []
        val_losses    = []

        for epoch in range(1, epochs + 1):
            # ---- 训练 ----
            model.train()
            epoch_loss = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(Xb)
            train_loss = epoch_loss / len(Xt)

            # ---- 验证 ----
            model.eval()
            with torch.no_grad():
                val_pred = model(Xv)
                val_loss = criterion(val_pred, yv).item()

            scheduler.step(val_loss)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs}  "
                      f"train_loss={train_loss:.4f}  "
                      f"val_loss={val_loss:.4f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")

            if no_improve >= patience:
                print(f"\n  早停触发（连续 {patience} 轮无改善），"
                      f"最佳 val_loss={best_val_loss:.4f}")
                break

        # 恢复最佳参数
        model.load_state_dict(best_state)
        model.eval()
        print(f"\n✓ 训练完成，共 {epoch} 轮")

        self._model       = model
        self._train_losses = train_losses
        self._val_losses   = val_losses
        return model

    # ----------------------------------------------------------
    # 4. 评估
    # ----------------------------------------------------------

    def evaluate_model(self, X_test, y_test, test_df):
        print("\n" + "=" * 80)
        print("4. 模型评估")
        print("=" * 80)

        X_test_sc = self.scaler.transform(X_test)
        Xt = torch.FloatTensor(X_test_sc).to(self.device)

        self._model.eval()
        with torch.no_grad():
            pred = self._model(Xt).cpu().numpy()

        y_arr = y_test.values

        mae  = mean_absolute_error(y_arr, pred)
        rmse = np.sqrt(mean_squared_error(y_arr, pred))
        r2   = r2_score(y_arr, pred)

        mask = y_arr > 0
        mape = (np.mean(np.abs((y_arr[mask] - pred[mask]) / y_arr[mask])) * 100
                if mask.sum() > 0 else 0.0)

        errors     = np.abs(y_arr - pred)
        acc_5min   = (errors <= 5).sum()  / len(errors) * 100
        acc_10min  = (errors <= 10).sum() / len(errors) * 100
        acc_20min  = (errors <= 20).sum() / len(errors) * 100

        print(f"\nMLP:")
        print(f"  MAE:  {mae:.4f} 分钟")
        print(f"  RMSE: {rmse:.4f} 分钟")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  准确率（误差≤5分钟）:  {acc_5min:.2f}%")
        print(f"  准确率（误差≤10分钟）: {acc_10min:.2f}%")
        print(f"  准确率（误差≤20分钟）: {acc_20min:.2f}%")

        results_df = pd.DataFrame([{
            'Model': 'MLP',
            'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE': mape,
            'Acc@5min': acc_5min, 'Acc@10min': acc_10min, 'Acc@20min': acc_20min
        }])

        test_results = test_df.copy()
        test_results['pred_mlp'] = pred
        test_results['error']    = errors
        test_results['error_pct'] = errors / np.where(y_arr > 0, y_arr, 1) * 100

        return results_df, test_results

    # ----------------------------------------------------------
    # 5. Flow 影响分析
    # ----------------------------------------------------------

    def analyze_flow_impact(self, test_results):
        print("\n" + "=" * 80)
        print("5. Flow 影响分析")
        print("=" * 80)

        test_results['flow_group'] = pd.cut(
            test_results['flow'],
            bins=[-0.1, 0.1, 1, 3, 5, 10, 100],
            labels=['0', '0-1', '1-3', '3-5', '5-10', '10+'])

        flow_analysis = test_results.groupby('flow_group').agg(
            真实平均时间=('avg_time', 'mean'),
            预测平均时间=('pred_mlp', 'mean'),
            平均误差=('error', 'mean'),
            样本数=('flow', 'count')
        ).round(2)

        print("\nFlow 分组分析:")
        print(flow_analysis)

        corr = test_results[['flow', 'avg_time']].corr().iloc[0, 1]
        print(f"\nFlow 与 Travel Time 相关系数: {corr:.4f}")

        return flow_analysis

    # ----------------------------------------------------------
    # 6. 可视化
    # ----------------------------------------------------------

    def visualize_results(self, test_results):
        print("\n" + "=" * 80)
        print("6. 可视化结果")
        print("=" * 80)

        import os
        fig_dir = f'{self.output_dir}/figures'
        os.makedirs(fig_dir, exist_ok=True)

        def save_fig(name):
            path = f'{fig_dir}/{name}'
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ {path}")

        print(f"保存目录: {fig_dir}")

        # (a) 真实 vs 预测
        fig, ax = plt.subplots(figsize=(6, 6))
        sample = test_results.sample(min(5000, len(test_results)), random_state=42)
        ax.scatter(sample['avg_time'], sample['pred_mlp'], alpha=0.3, s=10)
        mv = max(sample['avg_time'].max(), sample['pred_mlp'].max())
        ax.plot([0, mv], [0, mv], 'r--', lw=2, label='y=x')
        ax.set_xlabel('True Travel Time (min)')
        ax.set_ylabel('Predicted Travel Time (min)')
        ax.set_title('True vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_fig('01_true_vs_predicted.png')

        # (b) 误差分布
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(test_results['error'], bins=50, edgecolor='black', alpha=0.7)
        med = test_results['error'].median()
        ax.axvline(med, color='r', linestyle='--', label=f'Median: {med:.2f} min')
        ax.set_xlabel('Absolute Error (min)')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_fig('02_error_distribution.png')

        # (c) Flow vs Travel Time（分组柱状图）
        fig, ax = plt.subplots(figsize=(7, 4))
        fg = test_results.groupby('flow_group')[['avg_time', 'pred_mlp']].mean()
        x = range(len(fg))
        ax.bar([i - 0.2 for i in x], fg['avg_time'],  width=0.4, label='True',      alpha=0.7)
        ax.bar([i + 0.2 for i in x], fg['pred_mlp'],  width=0.4, label='Predicted', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(fg.index, rotation=45)
        ax.set_xlabel('Flow Group')
        ax.set_ylabel('Avg Travel Time (min)')
        ax.set_title('Flow vs Travel Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_fig('03_flow_vs_travel_time.png')

        # (d) 按小时误差
        fig, ax = plt.subplots(figsize=(7, 4))
        hourly_err = test_results.groupby('hour')['error'].mean()
        ax.plot(hourly_err.index, hourly_err.values, marker='o', linewidth=2)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Mean Absolute Error (min)')
        ax.set_title('Error by Hour')
        ax.grid(True, alpha=0.3)
        save_fig('04_error_by_hour.png')

        # (e) 训练/验证 Loss 曲线
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self._train_losses, label='Train Loss', linewidth=2)
        ax.plot(self._val_losses,   label='Val Loss',   linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Huber Loss')
        ax.set_title('Training Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_fig('05_training_curve.png')

        # (f) 累积误差分布（CDF）
        fig, ax = plt.subplots(figsize=(7, 4))
        sorted_err = np.sort(test_results['error'])
        cumulative = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
        ax.plot(sorted_err, cumulative, linewidth=2)
        for thresh, color, lbl in [(5, 'r', '5 min'), (10, 'g', '10 min'), (20, 'b', '20 min')]:
            pct = (test_results['error'] <= thresh).mean() * 100
            ax.axvline(thresh, color=color, linestyle='--', alpha=0.7,
                       label=f'{lbl}: {pct:.1f}%')
        ax.set_xlabel('Absolute Error (min)')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title('Cumulative Error Distribution (CDF)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50)
        save_fig('06_error_cdf.png')

        # (g) Flow 散点（颜色=误差）
        fig, ax = plt.subplots(figsize=(6, 5))
        s = test_results[test_results['flow'] > 0].sample(
            min(3000, len(test_results)), random_state=42)
        sc = ax.scatter(s['flow'], s['avg_time'],
                        c=s['error'], cmap='RdYlGn_r', alpha=0.5, s=20)
        plt.colorbar(sc, ax=ax, label='Error (min)')
        ax.set_xlabel('Flow')
        ax.set_ylabel('Travel Time (min)')
        ax.set_title('Flow vs Travel Time (colored by error)')
        ax.grid(True, alpha=0.3)
        save_fig('07_flow_scatter.png')

        # (h) 高峰/平峰误差
        fig, ax = plt.subplots(figsize=(5, 4))
        tr = test_results
        labels = ['Off-Peak', 'Morning Peak\n(7-9h)', 'Evening Peak\n(17-19h)']
        peak_errors = [
            tr[tr['is_peak'] == 0]['error'].mean(),
            tr[tr['is_morning_peak'] == 1]['error'].mean(),
            tr[tr['is_evening_peak'] == 1]['error'].mean(),
        ]
        bars = ax.bar(labels, peak_errors, alpha=0.7,
                      color=['steelblue', 'tomato', 'orange'])
        for bar, val in zip(bars, peak_errors):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        ax.set_ylabel('Mean Absolute Error (min)')
        ax.set_title('Error by Time Period')
        ax.grid(True, alpha=0.3, axis='y')
        save_fig('08_error_by_period.png')

    # ----------------------------------------------------------
    # 7. 保存结果
    # ----------------------------------------------------------

    def save_results(self, results_df, test_results):
        print("\n" + "=" * 80)
        print("7. 保存结果")
        print("=" * 80)

        eval_path = f'{self.output_dir}/mlp_evaluation.csv'
        pred_path = f'{self.output_dir}/mlp_predictions.csv'

        results_df.to_csv(eval_path, index=False)
        print(f"✓ 评估指标: {eval_path}")

        test_results.to_csv(pred_path, index=False)
        print(f"✓ 预测结果: {pred_path}（含真实值对比）")

        # 保存 PyTorch 模型
        model_path = f'{self.output_dir}/mlp_model.pt'
        torch.save(self._model.state_dict(), model_path)
        print(f"✓ 模型权重: {model_path}")


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 80)
    print("MLP 行程时间预测（无数据泄露）")
    print("=" * 80)
    print(f"执行时间: {datetime.now()}")

    # ---------- 配置 ----------
    DATA_PATH  = '/data/alice/cjtest/TRC/Travel_Time/od_flow_interpolated.csv'
    OUTPUT_DIR = '/data/alice/cjtest/TRC/mlp'

    TRAIN_DATES = ['2008-02-02', '2008-02-03', '2008-02-04', '2008-02-05', '2008-02-06']
    VAL_DATES   = ['2008-02-07']
    TEST_DATES  = ['2008-02-08']

    HIDDEN_DIMS  = (256, 128, 64)   # MLP 隐藏层
    DROPOUT      = 0.2
    EPOCHS       = 100
    BATCH_SIZE   = 2048
    LR           = 1e-3
    PATIENCE     = 10               # 早停耐心

    FEATURE_COLS = [
        'hour_norm', 'minute_norm', 'time_slot_norm',
        'day_of_week', 'is_weekend', 'is_morning_peak', 'is_evening_peak', 'is_peak',
        'flow_raw', 'flow_log', 'flow_sqrt', 'flow_squared',
        'flow_density_morning', 'flow_density_evening',
        'hour_mean', 'hour_std',
        'flow_bin_mean', 'flow_bin_std',
        'lag_1', 'lag_2', 'lag_4', 'lag_8', 'lag_16',
        'rolling_4h_mean', 'rolling_4h_std',
    ]

    # ---------- 流程 ----------
    predictor = MLPTravelTimePredictor(DATA_PATH, OUTPUT_DIR)

    # 1. 读取数据
    predictor.load_interpolated_data()

    # 2. 特征工程
    train_df, val_df, test_df, _ = predictor.create_features_no_leak(
        TRAIN_DATES, VAL_DATES, TEST_DATES)

    print(f"\n特征列表 ({len(FEATURE_COLS)} 个):")
    for i, col in enumerate(FEATURE_COLS, 1):
        print(f"  {i:2d}. {col}")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df['avg_time']
    X_val   = val_df[FEATURE_COLS]
    y_val   = val_df['avg_time']
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df['avg_time']

    print(f"\n数据形状:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    # 3. 训练
    predictor.train_model(
        X_train, y_train, X_val, y_val,
        hidden_dims=HIDDEN_DIMS, dropout=DROPOUT,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE)

    # 4. 评估
    results_df, test_results = predictor.evaluate_model(X_test, y_test, test_df)

    # 5. Flow 分析
    predictor.analyze_flow_impact(test_results)

    # 6. 可视化
    predictor.visualize_results(test_results)

    # 7. 保存
    predictor.save_results(results_df, test_results)

    print("\n" + "=" * 80)
    print("✓ 完成！")
    print("=" * 80)
    print(f"\n关键结果:")
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
