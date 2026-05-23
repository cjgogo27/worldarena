#!/usr/bin/env python3
"""
基于LSTM神经网络的行程时间预测系统
功能：使用深度学习模型预测OD对的行程时间（avg_time）
支持：多天数据、训练/验证/测试集划分、CSV格式输出（包含真实值对比）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，请运行: pip install torch")

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TravelTimeDataset(Dataset):
    """行程时间数据集（用于PyTorch DataLoader）"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMTravelTimePredictor(nn.Module):
    """
    LSTM-based Travel Time Predictor
    
    架构:
        - Input: (batch, sequence_length, num_od_pairs)
        - LSTM layers: 多层LSTM提取时间模式
        - Dropout: 防止过拟合
        - FC layers: 全连接层输出预测
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, 
                 output_size=1, dropout=0.2):
        super(LSTMTravelTimePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out


class LSTMTravelTimePrediction:
    """LSTM 行程时间预测器"""
    
    def __init__(self, data_path, output_dir, device=None):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df_temporal = None
        self.od_time_3d = None  # (days, time_slots, n_zones, n_zones)
        self.od_flow_3d = None  # 用于辅助特征
        
        # 自动选择设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 模型参数
        self.sequence_length = 12  # 使用前12个时间槽（3小时）预测
        self.prediction_horizon = 1  # 预测下1个时间槽（15分钟）
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # 日期划分
        self.train_dates = None
        self.val_dates = None
        self.test_dates = None
        
    def load_data(self, train_dates=None, val_dates=None, test_dates=None):
        """
        加载多天行程时间数据
        
        Args:
            train_dates: 训练集日期列表
            val_dates: 验证集日期列表
            test_dates: 测试集日期列表
        """
        print("="*80)
        print("加载多天时序数据")
        print("="*80)
        
        # 加载CSV数据
        print(f"\n正在加载: {self.data_path}")
        self.df_temporal = pd.read_csv(self.data_path)
        
        print(f"  总记录数: {len(self.df_temporal):,}")
        print(f"  日期范围: {self.df_temporal['date'].min()} 到 {self.df_temporal['date'].max()}")
        
        # 获取所有唯一日期
        all_dates = sorted(self.df_temporal['date'].unique())
        print(f"  总天数: {len(all_dates)}")
        
        # 设置日期划分
        if train_dates is None:
            train_dates = ['2008-02-02', '2008-02-03', '2008-02-04', '2008-02-05', '2008-02-06']
            val_dates = ['2008-02-07']
            test_dates = ['2008-02-08']
        
        self.train_dates = train_dates
        self.val_dates = val_dates
        self.test_dates = test_dates
        
        print(f"\n数据集划分:")
        print(f"  训练集: {train_dates}")
        print(f"  验证集: {val_dates}")
        print(f"  测试集: {test_dates}")
        
        # 构建3D 行程时间矩阵: (days, time_slots, n_zones, n_zones)
        n_zones = self.df_temporal['origin'].max()
        n_time_slots = 96
        n_days = len(all_dates)
        
        self.od_time_3d = np.zeros((n_days, n_time_slots, n_zones, n_zones))
        self.od_flow_3d = np.zeros((n_days, n_time_slots, n_zones, n_zones))
        
        # 填充数据
        for day_idx, date in enumerate(all_dates):
            df_day = self.df_temporal[self.df_temporal['date'] == date]
            
            for _, row in df_day.iterrows():
                t = int(row['time_slot'])
                o = int(row['origin']) - 1  # 转换为0-based
                d = int(row['dest']) - 1
                
                self.od_time_3d[day_idx, t, o, d] = row['avg_time']
                self.od_flow_3d[day_idx, t, o, d] = row['flow']
        
        print(f"\n行程时间矩阵形状: {self.od_time_3d.shape}")
        print(f"  天数: {self.od_time_3d.shape[0]}")
        print(f"  时间槽数/天: {self.od_time_3d.shape[1]}")
        print(f"  区域数: {self.od_time_3d.shape[2]} × {self.od_time_3d.shape[3]}")
        
        # 统计每天的平均行程时间
        print(f"\n每天平均行程时间:")
        for day_idx, date in enumerate(all_dates):
            mask = self.od_time_3d[day_idx] > 0
            if mask.sum() > 0:
                avg_time = self.od_time_3d[day_idx][mask].mean()
                print(f"  {date}: {avg_time:.2f} 分钟")
        
        return self.od_time_3d.shape
        
    def prepare_sequences(self, date_list, active_indices=None, fit_scaler=False):
        """
        准备LSTM训练序列
        
        Args:
            date_list: 要使用的日期列表
            active_indices: 活跃OD对的索引
            fit_scaler: 是否重新拟合scaler
            
        Returns:
            X: 输入序列 (N, sequence_length, n_od_pairs)
            y: 目标值 (N, n_od_pairs)
            od_pairs: OD对列表
            active_indices: 活跃OD对的索引
            date_info: 每个样本的日期和时间槽信息
        """
        all_dates = sorted(self.df_temporal['date'].unique())
        n_zones = self.od_time_3d.shape[2]
        
        # 获取指定日期的索引
        day_indices = [all_dates.index(date) for date in date_list]
        
        print(f"\n准备序列数据（{len(date_list)}天）")
        
        # 将多天数据连接成一个长序列
        od_time_list = []
        for day_idx in day_indices:
            od_time_list.append(self.od_time_3d[day_idx])
        
        od_time_concat = np.concatenate(od_time_list, axis=0)
        T_total = od_time_concat.shape[0]
        
        print(f"  合并后总时间槽数: {T_total} ({len(date_list)}天 × 96槽)")
        
        # 展平为 (T, n_od_pairs)
        od_flat = od_time_concat.reshape(T_total, -1)
        
        # 如果没有提供active_indices，则基于当前数据计算
        if active_indices is None:
            # 选择至少有一次非零行程时间的OD对
            has_data = (od_flat > 0).sum(axis=0)
            active_indices = np.where(has_data > 0)[0]
            print(f"  活跃OD对数: {len(active_indices)} / {n_zones*n_zones}")
        else:
            print(f"  使用预定义的活跃OD对数: {len(active_indices)} / {n_zones*n_zones}")
        
        # 只使用活跃的OD对
        od_data = od_flat[:, active_indices]
        
        # 处理零值：对于没有数据的地方，用全局平均值填充（归一化前）
        # 这样LSTM可以学习到"无数据"和"有数据"的模式
        global_mean = od_data[od_data > 0].mean() if (od_data > 0).sum() > 0 else 0
        od_data_processed = od_data.copy()
        # 保持0为0，这样模型可以学习到OD对的活跃模式
        
        # 归一化
        if fit_scaler:
            # 只基于非零值拟合scaler
            non_zero_mask = od_data_processed > 0
            if non_zero_mask.sum() > 0:
                od_data_scaled = np.zeros_like(od_data_processed)
                temp_data = od_data_processed[non_zero_mask].reshape(-1, 1)
                self.scaler.fit(temp_data)
                od_data_scaled[non_zero_mask] = self.scaler.transform(temp_data).flatten()
            else:
                od_data_scaled = od_data_processed
        else:
            od_data_scaled = np.zeros_like(od_data_processed)
            non_zero_mask = od_data_processed > 0
            if non_zero_mask.sum() > 0:
                temp_data = od_data_processed[non_zero_mask].reshape(-1, 1)
                od_data_scaled[non_zero_mask] = self.scaler.transform(temp_data).flatten()
        
        # 创建滑动窗口序列
        X, y = [], []
        date_info = []
        
        for i in range(len(od_data_scaled) - self.sequence_length - self.prediction_horizon + 1):
            # 输入：过去sequence_length个时间槽的所有OD行程时间
            seq = od_data_scaled[i:i + self.sequence_length]
            
            # 目标：未来prediction_horizon个时间槽的所有OD行程时间
            target = od_data_scaled[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
            
            X.append(seq)
            y.append(target.flatten())
            
            # 记录日期和时间槽
            target_slot = i + self.sequence_length
            day_idx = target_slot // 96
            time_slot = target_slot % 96
            if day_idx < len(day_indices):
                date = all_dates[day_indices[day_idx]]
                date_info.append({
                    'date': date, 
                    'time_slot': time_slot, 
                    'global_slot': target_slot,
                    'day_idx': day_indices[day_idx]
                })
        
        X = np.array(X)
        y = np.array(y)
        
        # 转换active_indices回OD对
        od_pairs = []
        for idx in active_indices:
            o = idx // n_zones + 1  # 1-based
            d = idx % n_zones + 1
            od_pairs.append((o, d))
        
        print(f"  序列数: {len(X)}")
        print(f"  序列形状: X={X.shape}, y={y.shape}")
        
        return X, y, od_pairs, active_indices, date_info
    
    def train_model(self, hidden_size=128, num_layers=3, 
                   epochs=100, batch_size=32, learning_rate=0.001):
        """
        训练LSTM模型预测行程时间
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，无法训练模型")
        
        print("\n" + "="*80)
        print("训练LSTM模型（预测行程时间）")
        print("="*80)
        
        # 确定全局活跃OD对
        print("\n确定全局活跃OD对...")
        all_dates = sorted(self.df_temporal['date'].unique())
        all_od_data = []
        for day_idx in range(len(all_dates)):
            all_od_data.append(self.od_time_3d[day_idx])
        all_od_matrix = np.concatenate(all_od_data, axis=0)
        all_od_flat = all_od_matrix.reshape(all_od_matrix.shape[0], -1)
        has_data = (all_od_flat > 0).sum(axis=0)
        active_indices = np.where(has_data > 0)[0]
        print(f"全局活跃OD对数: {len(active_indices)} / {all_od_flat.shape[1]}")
        
        # 准备训练集数据
        X_train, y_train, od_pairs, active_indices, train_date_info = self.prepare_sequences(
            self.train_dates, active_indices=active_indices, fit_scaler=True)
        
        # 准备验证集数据
        X_val, y_val, _, _, val_date_info = self.prepare_sequences(
            self.val_dates, active_indices=active_indices, fit_scaler=False)
        
        # 准备测试集数据（包含验证集最后一天提供历史数据）
        test_dates_with_history = list(self.val_dates[-1:]) + list(self.test_dates)
        X_test_all, y_test_all, _, _, test_date_info_all = self.prepare_sequences(
            test_dates_with_history, active_indices=active_indices, fit_scaler=False)
        
        # 只保留属于测试日期的样本
        test_indices = [i for i, info in enumerate(test_date_info_all) 
                       if info['date'] in self.test_dates]
        X_test = X_test_all[test_indices]
        y_test = y_test_all[test_indices]
        test_date_info = [test_date_info_all[i] for i in test_indices]
        
        print(f"\n数据集大小:")
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  验证集: {len(X_val)} 样本")
        print(f"  测试集: {len(X_test)} 样本")
        
        # 创建数据加载器
        train_dataset = TravelTimeDataset(X_train, y_train)
        val_dataset = TravelTimeDataset(X_val, y_val)
        test_dataset = TravelTimeDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        input_size = X_train.shape[2]  # OD对数
        output_size = y_train.shape[1]  # OD对数
        
        model = LSTMTravelTimePredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=0.3
        ).to(self.device)
        
        print(f"\n模型架构:")
        print(f"  输入维度: {input_size} (活跃OD对数)")
        print(f"  隐藏层大小: {hidden_size}")
        print(f"  LSTM层数: {num_layers}")
        print(f"  输出维度: {output_size}")
        print(f"  总参数: {sum(p.numel() for p in model.parameters()):,}")
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # 训练
        history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 20
        
        print(f"\n开始训练 (epochs={epochs}, batch_size={batch_size}, lr={learning_rate})...")
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_losses = []
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # 验证阶段
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_losses.append(loss.item())
            
            # 测试阶段
            test_losses = []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    test_losses.append(loss.item())
            
            # 记录
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            test_loss = np.mean(test_losses)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['test_loss'].append(test_loss)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, Test: {test_loss:.6f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'{self.output_dir}/best_lstm_traveltime_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print(f"\n训练完成!")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        
        # 加载最佳模型
        model.load_state_dict(torch.load(f'{self.output_dir}/best_lstm_traveltime_model.pth'))
        
        return model, history, X_test, y_test, od_pairs, active_indices, test_date_info
    
    def evaluate_and_save(self, model, X_test, y_test, od_pairs, test_date_info):
        """评估模型并保存预测结果（包含真实值）"""
        print("\n" + "="*80)
        print("模型评估与结果保存")
        print("="*80)
        
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            predictions_scaled = model(X_test_tensor).cpu().numpy()
        
        # 反归一化（只对非零值）
        y_test_original = np.zeros_like(y_test)
        predictions_original = np.zeros_like(predictions_scaled)
        
        for i in range(len(y_test)):
            for j in range(y_test.shape[1]):
                if y_test[i, j] > 0:
                    y_test_original[i, j] = self.scaler.inverse_transform([[y_test[i, j]]])[0, 0]
                if predictions_scaled[i, j] > 0:
                    predictions_original[i, j] = self.scaler.inverse_transform([[predictions_scaled[i, j]]])[0, 0]
        
        # 计算指标（只针对非零真实值）
        mask = y_test_original.flatten() > 0
        if mask.sum() > 0:
            mae = mean_absolute_error(y_test_original.flatten()[mask], 
                                     predictions_original.flatten()[mask])
            rmse = np.sqrt(mean_squared_error(y_test_original.flatten()[mask], 
                                             predictions_original.flatten()[mask]))
            r2 = r2_score(y_test_original.flatten()[mask], 
                         predictions_original.flatten()[mask])
            
            # MAPE
            mape = np.mean(np.abs((y_test_original.flatten()[mask] - predictions_original.flatten()[mask]) 
                                  / y_test_original.flatten()[mask])) * 100
        else:
            mae = rmse = r2 = mape = 0
        
        print(f"\n性能指标（测试集）:")
        print(f"  MAE:  {mae:.4f} 分钟")
        print(f"  RMSE: {rmse:.4f} 分钟")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}")
        
        # 保存为CSV（包含真实值和预测值）
        records = []
        n_zones = self.od_time_3d.shape[2]
        
        for i, pred_values in enumerate(predictions_original):
            info = test_date_info[i]
            date = info['date']
            time_slot = info['time_slot']
            day_idx = info['day_idx']
            
            # 计算小时和分钟
            hour = (time_slot * 15) // 60
            minute = (time_slot * 15) % 60
            
            for j, (o, d) in enumerate(od_pairs):
                true_time = y_test_original[i, j]
                pred_time = pred_values[j]
                
                # 保存所有有真实值或预测值的记录
                if true_time > 0.01 or pred_time > 0.01:
                    # 获取流量（从原始数据）
                    flow = self.od_flow_3d[day_idx, time_slot, o-1, d-1]
                    
                    records.append({
                        'date': date,
                        'time_slot': time_slot,
                        'hour': hour,
                        'minute': minute,
                        'origin': o,
                        'dest': d,
                        'flow': round(flow, 6),
                        'true_avg_time': round(true_time, 6),
                        'predicted_avg_time': round(pred_time, 6),
                        'absolute_error': round(abs(true_time - pred_time), 6),
                        'relative_error': round(abs(true_time - pred_time) / true_time * 100, 2) if true_time > 0 else 0
                    })
        
        # 转换为DataFrame
        df = pd.DataFrame(records)
        
        # 保存
        output_file = f'{self.output_dir}/travel_time_predictions_lstm.csv'
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ 预测结果已保存: {output_file}")
        print(f"  总记录数: {len(df):,}")
        print(f"  包含字段: 真实值(true_avg_time) + 预测值(predicted_avg_time) + 误差")
        
        # 保存评估指标
        metrics_df = pd.DataFrame([{
            'Model': 'LSTM',
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Train_Dates': ', '.join(self.train_dates),
            'Val_Dates': ', '.join(self.val_dates),
            'Test_Dates': ', '.join(self.test_dates)
        }])
        metrics_df.to_csv(f'{self.output_dir}/lstm_traveltime_evaluation.csv', index=False)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }, df
    
    def visualize_results(self, history, predictions_df):
        """可视化训练过程和预测结果"""
        fig = plt.figure(figsize=(18, 12))
        
        # 1. 训练历史
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax1.plot(history['test_loss'], label='Test Loss', linestyle='--', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 真实值 vs 预测值
        ax2 = plt.subplot(2, 3, 2)
        sample = predictions_df.sample(min(5000, len(predictions_df)))
        ax2.scatter(sample['true_avg_time'], sample['predicted_avg_time'], 
                   alpha=0.3, s=10)
        max_val = max(sample['true_avg_time'].max(), sample['predicted_avg_time'].max())
        ax2.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax2.set_xlabel('True Travel Time (min)')
        ax2.set_ylabel('Predicted Travel Time (min)')
        ax2.set_title('True vs Predicted')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 误差分布
        ax3 = plt.subplot(2, 3, 3)
        errors = predictions_df['absolute_error']
        ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(x=errors.mean(), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {errors.mean():.2f}')
        ax3.set_xlabel('Absolute Error (min)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 按时段的误差
        ax4 = plt.subplot(2, 3, 4)
        hourly_error = predictions_df.groupby('hour')['absolute_error'].mean()
        ax4.plot(hourly_error.index, hourly_error.values, marker='o', linewidth=2)
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Mean Absolute Error (min)')
        ax4.set_title('Error by Hour')
        ax4.grid(True, alpha=0.3)
        
        # 5. 按OD对的误差（Top 10最大误差）
        ax5 = plt.subplot(2, 3, 5)
        od_error = predictions_df.groupby(['origin', 'dest'])['absolute_error'].mean()
        top_errors = od_error.nlargest(10).sort_values()
        od_labels = [f"{o}→{d}" for (o, d) in top_errors.index]
        ax5.barh(range(len(top_errors)), top_errors.values)
        ax5.set_yticks(range(len(top_errors)))
        ax5.set_yticklabels(od_labels, fontsize=9)
        ax5.set_xlabel('Mean Absolute Error (min)')
        ax5.set_title('Top 10 OD Pairs by Error')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # 6. 相对误差分布
        ax6 = plt.subplot(2, 3, 6)
        rel_errors = predictions_df[predictions_df['true_avg_time'] > 0]['relative_error']
        ax6.hist(rel_errors.clip(upper=200), bins=50, edgecolor='black', alpha=0.7)
        ax6.set_xlabel('Relative Error (%)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Relative Error Distribution')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/lstm_traveltime_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 可视化结果已保存: lstm_traveltime_results.png")


def main():
    """主函数"""
    print("="*80)
    print("基于LSTM的行程时间预测系统")
    print("="*80)
    
    if not TORCH_AVAILABLE:
        print("\n错误: PyTorch未安装!")
        print("请运行: pip install torch")
        return
    
    # 配置路径
    data_path = '/data/alice/cjtest/TRC/haidian_od_analysis/output/od_flow_temporal.csv'
    output_dir = '/data/alice/cjtest/TRC/Travel_Time'
    
    predictor = LSTMTravelTimePrediction(data_path, output_dir)
    
    # 1. 加载数据
    train_dates = ['2008-02-02', '2008-02-03', '2008-02-04', '2008-02-05', '2008-02-06']
    val_dates = ['2008-02-07']
    test_dates = ['2008-02-08']
    
    predictor.load_data(train_dates=train_dates, val_dates=val_dates, test_dates=test_dates)
    
    # 2. 训练模型
    model, history, X_test, y_test, od_pairs, active_indices, test_date_info = predictor.train_model(
        hidden_size=128,
        num_layers=3,
        epochs=100,
        batch_size=32,
        learning_rate=0.001
    )
    
    # 3. 评估并保存结果
    metrics, predictions_df = predictor.evaluate_and_save(
        model, X_test, y_test, od_pairs, test_date_info
    )
    
    # 4. 可视化
    predictor.visualize_results(history, predictions_df)
    
    # 5. 显示结果摘要
    print("\n" + "="*80)
    print("预测完成！")
    print("="*80)
    print(f"\n性能指标:")
    print(f"  MAE:  {metrics['MAE']:.4f} 分钟")
    print(f"  RMSE: {metrics['RMSE']:.4f} 分钟")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R²:   {metrics['R2']:.4f}")
    
    print(f"\n输出文件:")
    print(f"  1. travel_time_predictions_lstm.csv - 预测结果（含真实值对比）")
    print(f"  2. lstm_traveltime_evaluation.csv - 评估指标")
    print(f"  3. lstm_traveltime_results.png - 可视化图表")
    print(f"  4. best_lstm_traveltime_model.pth - 最佳模型权重")
    
    print(f"\n预测结果示例:")
    print(predictions_df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
