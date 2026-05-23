#!/usr/bin/env python3
"""
基于LSTM神经网络的OD流量预测系统 - 多天时序版本
功能：使用深度学习模型预测未来时间槽的OD流量
支持：多天数据、训练/验证/测试集划分、日期感知、CSV格式输出
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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


class ODFlowDataset(Dataset):
    """OD流量数据集（用于PyTorch DataLoader）"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMODPredictor(nn.Module):
    """
    LSTM-based OD Flow Predictor
    
    架构:
        - Input: (batch, sequence_length, num_od_pairs)
        - LSTM layers: 多层LSTM提取时间模式
        - Dropout: 防止过拟合
        - FC layers: 全连接层输出预测
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, 
                 output_size=1, dropout=0.2):
        super(LSTMODPredictor, self).__init__()
        
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


class LSTMODFlowPredictor:
    """LSTM OD流量预测器 - 支持多天数据"""
    
    def __init__(self, data_dir='output', device=None):
        self.data_dir = data_dir
        self.df_temporal = None
        self.od_flow_3d = None  # (days, time_slots, n_zones, n_zones)
        self.od_time_3d = None
        
        # 自动选择设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 模型参数 - 使用滑动窗口预测
        self.sequence_length = 12  # 使用前12个时间槽（3小时）预测
        self.prediction_horizon = 1  # 预测下1个时间槽（15分钟）
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # 日期划分
        self.train_dates = None
        self.val_dates = None
        self.test_dates = None
        
    def load_data(self, train_dates=None, val_dates=None, test_dates=None):
        """
        加载多天OD流量数据
        
        Args:
            train_dates: 训练集日期列表，例如 ['2008-02-02', '2008-02-03', ...]
            val_dates: 验证集日期列表
            test_dates: 测试集日期列表
        """
        print("="*80)
        print("加载多天时序数据")
        print("="*80)
        
        # 加载CSV数据
        csv_file = f'{self.data_dir}/od_flow_temporal.csv'
        print(f"\n正在加载: {csv_file}")
        self.df_temporal = pd.read_csv(csv_file)
        
        print(f"  总记录数: {len(self.df_temporal):,}")
        print(f"  日期范围: {self.df_temporal['date'].min()} 到 {self.df_temporal['date'].max()}")
        
        # 获取所有唯一日期
        all_dates = sorted(self.df_temporal['date'].unique())
        print(f"  总天数: {len(all_dates)}")
        print(f"  日期列表: {all_dates}")
        
        # 设置日期划分
        if train_dates is None:
            # 默认划分：前5天训练，第6天验证，第7天测试
            train_dates = all_dates[:5]
            val_dates = [all_dates[5]] if len(all_dates) > 5 else []
            test_dates = [all_dates[6]] if len(all_dates) > 6 else []
        
        self.train_dates = train_dates
        self.val_dates = val_dates
        self.test_dates = test_dates
        
        print(f"\n数据集划分:")
        print(f"  训练集: {train_dates}")
        print(f"  验证集: {val_dates}")
        print(f"  测试集: {test_dates}")
        
        # 构建3D OD流量矩阵: (days, time_slots, n_zones, n_zones)
        n_zones = self.df_temporal['origin'].max()
        n_time_slots = 96
        n_days = len(all_dates)
        
        self.od_flow_3d = np.zeros((n_days, n_time_slots, n_zones, n_zones))
        self.od_time_3d = np.zeros((n_days, n_time_slots, n_zones, n_zones))
        
        # 填充数据
        for day_idx, date in enumerate(all_dates):
            df_day = self.df_temporal[self.df_temporal['date'] == date]
            
            for _, row in df_day.iterrows():
                t = int(row['time_slot'])
                o = int(row['origin']) - 1  # 转换为0-based
                d = int(row['dest']) - 1
                
                self.od_flow_3d[day_idx, t, o, d] = row['flow']
                self.od_time_3d[day_idx, t, o, d] = row['avg_time']
        
        print(f"\nOD流量矩阵形状: {self.od_flow_3d.shape}")
        print(f"  天数: {self.od_flow_3d.shape[0]}")
        print(f"  时间槽数/天: {self.od_flow_3d.shape[1]}")
        print(f"  区域数: {self.od_flow_3d.shape[2]} × {self.od_flow_3d.shape[3]}")
        
        # 统计每天的流量
        print(f"\n每天总流量:")
        for day_idx, date in enumerate(all_dates):
            total_flow = self.od_flow_3d[day_idx].sum()
            print(f"  {date}: {total_flow:,.0f} trips")
        
        return self.od_flow_3d.shape
        
    def prepare_sequences(self, date_list, active_indices=None, fit_scaler=False):
        """
        准备LSTM训练序列 - 支持多天数据
        
        Args:
            date_list: 要使用的日期列表
            active_indices: 活跃OD对的索引（如果为None则重新计算）
            fit_scaler: 是否重新拟合scaler（仅训练集需要）
            
        Returns:
            X: 输入序列 (N, sequence_length, n_od_pairs)
            y: 目标值 (N, n_od_pairs)
            od_pairs: OD对列表 [(o, d), ...]
            active_indices: 活跃OD对的索引
            date_info: 每个样本对应的日期和时间槽信息
        """
        all_dates = sorted(self.df_temporal['date'].unique())
        n_zones = self.od_flow_3d.shape[2]
        
        # 获取指定日期的索引
        day_indices = [all_dates.index(date) for date in date_list]
        
        print(f"\n准备序列数据（{len(date_list)}天）")
        
        # 将多天数据连接成一个长序列: (total_time_slots, n_zones, n_zones)
        od_data_list = []
        for day_idx in day_indices:
            od_data_list.append(self.od_flow_3d[day_idx])
        
        od_matrix_concat = np.concatenate(od_data_list, axis=0)
        T_total = od_matrix_concat.shape[0]
        
        print(f"  合并后总时间槽数: {T_total} ({len(date_list)}天 × 96槽)")
        
        # 展平为 (T, n_od_pairs)
        od_flat = od_matrix_concat.reshape(T_total, -1)
        
        # 如果没有提供active_indices，则基于当前数据计算
        if active_indices is None:
            total_flow = od_flat.sum(axis=0)
            active_indices = np.where(total_flow > 0)[0]
            print(f"  活跃OD对数: {len(active_indices)} / {n_zones*n_zones}")
        else:
            print(f"  使用预定义的活跃OD对数: {len(active_indices)} / {n_zones*n_zones}")
        
        # 只使用活跃的OD对
        od_data = od_flat[:, active_indices]
        
        # 归一化
        if fit_scaler:
            od_data_scaled = self.scaler.fit_transform(od_data)
        else:
            od_data_scaled = self.scaler.transform(od_data)
        
        # 创建滑动窗口序列
        X, y = [], []
        date_info = []  # 记录每个样本对应的日期和时间槽
        
        for i in range(len(od_data_scaled) - self.sequence_length - self.prediction_horizon + 1):
            # 输入：过去sequence_length个时间槽的所有OD流量
            seq = od_data_scaled[i:i + self.sequence_length]
            
            # 目标：未来prediction_horizon个时间槽的所有OD流量
            target = od_data_scaled[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
            
            X.append(seq)
            y.append(target.flatten())
            
            # 记录日期和时间槽
            target_slot = i + self.sequence_length
            day_idx = target_slot // 96
            time_slot = target_slot % 96
            if day_idx < len(day_indices):
                date = all_dates[day_indices[day_idx]]
                date_info.append({'date': date, 'time_slot': time_slot, 'global_slot': target_slot})
        
        X = np.array(X)
        y = np.array(y)
        
        # 转换active_indices回OD对
        od_pairs = []
        for idx in active_indices:
            o = idx // n_zones + 1  # 1-based
            d = idx % n_zones + 1
            od_pairs.append((o, d))
        
        return X, y, od_pairs, active_indices, date_info
    
    def train_model(self, hidden_size=64, num_layers=2, 
                   epochs=50, batch_size=16, learning_rate=0.001):
        """
        训练LSTM模型预测所有OD流量 - 支持训练/验证/测试集划分
        
        Args:
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            
        Returns:
            model: 训练好的模型
            history: 训练历史
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，无法训练模型")
        
        print("\n" + "="*80)
        print("训练LSTM模型（按日期划分数据集）")
        print("="*80)
        
        # 首先基于所有数据确定活跃OD对（确保一致性）
        print("\n确定全局活跃OD对...")
        all_od_data = []
        all_dates = sorted(self.df_temporal['date'].unique())
        for day_idx in range(len(all_dates)):
            all_od_data.append(self.od_flow_3d[day_idx])
        all_od_matrix = np.concatenate(all_od_data, axis=0)
        all_od_flat = all_od_matrix.reshape(all_od_matrix.shape[0], -1)
        total_flow = all_od_flat.sum(axis=0)
        active_indices = np.where(total_flow > 0)[0]
        print(f"全局活跃OD对数: {len(active_indices)} / {all_od_flat.shape[1]}")
        
        # 准备训练集数据（拟合scaler）
        X_train, y_train, od_pairs, active_indices, train_date_info = self.prepare_sequences(
            self.train_dates, active_indices=active_indices, fit_scaler=True)
        
        # 准备验证集数据（使用训练集的scaler）
        X_val, y_val, _, _, val_date_info = self.prepare_sequences(
            self.val_dates, active_indices=active_indices, fit_scaler=False)
        
        # 准备测试集数据（包含验证集最后一天，以提供跨天历史数据）
        # 这样测试日的前12个时间槽也能有足够的历史数据
        test_dates_with_history = list(self.val_dates[-1:]) + list(self.test_dates)
        X_test_all, y_test_all, _, _, test_date_info_all = self.prepare_sequences(
            test_dates_with_history, active_indices=active_indices, fit_scaler=False)
        
        # 只保留属于测试日期的样本
        test_indices = [i for i, info in enumerate(test_date_info_all) 
                       if info['date'] in self.test_dates]
        X_test = X_test_all[test_indices]
        y_test = y_test_all[test_indices]
        test_date_info = [test_date_info_all[i] for i in test_indices]
        
        print(f"\n序列形状:")
        print(f"  训练集 X: {X_train.shape}  # (样本数, 序列长度, OD对数)")
        print(f"  训练集 y: {y_train.shape}  # (样本数, OD对数)")
        print(f"  验证集 X: {X_val.shape}")
        print(f"  验证集 y: {y_val.shape}")
        print(f"  测试集 X: {X_test.shape}")
        print(f"  测试集 y: {y_test.shape}")
        
        print(f"\n数据划分:")
        print(f"  训练集: {len(X_train)} 样本 ({len(self.train_dates)}天)")
        print(f"  验证集: {len(X_val)} 样本 ({len(self.val_dates)}天)")
        print(f"  测试集: {len(X_test)} 样本 ({len(self.test_dates)}天)")
        
        # 创建数据加载器
        train_dataset = ODFlowDataset(X_train, y_train)
        val_dataset = ODFlowDataset(X_val, y_val)
        test_dataset = ODFlowDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        input_size = X_train.shape[2]  # OD对数
        output_size = y_train.shape[1]  # OD对数
        
        model = LSTMODPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=0.2
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
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 训练
        history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 15
        
        print(f"\n开始训练 (epochs={epochs}, batch_size={batch_size}, lr={learning_rate})...")
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_losses = []
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 前向传播
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
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
            
            # 测试阶段（仅用于监控，不参与早停）
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
            
            # 早停（基于验证集）
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), f'{self.data_dir}/best_lstm_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print(f"\n训练完成!")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        
        # 加载最佳模型
        model.load_state_dict(torch.load(f'{self.data_dir}/best_lstm_model.pth'))
        
        return model, history, X_val, y_val, X_test, y_test, od_pairs, active_indices, test_date_info
    
    def evaluate_model(self, model, X_test, y_test):
        """评估模型性能"""
        print("\n" + "="*80)
        print("模型评估")
        print("="*80)
        
        model.eval()
        
        # 转换数据
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            predictions = model(X_test_tensor).cpu().numpy()
        
        # 反归一化
        y_test_original = self.scaler.inverse_transform(y_test)
        predictions_original = self.scaler.inverse_transform(predictions)
        
        # 计算指标
        mae = mean_absolute_error(y_test_original.flatten(), predictions_original.flatten())
        rmse = np.sqrt(mean_squared_error(y_test_original.flatten(), predictions_original.flatten()))
        
        # MAPE (只计算非零值)
        mask = y_test_original.flatten() > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test_original.flatten()[mask] - predictions_original.flatten()[mask]) 
                                  / y_test_original.flatten()[mask])) * 100
        else:
            mape = 0
        
        # R²
        r2 = r2_score(y_test_original.flatten(), predictions_original.flatten())
        
        print(f"\n性能指标:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}")
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }, predictions_original
    
    def predict_all_timeslots(self, model, od_pairs, active_indices):
        """
        预测所有时间槽的OD流量
        
        Returns:
            predictions: 预测的OD流量矩阵 (T, n_zones, n_zones)
        """
        print("\n" + "="*80)
        print("预测所有时间槽")
        print("="*80)
        
        model.eval()
        T, n_zones, _ = self.od_flow.shape
        
        # 准备序列数据（使用全部数据）
        X, y, _, _ = self.prepare_sequences(self.od_flow)
        
        # 预测
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predictions_scaled = model(X_tensor).cpu().numpy()
        
        # 反归一化
        predictions = self.scaler.inverse_transform(predictions_scaled)
        
        # 重建完整的OD矩阵
        # 预测从第sequence_length个时间槽开始
        predictions_full = np.zeros((T, n_zones, n_zones))
        
        # 填充预测值
        for i, pred in enumerate(predictions):
            time_slot = i + self.sequence_length  # 对应的时间槽
            
            # 将预测值填回对应的OD对
            for j, (o, d) in enumerate(od_pairs):
                predictions_full[time_slot, o-1, d-1] = max(0, pred[j])  # 确保非负
        
        print(f"\n预测完成: {T}个时间槽, {len(od_pairs)}个活跃OD对")
        
        return predictions_full
    
    def save_predictions_to_csv(self, predictions, od_pairs, date_info, 
                                 output_file='output/od_flow_predictions_lstm.csv'):
        """
        将测试集预测结果保存为CSV格式（和od_flow_temporal.csv相同格式）
        
        Args:
            predictions: 预测值 (N, n_od_pairs)，已反归一化
            od_pairs: OD对列表 [(o, d), ...]
            date_info: 日期信息列表 [{'date': ..., 'time_slot': ..., ...}, ...]
            output_file: 输出文件路径
        """
        print("\n" + "="*80)
        print("保存预测结果到CSV")
        print("="*80)
        
        all_dates = sorted(self.df_temporal['date'].unique())
        n_zones = self.od_flow_3d.shape[2]
        
        # 创建列表存储所有记录
        records = []
        
        for i, pred_values in enumerate(predictions):
            info = date_info[i]
            date = info['date']
            time_slot = info['time_slot']
            
            # 计算小时和分钟
            hour = (time_slot * 15) // 60
            minute = (time_slot * 15) % 60
            
            # 计算day_index
            day_index = all_dates.index(date)
            
            for j, (o, d) in enumerate(od_pairs):
                flow = pred_values[j]
                
                # 只保存非零流量
                if flow > 0.01:  # 小阈值过滤噪声
                    # 获取平均时间（从原始数据）
                    avg_time = self.od_time_3d[day_index, time_slot, o-1, d-1]
                    
                    records.append({
                        'date': date,
                        'day_index': day_index,
                        'time_slot': time_slot,
                        'hour': hour,
                        'minute': minute,
                        'origin': o,
                        'dest': d,
                        'flow': round(flow, 6),
                        'avg_time': round(avg_time, 6),
                        'is_predicted': True  # 标记为预测值
                    })
        
        # 转换为DataFrame
        df = pd.DataFrame(records)
        
        # 保存
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ 预测结果已保存: {output_file}")
        print(f"  总记录数: {len(df):,}")
        print(f"  日期范围: {df['date'].min()} - {df['date'].max()}")
        print(f"  时间槽范围: {df['time_slot'].min()} - {df['time_slot'].max()}")
        print(f"  OD对数: {df.groupby(['origin', 'dest']).size().shape[0]}")
        
        return df
    
    def visualize_training(self, history):
        """可视化训练过程"""
        plt.figure(figsize=(18, 5))
        
        train_loss = np.array(history['train_loss'])
        val_loss = np.array(history['val_loss'])
        test_loss = np.array(history['test_loss'])
        
        plt.subplot(1, 3, 1)
        plt.plot(train_loss, label='Train Loss', linewidth=2)
        plt.plot(val_loss, label='Val Loss', linewidth=2)
        plt.plot(test_loss, label='Test Loss', linewidth=2, linestyle='--', alpha=0.7)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Training History', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(np.log10(train_loss + 1e-10), label='Train Loss (log)', linewidth=2)
        plt.plot(np.log10(val_loss + 1e-10), label='Val Loss (log)', linewidth=2)
        plt.plot(np.log10(test_loss + 1e-10), label='Test Loss (log)', linewidth=2, linestyle='--', alpha=0.7)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Log10(Loss)', fontsize=12)
        plt.title('Training History (Log Scale)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        epochs = range( 1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, 'o-', label='Train', linewidth=2, markersize=4)
        plt.plot(epochs, val_loss, 's-', label='Val', linewidth=2, markersize=4)
        plt.plot(epochs, test_loss, '^--', label='Test', linewidth=2, markersize=4, alpha=0.7)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Loss Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/lstm_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 训练历史已保存: lstm_training_history.png")
    
    def visualize_od_predictions(self, predictions, num_od_pairs=6):
        """可视化部分OD对的预测结果"""
        T, n_zones, _ = predictions.shape
        
        # 选择流量最大的几个OD对
        total_flow = predictions.sum(axis=0)
        flat_indices = np.argsort(total_flow.flatten())[::-1]
        
        fig, axes = plt.subplots(num_od_pairs, 1, figsize=(16, 3*num_od_pairs))
        if num_od_pairs == 1:
            axes = [axes]
        
        hours = np.arange(T) / 4  # 转换为小时
        
        for i in range(min(num_od_pairs, len(flat_indices))):
            idx = flat_indices[i]
            o = idx // n_zones + 1
            d = idx % n_zones + 1
            
            ax = axes[i]
            
            # 原始数据
            true_flow = self.od_flow[:, o-1, d-1]
            pred_flow = predictions[:, o-1, d-1]
            
            ax.plot(hours, true_flow, 'b-', linewidth=2, label='True', alpha=0.7)
            ax.plot(hours, pred_flow, 'r--', linewidth=2, label='Predicted', alpha=0.7)
            
            ax.set_xlabel('Hour of Day', fontsize=11)
            ax.set_ylabel('Flow (trips)', fontsize=11)
            ax.set_title(f'OD Pair: {o} → {d}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 标注预测范围
            pred_start = self.sequence_length / 4
            ax.axvline(x=pred_start, color='green', linestyle=':', alpha=0.5, label='Prediction Start')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/lstm_od_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ OD预测可视化已保存: lstm_od_predictions.png")


def main():
    """主函数"""
    print("="*80)
    print("基于LSTM的OD流量预测系统 - 多天时序版本")
    print("预测时间槽级别的OD流量（15分钟粒度）")
    print("="*80)
    
    if not TORCH_AVAILABLE:
        print("\n错误: PyTorch未安装!")
        print("请运行: pip install torch torchvision")
        return
    
    predictor = LSTMODFlowPredictor(data_dir='output')
    
    # 1. 加载数据并划分训练/验证/测试集
    print("\n" + "="*80)
    print("数据加载与划分")
    print("="*80)
    
    train_dates = ['2008-02-02', '2008-02-03', '2008-02-04', '2008-02-05', '2008-02-06']
    val_dates = ['2008-02-07']
    test_dates = ['2008-02-08']
    
    predictor.load_data(train_dates=train_dates, val_dates=val_dates, test_dates=test_dates)
    
    # 2. 训练LSTM模型
    print("\n" + "="*80)
    print("训练LSTM模型")
    print("="*80)
    
    model, history, X_val, y_val, X_test, y_test, od_pairs, active_indices, test_date_info = predictor.train_model(
        hidden_size=64,
        num_layers=2,
        epochs=100,
        batch_size=16,
        learning_rate=0.001
    )
    
    # 3. 可视化训练过程
    predictor.visualize_training(history)
    
    # 4. 评估模型（在测试集上）
    print("\n" + "="*80)
    print("模型评估（测试集）")
    print("="*80)
    
    metrics, predictions_test = predictor.evaluate_model(model, X_test, y_test)
    
    # 5. 保存测试集预测结果为CSV
    df_predictions = predictor.save_predictions_to_csv(
        predictions_test, 
        od_pairs,
        test_date_info,
        output_file='output/od_flow_predictions_lstm_test.csv'
    )
    
    # 6. 保存评估指标
    results_df = pd.DataFrame([metrics])
    results_df['model'] = 'LSTM'
    results_df['sequence_length'] = predictor.sequence_length
    results_df['prediction_horizon'] = predictor.prediction_horizon
    results_df['train_dates'] = ', '.join(train_dates)
    results_df['val_dates'] = ', '.join(val_dates)
    results_df['test_dates'] = ', '.join(test_dates)
    results_df.to_csv('output/lstm_evaluation.csv', index=False)
    print(f"\n✓ 评估结果已保存: lstm_evaluation.csv")
    
    # 7. 显示预测样例
    print("\n" + "="*80)
    print("测试集预测结果示例")
    print("="*80)
    print(df_predictions.head(20))
    
    # 8. 与真实值对比
    print("\n" + "="*80)
    print("预测 vs 真实值对比（测试集样例）")
    print("="*80)
    
    # 选择几个样本展示
    sample_indices = [0, len(test_date_info)//4, len(test_date_info)//2, -1]
    for idx in sample_indices:
        if idx >= len(test_date_info) or idx < 0:
            idx = len(test_date_info) - 1
        info = test_date_info[idx]
        print(f"\n样本 {idx}: {info['date']} 时间槽{info['time_slot']}")
        
        # 显示前5个OD对的对比
        y_true_sample = predictor.scaler.inverse_transform(y_test[idx:idx+1])[0]
        y_pred_sample = predictions_test[idx]
        
        for j in range(min(5, len(od_pairs))):
            o, d = od_pairs[j]
            true_val = y_true_sample[j]
            pred_val = y_pred_sample[j]
            if true_val > 0.1 or pred_val > 0.1:
                error = abs(true_val - pred_val)
                print(f"  OD {o}→{d}: 真实={true_val:.2f}, 预测={pred_val:.2f}, 误差={error:.2f}")
    
    print("\n" + "="*80)
    print("LSTM预测完成!")
    print("="*80)
    print("\n生成的文件:")
    print("  1. best_lstm_model.pth - 最佳模型权重")
    print("  2. lstm_training_history.png - 训练/验证/测试损失曲线")
    print("  3. od_flow_predictions_lstm_test.csv - 测试集预测结果CSV文件")
    print("  4. lstm_evaluation.csv - 性能评估指标")
    
    print("\n" + "="*80)
    print("性能指标（测试集）")
    print("="*80)
    print(f"  MAE:  {metrics['MAE']:.4f} trips")
    print(f"  RMSE: {metrics['RMSE']:.4f} trips")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R²:   {metrics['R2']:.4f}")
    
    print("\n说明:")
    print(f"  - 训练集: {', '.join(train_dates)} (5天)")
    print(f"  - 验证集: {', '.join(val_dates)} (1天，用于早停)")
    print(f"  - 测试集: {', '.join(test_dates)} (1天，最终评估)")
    print(f"  - 使用前{predictor.sequence_length}个时间槽（{predictor.sequence_length*15}分钟）预测下{predictor.prediction_horizon}个时间槽")
    print(f"  - 预测结果包含{len(od_pairs)}个活跃OD对")
    print(f"  - 输出格式与od_flow_temporal.csv相同，包含日期信息")
    print(f"  - avg_time从原始数据引用（不是预测值）")


if __name__ == '__main__':
    main()
