#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于Transformer的OD流量预测 - 日周期性方法

预测逻辑：
  - 利用过去N天同一时间槽的流量预测今天该时间槽
  - 例如：用2月2-6日的00:00流量预测2月8日的00:00
  - 使用Transformer捕捉日间的长期依赖关系

与LSTM方法的区别：
  - LSTM: 根据过去3小时（12槽）预测下一个15分钟
  - Transformer: 根据过去5天同一时刻预测今天该时刻
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 检查PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装")


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerODPredictor(nn.Module):
    """
    Transformer模型用于OD流量预测（日周期性）
    
    输入：过去N天同一时间槽的OD流量 + 时间槽编码
    输出：预测今天该时间槽的OD流量
    """
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, 
                 dropout=0.1, max_timeslots=96):
        super().__init__()
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 时间槽嵌入（0-95）
        self.timeslot_embedding = nn.Embedding(max_timeslots, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, input_size)
        )
        
    def forward(self, x, timeslot_ids):
        """
        Args:
            x: (batch, seq_len, input_size) - 过去N天的OD流量
            timeslot_ids: (batch,) - 时间槽ID (0-95)
        
        Returns:
            (batch, input_size) - 预测的OD流量
        """
        # 输入投影
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # 添加时间槽信息到每个序列位置
        timeslot_emb = self.timeslot_embedding(timeslot_ids)  # (batch, d_model)
        timeslot_emb = timeslot_emb.unsqueeze(1)  # (batch, 1, d_model)
        x = x + timeslot_emb  # 广播加法
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # 取最后一个位置的输出（或平均池化）
        x = x[:, -1, :]  # (batch, d_model)
        
        # 输出预测
        output = self.output_layer(x)  # (batch, input_size)
        
        return output


class ODFlowDataset(Dataset):
    """PyTorch数据集"""
    def __init__(self, X, y, timeslot_ids):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.timeslot_ids = torch.LongTensor(timeslot_ids)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.timeslot_ids[idx]


class TransformerPeriodicPredictor:
    """基于Transformer的日周期性OD流量预测器"""
    
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
        
        # 模型参数
        self.history_days = 5  # 使用过去5天的数据
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # 日期划分
        self.train_dates = None
        self.val_dates = None
        self.test_dates = None
        
    def load_data(self, train_dates=None, val_dates=None, test_dates=None):
        """加载多天OD流量数据"""
        print("="*80)
        print("加载多天时序数据（日周期性方法）")
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
            train_dates = all_dates[:5]  # 2月2-6日
            val_dates = [all_dates[5]] if len(all_dates) > 5 else []  # 2月7日
            test_dates = [all_dates[6]] if len(all_dates) > 6 else []  # 2月8日
        
        self.train_dates = train_dates
        self.val_dates = val_dates
        self.test_dates = test_dates
        
        print(f"\n数据集划分:")
        print(f"  训练集: {train_dates} ({len(train_dates)}天)")
        print(f"  验证集: {val_dates} ({len(val_dates)}天)")
        print(f"  测试集: {test_dates} ({len(test_dates)}天)")
        
        # 构建3D OD流量矩阵
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
                o = int(row['origin']) - 1
                d = int(row['dest']) - 1
                self.od_flow_3d[day_idx, t, o, d] = row['flow']
                self.od_time_3d[day_idx, t, o, d] = row['avg_time']
        
        print(f"\n3D矩阵形状: {self.od_flow_3d.shape}")
        print(f"  维度: (天数={n_days}, 时间槽={n_time_slots}, 起点={n_zones}, 终点={n_zones})")
        print(f"  总flow: {self.od_flow_3d.sum():.0f} trips")
        
    def prepare_periodic_sequences(self, date_list, active_indices=None, fit_scaler=False):
        """
        准备日周期性序列数据
        
        对于每个时间槽t (0-95):
          - 输入X: date_list中所有日期在时间槽t的OD流量
          - 输出y: 下一个日期在时间槽t的OD流量
        
        Args:
            date_list: 日期列表
            active_indices: 活跃OD对索引
            fit_scaler: 是否拟合scaler
        
        Returns:
            X: (N, history_days, n_od_pairs) - 过去N天同一时间槽的流量
            y: (N, n_od_pairs) - 目标日期该时间槽的流量
            timeslot_ids: (N,) - 时间槽ID
            od_pairs: OD对列表
            date_info: 每个样本的日期和时间槽信息
        """
        all_dates = sorted(self.df_temporal['date'].unique())
        n_zones = self.od_flow_3d.shape[2]
        n_time_slots = 96
        
        # 获取日期索引
        day_indices = [all_dates.index(date) for date in date_list]
        
        print(f"\n准备日周期性序列（{len(date_list)}天）")
        print(f"  方法: 用过去{self.history_days}天同一时间槽预测今天")
        
        # 如果没有active_indices，基于所有数据计算
        if active_indices is None:
            all_od_data = self.od_flow_3d.reshape(-1, n_zones * n_zones)
            total_flow = all_od_data.sum(axis=0)
            active_indices = np.where(total_flow > 0)[0]
            print(f"  活跃OD对数: {len(active_indices)} / {n_zones*n_zones}")
        else:
            print(f"  使用预定义活跃OD对数: {len(active_indices)} / {n_zones*n_zones}")
        
        X, y = [], []
        timeslot_ids = []
        date_info = []
        
        # 对每个时间槽
        for t in range(n_time_slots):
            # 收集所有日期该时间槽的数据
            timeslot_data = []  # (n_days, n_od_pairs)
            
            for day_idx in day_indices:
                od_matrix = self.od_flow_3d[day_idx, t, :, :]  # (n_zones, n_zones)
                od_flat = od_matrix.flatten()[active_indices]  # (n_od_pairs,)
                timeslot_data.append(od_flat)
            
            timeslot_data = np.array(timeslot_data)  # (n_days, n_od_pairs)
            
            # 创建样本：用前history_days天预测下一天
            for i in range(len(date_list) - self.history_days):
                X_sample = timeslot_data[i:i+self.history_days]  # (history_days, n_od_pairs)
                y_sample = timeslot_data[i+self.history_days]  # (n_od_pairs,)
                
                X.append(X_sample)
                y.append(y_sample)
                timeslot_ids.append(t)
                
                # 记录信息
                target_date = date_list[i+self.history_days]
                date_info.append({
                    'date': target_date,
                    'time_slot': t,
                    'history_dates': date_list[i:i+self.history_days]
                })
        
        X = np.array(X)  # (N, history_days, n_od_pairs)
        y = np.array(y)  # (N, n_od_pairs)
        timeslot_ids = np.array(timeslot_ids)  # (N,)
        
        print(f"  生成样本数: {len(X)}")
        print(f"  每个时间槽样本数: {len(date_list) - self.history_days}")
        print(f"  总时间槽数: {n_time_slots}")
        
        # 归一化
        if fit_scaler:
            # 将所有数据展平进行归一化
            X_flat = X.reshape(-1, X.shape[-1])
            y_flat = y
            all_data = np.vstack([X_flat, y_flat])
            self.scaler.fit(all_data)
            print(f"  拟合scaler: min={self.scaler.data_min_[:5]}, max={self.scaler.data_max_[:5]}")
        
        # 转换
        X_scaled = np.array([self.scaler.transform(x) for x in X])
        y_scaled = self.scaler.transform(y)
        
        # 转换active_indices回OD对
        od_pairs = []
        for idx in active_indices:
            o = idx // n_zones + 1
            d = idx % n_zones + 1
            od_pairs.append((o, d))
        
        return X_scaled, y_scaled, timeslot_ids, od_pairs, active_indices, date_info
    
    def train_model(self, d_model=128, nhead=8, num_layers=3,
                   epochs=100, batch_size=32, learning_rate=0.001):
        """训练Transformer模型"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安装")
        
        print("\n" + "="*80)
        print("训练Transformer模型（日周期性方法）")
        print("="*80)
        
        # 确定全局活跃OD对
        print("\n确定全局活跃OD对...")
        all_dates = sorted(self.df_temporal['date'].unique())
        all_od_data = self.od_flow_3d.reshape(-1, self.od_flow_3d.shape[2] * self.od_flow_3d.shape[3])
        total_flow = all_od_data.sum(axis=0)
        active_indices = np.where(total_flow > 0)[0]
        print(f"全局活跃OD对数: {len(active_indices)}")
        
        # 准备训练集
        X_train, y_train, ts_train, od_pairs, active_indices, train_info = \
            self.prepare_periodic_sequences(self.train_dates, active_indices, fit_scaler=True)
        
        # 准备验证集（如果有）
        if self.val_dates:
            # 验证集需要包含训练集的最后几天作为历史
            val_with_history = self.train_dates[-(self.history_days-1):] + self.val_dates
            X_val_all, y_val_all, ts_val_all, _, _, val_info_all = \
                self.prepare_periodic_sequences(val_with_history, active_indices, fit_scaler=False)
            
            # 只保留验证日期的样本
            val_indices = [i for i, info in enumerate(val_info_all) if info['date'] in self.val_dates]
            X_val = X_val_all[val_indices]
            y_val = y_val_all[val_indices]
            ts_val = ts_val_all[val_indices]
            val_info = [val_info_all[i] for i in val_indices]
        else:
            X_val, y_val, ts_val, val_info = None, None, None, None
        
        # 准备测试集
        test_with_history = self.train_dates[-(self.history_days-1):] + self.val_dates + self.test_dates
        X_test_all, y_test_all, ts_test_all, _, _, test_info_all = \
            self.prepare_periodic_sequences(test_with_history, active_indices, fit_scaler=False)
        
        test_indices = [i for i, info in enumerate(test_info_all) if info['date'] in self.test_dates]
        X_test = X_test_all[test_indices]
        y_test = y_test_all[test_indices]
        ts_test = ts_test_all[test_indices]
        test_info = [test_info_all[i] for i in test_indices]
        
        print(f"\n序列形状:")
        print(f"  训练集 X: {X_train.shape}  # (样本数, 历史天数, OD对数)")
        print(f"  训练集 y: {y_train.shape}  # (样本数, OD对数)")
        if X_val is not None:
            print(f"  验证集 X: {X_val.shape}")
            print(f"  验证集 y: {y_val.shape}")
        print(f"  测试集 X: {X_test.shape}")
        print(f"  测试集 y: {y_test.shape}")
        
        # 创建数据加载器
        train_dataset = ODFlowDataset(X_train, y_train, ts_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = ODFlowDataset(X_val, y_val, ts_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        test_dataset = ODFlowDataset(X_test, y_test, ts_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        input_size = X_train.shape[2]  # OD对数
        
        model = TransformerODPredictor(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=0.1,
            max_timeslots=96
        ).to(self.device)
        
        print(f"\n模型架构:")
        print(f"  输入维度: {input_size} (活跃OD对)")
        print(f"  d_model: {d_model}")
        print(f"  注意力头数: {nhead}")
        print(f"  Transformer层数: {num_layers}")
        print(f"  总参数: {sum(p.numel() for p in model.parameters()):,}")
        
        # 损失和优化器
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # 训练
        history = {'train_loss': [], 'val_loss': [], 'test_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 20
        
        print(f"\n开始训练...")
        
        for epoch in range(epochs):
            # 训练
            model.train()
            train_losses = []
            
            for X_batch, y_batch, ts_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                ts_batch = ts_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch, ts_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            history['train_loss'].append(train_loss)
            
            # 验证
            if X_val is not None:
                model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for X_batch, y_batch, ts_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        ts_batch = ts_batch.to(self.device)
                        
                        outputs = model(X_batch, ts_batch)
                        loss = criterion(outputs, y_batch)
                        val_losses.append(loss.item())
                
                val_loss = np.mean(val_losses)
                history['val_loss'].append(val_loss)
                scheduler.step(val_loss)
            else:
                val_loss = train_loss
                history['val_loss'].append(val_loss)
            
            # 测试（监控）
            model.eval()
            test_losses = []
            with torch.no_grad():
                for X_batch, y_batch, ts_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    ts_batch = ts_batch.to(self.device)
                    
                    outputs = model(X_batch, ts_batch)
                    loss = criterion(outputs, y_batch)
                    test_losses.append(loss.item())
            
            test_loss = np.mean(test_losses)
            history['test_loss'].append(test_loss)
            
            # 打印
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train: {train_loss:.6f}, Val: {val_loss:.6f}, Test: {test_loss:.6f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'{self.data_dir}/best_transformer_periodic_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print(f"\n训练完成! 最佳验证损失: {best_val_loss:.6f}")
        
        # 加载最佳模型
        model.load_state_dict(torch.load(f'{self.data_dir}/best_transformer_periodic_model.pth'))
        
        return model, history, X_test, y_test, ts_test, od_pairs, active_indices, test_info
    
    def evaluate_model(self, model, X_test, y_test, ts_test):
        """评估模型"""
        print("\n" + "="*80)
        print("模型评估")
        print("="*80)
        
        model.eval()
        
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        ts_tensor = torch.LongTensor(ts_test).to(self.device)
        
        with torch.no_grad():
            predictions = model(X_tensor, ts_tensor).cpu().numpy()
        
        # 反归一化
        y_test_original = self.scaler.inverse_transform(y_test)
        predictions_original = self.scaler.inverse_transform(predictions)
        
        # 计算指标
        mae = mean_absolute_error(y_test_original.flatten(), predictions_original.flatten())
        rmse = np.sqrt(mean_squared_error(y_test_original.flatten(), predictions_original.flatten()))
        
        mask = y_test_original.flatten() > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test_original.flatten()[mask] - predictions_original.flatten()[mask]) 
                                  / y_test_original.flatten()[mask])) * 100
        else:
            mape = 0
        
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
    
    def save_predictions_to_csv(self, predictions, od_pairs, test_info, 
                                output_file='output/od_flow_predictions_transformer_test.csv'):
        """保存预测结果为CSV"""
        print("\n" + "="*80)
        print("保存预测结果")
        print("="*80)
        
        all_dates = sorted(self.df_temporal['date'].unique())
        n_zones = self.od_flow_3d.shape[2]
        
        records = []
        
        for i, pred_values in enumerate(predictions):
            info = test_info[i]
            date = info['date']
            time_slot = info['time_slot']
            
            hour = (time_slot * 15) // 60
            minute = (time_slot * 15) % 60
            day_index = all_dates.index(date)
            
            for j, (o, d) in enumerate(od_pairs):
                flow = pred_values[j]
                
                if flow > 0.01:
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
                        'is_predicted': True
                    })
        
        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ 预测结果已保存: {output_file}")
        print(f"  总记录数: {len(df):,}")
        print(f"  日期范围: {df['date'].min()} - {df['date'].max()}")
        
        return df
    
    def visualize_training(self, history):
        """可视化训练过程"""
        plt.figure(figsize=(18, 5))
        
        train_loss = np.array(history['train_loss'])
        val_loss = np.array(history['val_loss'])
        test_loss = np.array(history['test_loss'])
        
        # Log尺度
        plt.subplot(1, 3, 1)
        plt.plot(np.log10(train_loss + 1e-10), label='Train Loss (log)', linewidth=2)
        plt.plot(np.log10(val_loss + 1e-10), label='Val Loss (log)', linewidth=2)
        plt.plot(np.log10(test_loss + 1e-10), label='Test Loss (log)', linewidth=2, linestyle='--', alpha=0.7)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Log10(Loss)', fontsize=12)
        plt.title('Training History (Log Scale)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 线性尺度
        plt.subplot(1, 3, 2)
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, 'o-', label='Train', linewidth=2, markersize=4)
        plt.plot(epochs, val_loss, 's-', label='Val', linewidth=2, markersize=4)
        plt.plot(epochs, test_loss, '^--', label='Test', linewidth=2, markersize=4, alpha=0.7)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Loss Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 最后50轮
        plt.subplot(1, 3, 3)
        start_idx = max(0, len(train_loss) - 50)
        epochs_last = range(start_idx + 1, len(train_loss) + 1)
        plt.plot(epochs_last, train_loss[start_idx:], 'o-', label='Train', linewidth=2, markersize=4)
        plt.plot(epochs_last, val_loss[start_idx:], 's-', label='Val', linewidth=2, markersize=4)
        plt.plot(epochs_last, test_loss[start_idx:], '^--', label='Test', linewidth=2, markersize=4, alpha=0.7)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Last 50 Epochs', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/transformer_periodic_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 训练历史已保存")


def main():
    """主函数"""
    print("="*80)
    print("基于Transformer的OD流量预测 - 日周期性方法")
    print("预测逻辑: 用过去N天同一时间槽预测今天该时间槽")
    print("="*80)
    
    if not TORCH_AVAILABLE:
        print("\n错误: PyTorch未安装!")
        return
    
    predictor = TransformerPeriodicPredictor(data_dir='output')
    
    # 1. 加载数据
    train_dates = ['2008-02-02', '2008-02-03', '2008-02-04', '2008-02-05', '2008-02-06']
    val_dates = ['2008-02-07']
    test_dates = ['2008-02-08']
    
    predictor.load_data(train_dates=train_dates, val_dates=val_dates, test_dates=test_dates)
    
    # 2. 训练模型
    model, history, X_test, y_test, ts_test, od_pairs, active_indices, test_info = predictor.train_model(
        d_model=128,
        nhead=8,
        num_layers=3,
        epochs=150,
        batch_size=32,
        learning_rate=0.001
    )
    
    # 3. 可视化训练
    predictor.visualize_training(history)
    
    # 4. 评估
    metrics, predictions_test = predictor.evaluate_model(model, X_test, y_test, ts_test)
    
    # 5. 保存预测结果
    df_predictions = predictor.save_predictions_to_csv(
        predictions_test,
        od_pairs,
        test_info,
        output_file='output/od_flow_predictions_transformer_periodic_test.csv'
    )
    
    # 6. 保存评估指标
    results_df = pd.DataFrame([metrics])
    results_df['model'] = 'Transformer_Periodic'
    results_df['history_days'] = predictor.history_days
    results_df['train_dates'] = ', '.join(train_dates)
    results_df['val_dates'] = ', '.join(val_dates)
    results_df['test_dates'] = ', '.join(test_dates)
    results_df.to_csv('output/transformer_periodic_evaluation.csv', index=False)
    
    print("\n" + "="*80)
    print("Transformer日周期性预测完成!")
    print("="*80)
    print("\n生成的文件:")
    print("  1. best_transformer_periodic_model.pth - 模型权重")
    print("  2. transformer_periodic_training_history.png - 训练曲线")
    print("  3. od_flow_predictions_transformer_periodic_test.csv - 测试集预测")
    print("  4. transformer_periodic_evaluation.csv - 评估指标")
    
    print("\n性能指标（测试集）:")
    print(f"  MAE:  {metrics['MAE']:.4f} trips")
    print(f"  RMSE: {metrics['RMSE']:.4f} trips")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R²:   {metrics['R2']:.4f}")
    
    print("\n说明:")
    print(f"  - 训练集: {', '.join(train_dates)} ({len(train_dates)}天)")
    print(f"  - 验证集: {', '.join(val_dates)} ({len(val_dates)}天)")
    print(f"  - 测试集: {', '.join(test_dates)} ({len(test_dates)}天)")
    print(f"  - 预测方法: 用过去{predictor.history_days}天同一时间槽预测今天该槽")
    print(f"  - 模型: Transformer with {active_indices.shape[0]} OD pairs")
    print("="*80)


if __name__ == '__main__':
    main()
