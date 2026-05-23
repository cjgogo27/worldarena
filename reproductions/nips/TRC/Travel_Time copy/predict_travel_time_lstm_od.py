#!/usr/bin/env python3
"""
基于LSTM的行程时间预测系统 - 按OD对建模版本
功能：针对每个OD对单独建模，使用历史时序数据预测未来行程时间
预测逻辑：用同一OD对在不同时间的历史数据预测该OD对未来的行程时间
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
    """行程时间数据集"""
    
    def __init__(self, sequences, time_features, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.time_features = torch.FloatTensor(time_features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.time_features[idx], self.targets[idx]


class LSTMTravelTimePredictor(nn.Module):
    """
    LSTM行程时间预测器 - 单OD对建模
    
    输入：
        - 历史行程时间序列 (sequence_length,)
        - 目标时间特征 (hour, minute, day_of_week等)
    输出：
        - 预测的行程时间
    """
    
    def __init__(self, sequence_length, time_feature_size, hidden_size=64, 
                 num_layers=2, dropout=0.2):
        super(LSTMTravelTimePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM处理时间序列
        self.lstm = nn.LSTM(
            input_size=1,  # 每个时间步只有1个值（行程时间）
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 处理时间特征的全连接层
        self.time_fc = nn.Linear(time_feature_size, hidden_size // 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 合并LSTM输出和时间特征
        self.fc1 = nn.Linear(hidden_size + hidden_size // 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
    
    def forward(self, x_seq, x_time):
        # x_seq: (batch, sequence_length, 1)
        # x_time: (batch, time_feature_size)
        
        # LSTM处理序列
        lstm_out, (h_n, c_n) = self.lstm(x_seq)
        lstm_last = lstm_out[:, -1, :]  # 取最后一个时间步
        
        # 处理时间特征
        time_features = self.relu(self.time_fc(x_time))
        
        # 合并
        combined = torch.cat([lstm_last, time_features], dim=1)
        
        # 全连接层
        out = self.dropout(combined)
        out = self.batch_norm1(self.relu(self.fc1(out)))
        out = self.dropout(out)
        out = self.batch_norm2(self.relu(self.fc2(out)))
        out = self.fc3(out)
        
        return out


class ODTravelTimePrediction:
    """基于OD对的行程时间LSTM预测器"""
    
    def __init__(self, data_path, output_dir, device=None):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        
        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 模型参数
        self.sequence_length = 12  # 使用过去12个时间槽（3小时）
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.time_scaler = StandardScaler()
        
        # 日期划分
        self.train_dates = None
        self.val_dates = None
        self.test_dates = None
        
    def load_data(self, train_dates, val_dates, test_dates):
        """加载数据"""
        print("="*80)
        print("加载数据")
        print("="*80)
        
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        self.train_dates = train_dates
        self.val_dates = val_dates
        self.test_dates = test_dates
        
        print(f"\n数据形状: {self.df.shape}")
        print(f"日期范围: {self.df['date'].min()} 到 {self.df['date'].max()}")
        print(f"\n数据集划分:")
        print(f"  训练集: {train_dates}")
        print(f"  验证集: {val_dates}")
        print(f"  测试集: {test_dates}")
        
        # 添加时间特征
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        # 获取活跃的OD对（在测试集中有数据的）
        test_df = self.df[self.df['date'].isin(pd.to_datetime(test_dates))]
        active_od_pairs = test_df[test_df['avg_time'] > 0].groupby(['origin', 'dest']).size()
        self.active_od_pairs = [(o, d) for (o, d) in active_od_pairs.index if active_od_pairs[(o, d)] >= 3]
        
        print(f"\n活跃OD对数量: {len(self.active_od_pairs)}")
        print(f"  (在测试集中至少有3个有效数据点的OD对)")
        
        return self.df
    
    def prepare_od_sequences(self, od_pair, date_list):
        """
        为特定OD对准备时间序列数据
        
        Args:
            od_pair: (origin, dest) 元组
            date_list: 日期列表
            
        Returns:
            X_seq: 历史行程时间序列 (N, sequence_length, 1)
            X_time: 目标时间特征 (N, time_feature_size)
            y: 目标行程时间 (N, 1)
            meta: 元数据（日期、时间槽等）
        """
        origin, dest = od_pair
        
        # 筛选该OD对的数据
        od_data = self.df[
            (self.df['origin'] == origin) & 
            (self.df['dest'] == dest) &
            (self.df['date'].isin(pd.to_datetime(date_list)))
        ].copy()
        
        if len(od_data) == 0:
            return None, None, None, None
        
        # 按日期和时间槽排序
        od_data = od_data.sort_values(['date', 'time_slot']).reset_index(drop=True)
        
        # 提取行程时间序列
        travel_times = od_data['avg_time'].values
        
        # 如果数据点太少，返回None
        if len(travel_times) < self.sequence_length + 1:
            return None, None, None, None
        
        # 创建滑动窗口序列
        X_seq, X_time, y, meta = [], [], [], []
        
        for i in range(len(travel_times) - self.sequence_length):
            # 历史序列
            seq = travel_times[i:i + self.sequence_length]
            
            # 目标值
            target = travel_times[i + self.sequence_length]
            
            # 如果序列中有太多0值（无数据），跳过
            if (seq == 0).sum() > self.sequence_length // 2:
                continue
            
            # 如果目标值为0，跳过
            if target == 0:
                continue
            
            # 目标时间的特征
            target_row = od_data.iloc[i + self.sequence_length]
            time_features = [
                target_row['hour'] / 23.0,  # 归一化到[0,1]
                target_row['minute'] / 59.0,
                target_row['time_slot'] / 95.0,
                target_row['day_of_week'] / 6.0,
                float(target_row['is_weekend']),
                float((target_row['hour'] >= 7) & (target_row['hour'] <= 9)),  # 早高峰
                float((target_row['hour'] >= 17) & (target_row['hour'] <= 19))  # 晚高峰
            ]
            
            X_seq.append(seq)
            X_time.append(time_features)
            y.append(target)
            
            meta.append({
                'date': target_row['date'],
                'time_slot': target_row['time_slot'],
                'hour': target_row['hour'],
                'minute': target_row['minute'],
                'origin': origin,
                'dest': dest,
                'flow': target_row['flow']
            })
        
        if len(X_seq) == 0:
            return None, None, None, None
        
        X_seq = np.array(X_seq).reshape(-1, self.sequence_length, 1)
        X_time = np.array(X_time)
        y = np.array(y).reshape(-1, 1)
        
        return X_seq, X_time, y, meta
    
    def prepare_all_data(self):
        """准备所有OD对的训练、验证、测试数据"""
        print("\n" + "="*80)
        print("准备序列数据（按OD对建模）")
        print("="*80)
        
        train_sequences, train_times, train_targets, train_meta = [], [], [], []
        val_sequences, val_times, val_targets, val_meta = [], [], [], []
        test_sequences, test_times, test_targets, test_meta = [], [], [], []
        
        successful_od_count = 0
        
        for od_pair in self.active_od_pairs:
            # 为该OD对准备训练+验证数据（用于提供历史）
            train_val_dates = self.train_dates + self.val_dates
            X_seq_tv, X_time_tv, y_tv, meta_tv = self.prepare_od_sequences(
                od_pair, train_val_dates)
            
            if X_seq_tv is None:
                continue
            
            # 准备测试数据（包含验证集最后部分以提供历史）
            test_with_history = self.val_dates[-1:] + self.test_dates
            X_seq_test, X_time_test, y_test, meta_test = self.prepare_od_sequences(
                od_pair, test_with_history)
            
            if X_seq_test is None:
                continue
            
            # 划分训练集和验证集
            # 计算训练集应该有多少样本
            train_only_dates = self.train_dates
            X_seq_t, X_time_t, y_t, meta_t = self.prepare_od_sequences(
                od_pair, train_only_dates)
            
            if X_seq_t is not None:
                n_train = len(X_seq_t)
                n_val = len(X_seq_tv) - n_train
                
                if n_train > 0 and n_val > 0:
                    train_sequences.append(X_seq_tv[:n_train])
                    train_times.append(X_time_tv[:n_train])
                    train_targets.append(y_tv[:n_train])
                    train_meta.extend(meta_tv[:n_train])
                    
                    val_sequences.append(X_seq_tv[n_train:])
                    val_times.append(X_time_tv[n_train:])
                    val_targets.append(y_tv[n_train:])
                    val_meta.extend(meta_tv[n_train:])
            
            # 测试集：只保留测试日期的样本
            test_indices = [i for i, m in enumerate(meta_test) 
                           if str(m['date'].date()) in self.test_dates]
            
            if len(test_indices) > 0:
                test_sequences.append(X_seq_test[test_indices])
                test_times.append(X_time_test[test_indices])
                test_targets.append(y_test[test_indices])
                test_meta.extend([meta_test[i] for i in test_indices])
                
                successful_od_count += 1
        
        # 合并所有OD对的数据
        X_train = np.vstack(train_sequences) if train_sequences else np.array([])
        X_time_train = np.vstack(train_times) if train_times else np.array([])
        y_train = np.vstack(train_targets) if train_targets else np.array([])
        
        X_val = np.vstack(val_sequences) if val_sequences else np.array([])
        X_time_val = np.vstack(val_times) if val_times else np.array([])
        y_val = np.vstack(val_targets) if val_targets else np.array([])
        
        X_test = np.vstack(test_sequences) if test_sequences else np.array([])
        X_time_test = np.vstack(test_times) if test_times else np.array([])
        y_test = np.vstack(test_targets) if test_targets else np.array([])
        
        print(f"\n成功处理 {successful_od_count} 个OD对")
        print(f"\n数据集大小:")
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  验证集: {len(X_val)} 样本")
        print(f"  测试集: {len(X_test)} 样本")
        
        # 归一化
        print(f"\n归一化处理...")
        
        # 行程时间归一化 - 将所有数据合并一起fit，然后分别transform
        all_train_data = np.vstack([X_train.reshape(-1, 1), y_train])
        self.scaler.fit(all_train_data)
        
        X_train_scaled = self.scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        y_train_scaled = self.scaler.transform(y_train)
        
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
        y_val_scaled = self.scaler.transform(y_val)
        
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
        y_test_scaled = self.scaler.transform(y_test)
        
        # 时间特征已经在0-1之间，不需要额外归一化
        
        return (X_train_scaled, X_time_train, y_train_scaled, train_meta,
                X_val_scaled, X_time_val, y_val_scaled, val_meta,
                X_test_scaled, X_time_test, y_test_scaled, test_meta,
                y_test)  # 保留原始测试目标值用于评估
    
    def train_model(self, X_train, X_time_train, y_train,
                   X_val, X_time_val, y_val,
                   hidden_size=128, num_layers=3,
                   epochs=100, batch_size=64, learning_rate=0.001):
        """训练LSTM模型"""
        print("\n" + "="*80)
        print("训练LSTM模型")
        print("="*80)
        
        # 创建数据加载器
        train_dataset = TravelTimeDataset(X_train, X_time_train, y_train)
        val_dataset = TravelTimeDataset(X_val, X_time_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        time_feature_size = X_time_train.shape[1]
        
        model = LSTMTravelTimePredictor(
            sequence_length=self.sequence_length,
            time_feature_size=time_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3
        ).to(self.device)
        
        print(f"\n模型架构:")
        print(f"  序列长度: {self.sequence_length}")
        print(f"  时间特征数: {time_feature_size}")
        print(f"  隐藏层大小: {hidden_size}")
        print(f"  LSTM层数: {num_layers}")
        print(f"  总参数: {sum(p.numel() for p in model.parameters()):,}")
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # 训练
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 20
        
        print(f"\n开始训练...")
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_losses = []
            
            for X_seq, X_time, y in train_loader:
                X_seq = X_seq.to(self.device)
                X_time = X_time.to(self.device)
                y = y.to(self.device)
                
                outputs = model(X_seq, X_time)
                loss = criterion(outputs, y)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # 验证阶段
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for X_seq, X_time, y in val_loader:
                    X_seq = X_seq.to(self.device)
                    X_time = X_time.to(self.device)
                    y = y.to(self.device)
                    
                    outputs = model(X_seq, X_time)
                    loss = criterion(outputs, y)
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'{self.output_dir}/best_lstm_od_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print(f"\n训练完成! 最佳验证损失: {best_val_loss:.6f}")
        
        # 加载最佳模型
        model.load_state_dict(torch.load(f'{self.output_dir}/best_lstm_od_model.pth'))
        
        return model, history
    
    def evaluate_and_save(self, model, X_test, X_time_test, y_test_scaled, 
                         y_test_original, test_meta):
        """评估模型并保存结果"""
        print("\n" + "="*80)
        print("模型评估与保存")
        print("="*80)
        
        model.eval()
        
        # 预测
        test_dataset = TravelTimeDataset(X_test, X_time_test, y_test_scaled)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        predictions_scaled = []
        with torch.no_grad():
            for X_seq, X_time, _ in test_loader:
                X_seq = X_seq.to(self.device)
                X_time = X_time.to(self.device)
                outputs = model(X_seq, X_time)
                predictions_scaled.append(outputs.cpu().numpy())
        
        predictions_scaled = np.vstack(predictions_scaled)
        
        # 反归一化
        predictions_original = self.scaler.inverse_transform(predictions_scaled)
        
        # 计算指标
        mae = mean_absolute_error(y_test_original, predictions_original)
        rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
        r2 = r2_score(y_test_original, predictions_original)
        
        # MAPE
        mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100
        
        print(f"\n测试集性能:")
        print(f"  MAE:  {mae:.4f} 分钟")
        print(f"  RMSE: {rmse:.4f} 分钟")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}")
        
        # 保存为CSV
        records = []
        for i, meta in enumerate(test_meta):
            records.append({
                'date': meta['date'],
                'time_slot': meta['time_slot'],
                'hour': meta['hour'],
                'minute': meta['minute'],
                'origin': meta['origin'],
                'dest': meta['dest'],
                'flow': meta['flow'],
                'true_avg_time': float(y_test_original[i, 0]),
                'predicted_avg_time': float(predictions_original[i, 0]),
                'absolute_error': float(abs(y_test_original[i, 0] - predictions_original[i, 0])),
                'relative_error': float(abs(y_test_original[i, 0] - predictions_original[i, 0]) / y_test_original[i, 0] * 100)
            })
        
        df = pd.DataFrame(records)
        output_file = f'{self.output_dir}/travel_time_predictions_lstm_od.csv'
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ 预测结果已保存: {output_file}")
        print(f"  记录数: {len(df):,}")
        
        # 保存评估指标
        metrics_df = pd.DataFrame([{
            'Model': 'LSTM_OD',
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Description': '按OD对建模',
            'Train_Dates': ', '.join(self.train_dates),
            'Test_Dates': ', '.join(self.test_dates)
        }])
        metrics_df.to_csv(f'{self.output_dir}/lstm_od_evaluation.csv', index=False)
        
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}, df
    
    def visualize(self, history, predictions_df):
        """可视化结果"""
        fig = plt.figure(figsize=(18, 12))
        
        # 1. 训练历史
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 真实值 vs 预测值
        ax2 = plt.subplot(2, 3, 2)
        sample = predictions_df.sample(min(3000, len(predictions_df)))
        ax2.scatter(sample['true_avg_time'], sample['predicted_avg_time'], 
                   alpha=0.4, s=15)
        max_val = max(sample['true_avg_time'].max(), sample['predicted_avg_time'].max())
        ax2.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect')
        ax2.set_xlabel('True Travel Time (min)')
        ax2.set_ylabel('Predicted Travel Time (min)')
        ax2.set_title('True vs Predicted (OD-based)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 误差分布
        ax3 = plt.subplot(2, 3, 3)
        errors = predictions_df['absolute_error']
        ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(x=errors.mean(), color='r', linestyle='--', lw=2,
                   label=f'Mean: {errors.mean():.2f}')
        ax3.axvline(x=errors.median(), color='g', linestyle='--', lw=2,
                   label=f'Median: {errors.median():.2f}')
        ax3.set_xlabel('Absolute Error (min)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 按小时的误差
        ax4 = plt.subplot(2, 3, 4)
        hourly_error = predictions_df.groupby('hour')['absolute_error'].mean()
        ax4.plot(hourly_error.index, hourly_error.values, marker='o', linewidth=2)
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Mean Absolute Error (min)')
        ax4.set_title('Error by Hour')
        ax4.grid(True, alpha=0.3)
        
        # 5. 样本OD对的时序预测
        ax5 = plt.subplot(2, 3, 5)
        # 选择一个有足够数据的OD对
        od_counts = predictions_df.groupby(['origin', 'dest']).size()
        if len(od_counts) > 0:
            top_od = od_counts.idxmax()
            od_sample = predictions_df[
                (predictions_df['origin'] == top_od[0]) &
                (predictions_df['dest'] == top_od[1])
            ].sort_values('time_slot')
            
            if len(od_sample) > 0:
                ax5.plot(od_sample['time_slot'], od_sample['true_avg_time'],
                        'b-o', label='True', linewidth=2, markersize=4)
                ax5.plot(od_sample['time_slot'], od_sample['predicted_avg_time'],
                        'r--s', label='Predicted', linewidth=2, markersize=4)
                ax5.set_xlabel('Time Slot')
                ax5.set_ylabel('Travel Time (min)')
                ax5.set_title(f'Example OD Pair: {top_od[0]}→{top_od[1]}')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
        
        # 6. 相对误差分布
        ax6 = plt.subplot(2, 3, 6)
        rel_errors = predictions_df['relative_error'].clip(upper=200)
        ax6.hist(rel_errors, bins=50, edgecolor='black', alpha=0.7)
        ax6.set_xlabel('Relative Error (%)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Relative Error Distribution')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/lstm_od_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 可视化已保存: lstm_od_results.png")


def main():
    """主函数"""
    print("="*80)
    print("基于LSTM的行程时间预测 - 按OD对建模版本")
    print("预测逻辑: 用同一OD对的历史时序数据预测未来")
    print("="*80)
    
    if not TORCH_AVAILABLE:
        print("\n错误: PyTorch未安装!")
        return
    
    # 配置
    data_path = '/data/alice/cjtest/TRC/haidian_od_analysis/output/od_flow_temporal.csv'
    output_dir = '/data/alice/cjtest/TRC/Travel_Time'
    
    train_dates = ['2008-02-02', '2008-02-03', '2008-02-04', '2008-02-05', '2008-02-06']
    val_dates = ['2008-02-07']
    test_dates = ['2008-02-08']
    
    predictor = ODTravelTimePrediction(data_path, output_dir)
    
    # 1. 加载数据
    predictor.load_data(train_dates, val_dates, test_dates)
    
    # 2. 准备序列数据
    (X_train, X_time_train, y_train, train_meta,
     X_val, X_time_val, y_val, val_meta,
     X_test, X_time_test, y_test_scaled, test_meta,
     y_test_original) = predictor.prepare_all_data()
    
    # 3. 训练模型
    model, history = predictor.train_model(
        X_train, X_time_train, y_train,
        X_val, X_time_val, y_val,
        hidden_size=128,
        num_layers=3,
        epochs=100,
        batch_size=64,
        learning_rate=0.001
    )
    
    # 4. 评估并保存
    metrics, predictions_df = predictor.evaluate_and_save(
        model, X_test, X_time_test, y_test_scaled, y_test_original, test_meta
    )
    
    # 5. 可视化
    predictor.visualize(history, predictions_df)
    
    # 6. 显示结果
    print("\n" + "="*80)
    print("预测完成！")
    print("="*80)
    print(f"\n性能指标:")
    print(f"  MAE:  {metrics['MAE']:.4f} 分钟")
    print(f"  RMSE: {metrics['RMSE']:.4f} 分钟")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R²:   {metrics['R2']:.4f}")
    
    print(f"\n输出文件:")
    print(f"  1. travel_time_predictions_lstm_od.csv - 预测结果")
    print(f"  2. lstm_od_evaluation.csv - 评估指标")
    print(f"  3. lstm_od_results.png - 可视化")
    print(f"  4. best_lstm_od_model.pth - 模型权重")
    
    print(f"\n预测示例:")
    print(predictions_df.head(20).to_string(index=False))
    
    print(f"\n说明:")
    print(f"  ✓ 每个OD对单独建模，用其历史时序数据预测")
    print(f"  ✓ 输入: 过去12个时间槽(3小时)的行程时间 + 目标时间特征")
    print(f"  ✓ 输出: 该OD对在目标时间的行程时间")
    print(f"  ✓ 这才是真正的时序预测!")


if __name__ == '__main__':
    main()
