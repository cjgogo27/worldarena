#!/usr/bin/env python3
"""
基于LSTM的行程时间预测系统 - 完整时间序列版本
功能：针对每个OD对建模，构建完整时间序列（缺失填0），预测未来行程时间

关键改进：
1. 使用过去4小时（16个时间槽）的数据预测下一时刻
2. 为每个OD对构建完整的96个时间槽序列，缺失的用0填充
3. 保证序列连续性，让LSTM能学到真正的时序规律
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装")

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


class LSTMPredictor(nn.Module):
    """LSTM行程时间预测模型"""
    
    def __init__(self, sequence_length, time_feature_size, hidden_size=64, 
                 num_layers=2, dropout=0.3):
        super(LSTMPredictor, self).__init__()
        
        # LSTM处理时间序列
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 时间特征处理
        self.time_fc = nn.Linear(time_feature_size, hidden_size // 2)
        
        # 合并层
        self.fc1 = nn.Linear(hidden_size + hidden_size // 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
    
    def forward(self, x_seq, x_time):
        # LSTM
        lstm_out, _ = self.lstm(x_seq)
        lstm_last = lstm_out[:, -1, :]
        
        # 时间特征
        time_feat = self.relu(self.time_fc(x_time))
        
        # 合并
        combined = torch.cat([lstm_last, time_feat], dim=1)
        
        # 全连接
        out = self.dropout(combined)
        out = self.batch_norm1(self.relu(self.fc1(out)))
        out = self.dropout(out)
        out = self.batch_norm2(self.relu(self.fc2(out)))
        out = self.fc3(out)
        
        return out


class CompleteTravelTimePrediction:
    """完整时间序列的行程时间预测"""
    
    def __init__(self, data_path, output_dir, device=None):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # ✓ 改为4小时
        self.sequence_length = 16  # 16个时间槽 = 4小时
        self.n_time_slots = 96  # 一天96个时间槽
        
        self.scaler = MinMaxScaler(feature_range=(0.01, 1))  # 避免0值问题
        
        # 日期划分
        self.train_dates = None
        self.val_dates = None
        self.test_dates = None
        
    def load_data(self, train_dates, val_dates, test_dates):
        """加载数据"""
        print("="*80)
        print("加载数据并构建完整时间序列")
        print("="*80)
        
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        self.train_dates = [pd.Timestamp(d) for d in train_dates]
        self.val_dates = [pd.Timestamp(d) for d in val_dates]
        self.test_dates = [pd.Timestamp(d) for d in test_dates]
        
        print(f"\n数据形状: {self.df.shape}")
        print(f"日期范围: {self.df['date'].min()} 到 {self.df['date'].max()}")
        print(f"\n数据集划分:")
        print(f"  训练集: {[str(d.date()) for d in self.train_dates]}")
        print(f"  验证集: {[str(d.date()) for d in self.val_dates]}")
        print(f"  测试集: {[str(d.date()) for d in self.test_dates]}")
        
        # 添加时间特征
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        # 获取所有OD对
        all_od = self.df.groupby(['origin', 'dest']).size()
        self.all_od_pairs = list(all_od.index)
        
        # 筛选活跃OD对（在测试集中有足够数据）
        test_df = self.df[self.df['date'].isin(self.test_dates)]
        test_od = test_df[test_df['avg_time'] > 0].groupby(['origin', 'dest']).size()
        self.active_od_pairs = [(o, d) for (o, d) in test_od.index if test_od[(o, d)] >= 5]
        
        print(f"\n总OD对数: {len(self.all_od_pairs)}")
        print(f"活跃OD对数: {len(self.active_od_pairs)} (测试集中至少5个有效数据点)")
        
        return self.df
    
    def build_complete_od_matrix(self, od_pair, date_list):
        """
        为OD对构建完整的时间序列矩阵 (days × 96 time_slots)
        缺失的时间槽用0填充
        
        这是关键改进！保证序列连续性
        """
        origin, dest = od_pair
        
        # 筛选该OD对的数据
        od_data = self.df[
            (self.df['origin'] == origin) & 
            (self.df['dest'] == dest) &
            (self.df['date'].isin(date_list))
        ].copy()
        
        n_days = len(date_list)
        
        # 初始化完整矩阵：(n_days, 96) 都填充为0
        travel_time_matrix = np.zeros((n_days, self.n_time_slots))
        
        # 填充实际数据
        for _, row in od_data.iterrows():
            day_idx = date_list.index(row['date'])
            time_slot = int(row['time_slot'])
            travel_time_matrix[day_idx, time_slot] = row['avg_time']
        
        # 展平为一维序列
        travel_time_sequence = travel_time_matrix.flatten()
        
        return travel_time_sequence, travel_time_matrix
    
    def create_sequences(self, od_pair, date_list):
        """
        为OD对创建训练序列
        
        输入说明：
            - 历史序列：过去16个时间槽（4小时）的行程时间
            - 时间特征：目标时刻的时间属性
        
        输出说明：
            - 预测的行程时间（1个值）
        """
        origin, dest = od_pair
        
        # 构建完整时间序列
        travel_times, _ = self.build_complete_od_matrix(od_pair, date_list)
        
        if len(travel_times) < self.sequence_length + 1:
            return None, None, None, None
        
        X_seq, X_time, y, meta = [], [], [], []
        
        # 滑动窗口
        for i in range(len(travel_times) - self.sequence_length):
            seq = travel_times[i:i + self.sequence_length]
            target = travel_times[i + self.sequence_length]
            
            # 只有当目标值>0时才训练（预测有数据的时刻）
            if target <= 0:
                continue
            
            # 计算目标时刻是哪一天的哪个时间槽
            target_idx = i + self.sequence_length
            day_idx = target_idx // self.n_time_slots
            time_slot = target_idx % self.n_time_slots
            
            if day_idx >= len(date_list):
                continue
            
            target_date = date_list[day_idx]
            hour = (time_slot * 15) // 60
            minute = (time_slot * 15) % 60
            day_of_week = target_date.dayofweek
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # 时间特征
            time_features = [
                hour / 23.0,
                minute / 59.0,
                time_slot / 95.0,
                day_of_week / 6.0,
                float(is_weekend),
                float(7 <= hour <= 9),  # 早高峰
                float(17 <= hour <= 19)  # 晚高峰
            ]
            
            X_seq.append(seq)
            X_time.append(time_features)
            y.append(target)
            
            meta.append({
                'date': target_date,
                'time_slot': time_slot,
                'hour': hour,
                'minute': minute,
                'origin': origin,
                'dest': dest,
                'day_of_week': day_of_week
            })
        
        if len(X_seq) == 0:
            return None, None, None, None
        
        X_seq = np.array(X_seq).reshape(-1, self.sequence_length, 1)
        X_time = np.array(X_time)
        y = np.array(y).reshape(-1, 1)
        
        return X_seq, X_time, y, meta
    
    def prepare_all_data(self):
        """准备所有数据"""
        print("\n" + "="*80)
        print("准备完整时间序列数据")
        print("="*80)
        print(f"\n关键参数:")
        print(f"  ✓ 序列长度: {self.sequence_length} 个时间槽 = {self.sequence_length * 15 / 60:.1f} 小时")
        print(f"  ✓ 每天时间槽数: {self.n_time_slots}")
        print(f"  ✓ 缺失数据填充: 0")
        print(f"  ✓ 序列连续性: 保证")
        
        train_X_seq, train_X_time, train_y, train_meta = [], [], [], []
        val_X_seq, val_X_time, val_y, val_meta = [], [], [], []
        test_X_seq, test_X_time, test_y, test_meta = [], [], [], []
        
        print(f"\n构建{len(self.active_od_pairs)}个OD对的完整时间序列...")
        
        for idx, od in enumerate(self.active_od_pairs):
            if (idx + 1) % 50 == 0:
                print(f"  进度: {idx+1}/{len(self.active_od_pairs)}")
            
            # 训练集+验证集（用于训练）
            train_val_dates = self.train_dates + self.val_dates
            X_seq_tv, X_time_tv, y_tv, meta_tv = self.create_sequences(od, train_val_dates)
            
            if X_seq_tv is not None:
                # 分割训练和验证
                n_train_days = len(self.train_dates)
                n_total_slots = n_train_days * self.n_time_slots
                
                train_mask = np.array([m['date'] in self.train_dates for m in meta_tv])
                val_mask = ~train_mask
                
                if train_mask.sum() > 0:
                    train_X_seq.append(X_seq_tv[train_mask])
                    train_X_time.append(X_time_tv[train_mask])
                    train_y.append(y_tv[train_mask])
                    train_meta.extend([meta_tv[i] for i in range(len(meta_tv)) if train_mask[i]])
                
                if val_mask.sum() > 0:
                    val_X_seq.append(X_seq_tv[val_mask])
                    val_X_time.append(X_time_tv[val_mask])
                    val_y.append(y_tv[val_mask])
                    val_meta.extend([meta_tv[i] for i in range(len(meta_tv)) if val_mask[i]])
            
            # 测试集（包含验证集最后一天提供历史）
            test_dates_with_hist = self.val_dates[-1:] + self.test_dates
            X_seq_test, X_time_test, y_test, meta_test = self.create_sequences(od, test_dates_with_hist)
            
            if X_seq_test is not None:
                # 只保留测试日期的样本
                test_mask = np.array([m['date'] in self.test_dates for m in meta_test])
                if test_mask.sum() > 0:
                    test_X_seq.append(X_seq_test[test_mask])
                    test_X_time.append(X_time_test[test_mask])
                    test_y.append(y_test[test_mask])
                    test_meta.extend([meta_test[i] for i in range(len(meta_test)) if test_mask[i]])
        
        # 合并所有OD对的数据
        X_train = np.vstack(train_X_seq) if train_X_seq else np.array([])
        X_time_train = np.vstack(train_X_time) if train_X_time else np.array([])
        y_train = np.vstack(train_y) if train_y else np.array([])
        
        X_val = np.vstack(val_X_seq) if val_X_seq else np.array([])
        X_time_val = np.vstack(val_X_time) if val_X_time else np.array([])
        y_val = np.vstack(val_y) if val_y else np.array([])
        
        X_test = np.vstack(test_X_seq) if test_X_seq else np.array([])
        X_time_test = np.vstack(test_X_time) if test_X_time else np.array([])
        y_test = np.vstack(test_y) if test_y else np.array([])
        
        print(f"\n数据集统计:")
        print(f"  训练集: {len(X_train):,} 样本")
        print(f"  验证集: {len(X_val):,} 样本")
        print(f"  测试集: {len(X_test):,} 样本")
        
        # 归一化
        print(f"\n归一化处理...")
        if len(y_train) > 0:
            self.scaler.fit(y_train)
            y_train_scaled = self.scaler.transform(y_train)
            y_val_scaled = self.scaler.transform(y_val) if len(y_val) > 0 else y_val
            y_test_scaled = self.scaler.transform(y_test) if len(y_test) > 0 else y_test
            
            # 序列也要归一化
            X_train_scaled = np.zeros_like(X_train)
            for i in range(len(X_train)):
                non_zero = X_train[i] > 0
                if non_zero.sum() > 0:
                    X_train_scaled[i][non_zero] = self.scaler.transform(X_train[i][non_zero])
            
            X_val_scaled = np.zeros_like(X_val)
            for i in range(len(X_val)):
                non_zero = X_val[i] > 0
                if non_zero.sum() > 0:
                    X_val_scaled[i][non_zero] = self.scaler.transform(X_val[i][non_zero])
            
            X_test_scaled = np.zeros_like(X_test)
            for i in range(len(X_test)):
                non_zero = X_test[i] > 0
                if non_zero.sum() > 0:
                    X_test_scaled[i][non_zero] = self.scaler.transform(X_test[i][non_zero])
        
        return (X_train_scaled, X_time_train, y_train_scaled, train_meta,
                X_val_scaled, X_time_val, y_val_scaled, val_meta,
                X_test_scaled, X_time_test, y_test_scaled, test_meta)
    
    def train_model(self, X_train, X_time_train, y_train,
                   X_val, X_time_val, y_val,
                   hidden_size=64, num_layers=2, epochs=100, 
                   batch_size=64, learning_rate=0.001):
        """训练模型"""
        print("\n" + "="*80)
        print("训练LSTM模型")
        print("="*80)
        
        # 数据加载器
        train_dataset = TravelTimeDataset(X_train, X_time_train, y_train)
        val_dataset = TravelTimeDataset(X_val, X_time_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 模型
        model = LSTMPredictor(
            sequence_length=self.sequence_length,
            time_feature_size=X_time_train.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3
        ).to(self.device)
        
        print(f"\n模型配置:")
        print(f"  输入1: 历史序列 (batch, {self.sequence_length}, 1) - 过去{self.sequence_length*15/60:.1f}小时的行程时间")
        print(f"  输入2: 时间特征 (batch, {X_time_train.shape[1]}) - [hour, minute, time_slot, ...]")
        print(f"  输出:   预测值 (batch, 1) - 下一时刻的行程时间")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience = 0
        max_patience = 20
        
        print(f"\n开始训练...")
        
        for epoch in range(epochs):
            # 训练
            model.train()
            train_losses = []
            for X_seq, X_time, y in train_loader:
                X_seq = X_seq.to(self.device)
                X_time = X_time.to(self.device)
                y = y.to(self.device)
                
                optimizer.zero_grad()
                pred = model(X_seq, X_time)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # 验证
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_seq, X_time, y in val_loader:
                    X_seq = X_seq.to(self.device)
                    X_time = X_time.to(self.device)
                    y = y.to(self.device)
                    pred = model(X_seq, X_time)
                    loss = criterion(pred, y)
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Train: {train_loss:.6f}, Val: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), f'{self.output_dir}/best_lstm_complete_model.pth')
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        model.load_state_dict(torch.load(f'{self.output_dir}/best_lstm_complete_model.pth'))
        print(f"\n训练完成！最佳验证loss: {best_val_loss:.6f}")
        
        return model, history
    
    def evaluate(self, model, X_test, X_time_test, y_test, test_meta):
        """评估并保存结果"""
        print("\n" + "="*80)
        print("模型评估")
        print("="*80)
        
        model.eval()
        X_seq_tensor = torch.FloatTensor(X_test).to(self.device)
        X_time_tensor = torch.FloatTensor(X_time_test).to(self.device)
        
        with torch.no_grad():
            y_pred_scaled = model(X_seq_tensor, X_time_tensor).cpu().numpy()
        
        # 反归一化
        y_true = self.scaler.inverse_transform(y_test)
        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        
        # 计算指标
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        print(f"\n测试集性能:")
        print(f"  MAE:  {mae:.4f} 分钟")
        print(f"  RMSE: {rmse:.4f} 分钟")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        
        # 保存结果
        results = []
        for i in range(len(test_meta)):
            meta = test_meta[i]
            results.append({
                'date': meta['date'],
                'time_slot': meta['time_slot'],
                'hour': meta['hour'],
                'minute': meta['minute'],
                'origin': meta['origin'],
                'dest': meta['dest'],
                'true_time': y_true[i, 0],
                'pred_time': y_pred[i, 0],
                'abs_error': abs(y_true[i, 0] - y_pred[i, 0]),
                'rel_error': abs(y_true[i, 0] - y_pred[i, 0]) / y_true[i, 0] * 100
            })
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(f'{self.output_dir}/lstm_complete_predictions.csv', index=False)
        
        print(f"\n✓ 预测结果已保存: lstm_complete_predictions.csv")
        print(f"  共{len(df_results):,}条预测记录")
        
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}, df_results


def main():
    """主函数"""
    print("="*80)
    print("基于完整时间序列的LSTM行程时间预测")
    print("="*80)
    
    if not TORCH_AVAILABLE:
        print("\n错误: PyTorch未安装")
        return
    
    data_path = '/data/alice/cjtest/TRC/haidian_od_analysis/output/od_flow_temporal.csv'
    output_dir = '/data/alice/cjtest/TRC/Travel_Time'
    
    predictor = CompleteTravelTimePrediction(data_path, output_dir)
    
    # 日期
    train_dates = ['2008-02-02', '2008-02-03', '2008-02-04', '2008-02-05', '2008-02-06']
    val_dates = ['2008-02-07']
    test_dates = ['2008-02-08']
    
    # 加载数据
    predictor.load_data(train_dates, val_dates, test_dates)
    
    # 准备数据
    (X_train, X_time_train, y_train, train_meta,
     X_val, X_time_val, y_val, val_meta,
     X_test, X_time_test, y_test, test_meta) = predictor.prepare_all_data()
    
    # 训练
    model, history = predictor.train_model(
        X_train, X_time_train, y_train,
        X_val, X_time_val, y_val,
        hidden_size=64,
        num_layers=2,
        epochs=100,
        batch_size=64,
        learning_rate=0.001
    )
    
    # 评估
    metrics, df_results = predictor.evaluate(model, X_test, X_time_test, y_test, test_meta)
    
    print("\n" + "="*80)
    print("预测完成！")
    print("="*80)
    print(f"\n最终性能 (使用4小时历史数据，完整时间序列):")
    print(f"  MAE:  {metrics['MAE']:.4f} 分钟")
    print(f"  RMSE: {metrics['RMSE']:.4f} 分钟")
    print(f"  R²:   {metrics['R2']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    print(f"\n说明:")
    print(f"  ✓ 使用过去 {predictor.sequence_length} 个时间槽（4小时）预测")
    print(f"  ✓ 为每个OD对构建完整96时间槽序列")
    print(f"  ✓ 缺失数据用0填充，保证序列连续")
    print(f"  ✓ 输出包含真实值和预测值对比")


if __name__ == '__main__':
    main()
