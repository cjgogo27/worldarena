#!/usr/bin/env python3
"""
基于Transformer的OD流量预测
使用多头自注意力机制捕捉时空依赖关系
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ODTransformer(nn.Module):
    """基于Transformer的OD流量预测模型"""
    
    def __init__(self, input_dim=1, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=512, dropout=0.1, output_dim=1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, output_dim)
        )
    
    def forward(self, src):
        # src: (batch_size, seq_len, input_dim)
        
        # 输入嵌入
        src = self.input_embedding(src)  # (batch, seq_len, d_model)
        
        # 位置编码
        src = self.pos_encoder(src)
        
        # Transformer编码
        output = self.transformer_encoder(src)  # (batch, seq_len, d_model)
        
        # 只使用最后一个时间步的输出
        output = output[:, -1, :]  # (batch, d_model)
        
        # 输出预测
        output = self.fc_out(output)  # (batch, output_dim)
        
        return output


class ODFlowDataset(Dataset):
    """OD流量数据集"""
    
    def __init__(self, flow_data, seq_len=24, pred_len=1):
        """
        Args:
            flow_data: 流量数据 (时间步数, OD对数)
            seq_len: 输入序列长度（历史时间步数）
            pred_len: 预测长度（未来时间步数）
        """
        self.data = flow_data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 生成时间特征
        self.time_features = self._generate_time_features()
        
        # 创建序列
        self.sequences = []
        for i in range(len(flow_data) - seq_len - pred_len + 1):
            x = flow_data[i:i+seq_len]
            y = flow_data[i+seq_len:i+seq_len+pred_len]
            
            # 添加时间特征
            time_feat = self.time_features[i:i+seq_len]
            x_with_time = np.concatenate([x, time_feat], axis=-1)
            
            self.sequences.append((x_with_time, y))
    
    def _generate_time_features(self):
        """生成时间周期特征"""
        n_steps = len(self.data)
        features = []
        
        for t in range(n_steps):
            # 每日周期 (96个时间槽)
            hour_sin = np.sin(2 * np.pi * (t % 96) / 96)
            hour_cos = np.cos(2 * np.pi * (t % 96) / 96)
            
            # 周周期 (假设7天数据，7*96个时间槽)
            if n_steps > 96 * 7:
                week_sin = np.sin(2 * np.pi * (t % (96*7)) / (96*7))
                week_cos = np.cos(2 * np.pi * (t % (96*7)) / (96*7))
            else:
                week_sin = 0
                week_cos = 0
            
            features.append([hour_sin, hour_cos, week_sin, week_cos])
        
        return np.array(features)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x, y = self.sequences[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class ODFlowPredictor:
    """OD流量预测器"""
    
    def __init__(self, od_pair, model_params=None):
        """
        Args:
            od_pair: (origin, dest) OD对
            model_params: 模型参数字典
        """
        self.od_pair = od_pair
        self.model_params = model_params or {
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1
        }
        
        self.scaler = StandardScaler()
        self.model = None
        self.train_losses = []
        self.val_losses = []
    
    def prepare_data(self, flow_series, seq_len=24, train_ratio=0.8):
        """准备训练和验证数据"""
        # 数据标准化
        flow_scaled = self.scaler.fit_transform(flow_series.reshape(-1, 1))
        
        # 创建数据集
        dataset = ODFlowDataset(flow_scaled, seq_len=seq_len, pred_len=1)
        
        # 划分训练集和验证集
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        return train_dataset, val_dataset
    
    def build_model(self, input_dim):
        """构建模型"""
        self.model = ODTransformer(
            input_dim=input_dim,
            output_dim=1,
            **self.model_params
        ).to(device)
        
        return self.model
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """训练模型"""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print(f"\n开始训练 OD对 {self.od_pair}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters())} 个")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    output = self.model(x)
                    loss = criterion(output, y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 学习率调整
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), f'output/best_transformer_od_{self.od_pair[0]}_{self.od_pair[1]}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停于 epoch {epoch+1}")
                    break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(f'output/best_transformer_od_{self.od_pair[0]}_{self.od_pair[1]}.pth'))
    
    def predict(self, x):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(device)
            if len(x_tensor.shape) == 2:
                x_tensor = x_tensor.unsqueeze(0)
            pred = self.model(x_tensor)
            return self.scaler.inverse_transform(pred.cpu().numpy())
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = self.model(x)
                predictions.extend(pred.cpu().numpy())
                actuals.extend(y.cpu().numpy())
        
        predictions = self.scaler.inverse_transform(np.array(predictions))
        actuals = self.scaler.inverse_transform(np.array(actuals))
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-10))) * 100
        r2 = r2_score(actuals, predictions)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'predictions': predictions,
            'actuals': actuals
        }


def main():
    """主函数"""
    print("="*80)
    print("基于Transformer的OD流量预测")
    print("="*80)
    
    # 加载OD流量数据
    print("\n加载数据...")
    od_flow = np.load('output/od_flow_full.npy')  # (96, 29, 29)
    print(f"OD流量矩阵形状: {od_flow.shape}")
    
    # 转换为时间序列格式 (时间步, OD对)
    # 这里假设有7天数据（实际需要更多历史数据才能充分训练）
    n_time_slots = od_flow.shape[0]
    n_regions = od_flow.shape[1]
    
    # 展开为 (时间步, 841个OD对)
    flow_series = od_flow.reshape(n_time_slots, -1)
    print(f"时间序列形状: {flow_series.shape}")
    
    # 选择流量最大的Top 10 OD对进行演示
    total_flow = flow_series.sum(axis=0)
    top_k = 10
    top_od_indices = np.argsort(total_flow)[-top_k:]
    
    print(f"\n选择流量最大的{top_k}个OD对进行训练:")
    results = []
    
    os.makedirs('output/transformer_models', exist_ok=True)
    os.makedirs('output/transformer_predictions', exist_ok=True)
    
    for idx, od_idx in enumerate(top_od_indices):
        o = od_idx // n_regions
        d = od_idx % n_regions
        
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{top_k}] 训练 OD对: 区域{o+1} → 区域{d+1}")
        print(f"总流量: {total_flow[od_idx]:.0f} trips")
        
        # 获取该OD对的时间序列
        od_series = flow_series[:, od_idx]
        
        # 如果流量太小，跳过
        if total_flow[od_idx] < 100:
            print("流量过低，跳过")
            continue
        
        # 创建预测器
        predictor = ODFlowPredictor(
            od_pair=(o+1, d+1),
            model_params={
                'd_model': 64,
                'nhead': 4,
                'num_layers': 2,
                'dim_feedforward': 256,
                'dropout': 0.1
            }
        )
        
        # 准备数据
        train_dataset, val_dataset = predictor.prepare_data(
            od_series, seq_len=24, train_ratio=0.8
        )
        
        if len(train_dataset) < 10:
            print("训练数据不足，跳过")
            continue
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 获取输入维度（流量 + 时间特征）
        sample_x, _ = train_dataset[0]
        input_dim = sample_x.shape[-1]
        
        # 构建并训练模型
        predictor.build_model(input_dim)
        predictor.train(train_loader, val_loader, epochs=50, lr=0.001)
        
        # 评估
        metrics = predictor.evaluate(val_loader)
        results.append({
            'OD': f"{o+1}->{d+1}",
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE'],
            'R2': metrics['R2']
        })
        
        print(f"\n评估结果:")
        print(f"  MAE:  {metrics['MAE']:.2f}")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²:   {metrics['R2']:.4f}")
        
        # 可视化预测结果
        plt.figure(figsize=(14, 5))
        
        # 训练曲线
        plt.subplot(1, 2, 1)
        plt.plot(predictor.train_losses, label='Train Loss')
        plt.plot(predictor.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Curves - OD {o+1}→{d+1}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 预测vs实际
        plt.subplot(1, 2, 2)
        plt.scatter(metrics['actuals'], metrics['predictions'], alpha=0.5)
        plt.plot([metrics['actuals'].min(), metrics['actuals'].max()],
                [metrics['actuals'].min(), metrics['actuals'].max()],
                'r--', label='Perfect Prediction')
        plt.xlabel('Actual Flow')
        plt.ylabel('Predicted Flow')
        plt.title(f'Prediction Results - OD {o+1}→{d+1}\nR²={metrics["R2"]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'output/transformer_predictions/od_{o+1}_{d+1}.png', dpi=150)
        plt.close()
    
    # 保存结果摘要
    results_df = pd.DataFrame(results)
    results_df.to_csv('output/transformer_prediction_results.csv', index=False)
    
    print("\n" + "="*80)
    print("预测结果汇总:")
    print("="*80)
    print(results_df.to_string(index=False))
    
    print(f"\n平均性能:")
    print(f"  MAE:  {results_df['MAE'].mean():.2f}")
    print(f"  RMSE: {results_df['RMSE'].mean():.2f}")
    print(f"  MAPE: {results_df['MAPE'].mean():.2f}%")
    print(f"  R²:   {results_df['R2'].mean():.4f}")
    
    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print("输出文件:")
    print("  - output/transformer_prediction_results.csv")
    print("  - output/transformer_models/best_transformer_od_*.pth")
    print("  - output/transformer_predictions/od_*.png")


if __name__ == '__main__':
    main()
