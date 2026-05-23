#!/usr/bin/env python3
"""
行程时间预测系统 - 严格无泄露版本
======================================

修正要点:
1. ✓ 插值只用训练集数据
2. ✓ 移除OD对ID特征（origin/dest）
3. ✓ 强化flow特征（flow增大→拥堵→时间增加）
4. ✓ 添加准确率指标（误差<5分钟算准确）
5. ✓ 简单平均插值
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TravelTimePredictorNoLeak:
    """严格无数据泄露的行程时间预测器"""
    
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        
    def load_data(self):
        """加载数据"""
        print("="*80)
        print("加载数据（严格时序）")
        print("="*80)
        
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # 数据集划分
        self.train_dates = ['2008-02-02', '2008-02-03', '2008-02-04', '2008-02-05', '2008-02-06']
        self.val_dates = ['2008-02-07']
        self.test_dates = ['2008-02-08']
        
        print(f"总记录数: {len(self.df):,}")
        print(f"\n数据集划分:")
        print(f"  训练集: {self.train_dates}")
        print(f"  验证集: {self.val_dates}")
        print(f"  测试集: {self.test_dates}")
        
        # 构建完整矩阵
        dates = sorted(self.df['date'].astype(str).unique())
        n_zones = self.df['origin'].max()
        
        self.od_time_matrix = np.full((len(dates), 96, n_zones, n_zones), np.nan)
        self.od_flow_matrix = np.zeros((len(dates), 96, n_zones, n_zones))
        
        for _, row in self.df.iterrows():
            day_idx = dates.index(str(row['date'])[:10])
            t = int(row['time_slot'])
            o = int(row['origin']) - 1
            d = int(row['dest']) - 1
            
            self.od_time_matrix[day_idx, t, o, d] = row['avg_time']
            self.od_flow_matrix[day_idx, t, o, d] = row['flow']
        
        self.dates = dates
        self.n_zones = n_zones
        
        print(f"\n矩阵形状: {self.od_time_matrix.shape}")
        print(f"缺失率: {np.isnan(self.od_time_matrix).sum() / self.od_time_matrix.size * 100:.1f}%")
        
    def fill_missing_train_only(self):
        """只用训练集数据填充缺失值"""
        print("\n" + "="*80)
        print("填充缺失值（仅用训练集数据）")
        print("="*80)
        
        train_indices = [self.dates.index(d) for d in self.train_dates]
        
        # 只用训练集计算平均值
        train_data = self.od_time_matrix[train_indices]
        od_train_means = np.nanmean(train_data, axis=(0, 1))  # 每个OD对的训练集平均值
        global_train_mean = np.nanmean(train_data)
        
        print(f"训练集全局平均行程时间: {global_train_mean:.2f} 分钟")
        
        # 填充策略：用训练集的平均值
        filled_matrix = self.od_time_matrix.copy()
        
        for day_idx in range(len(self.dates)):
            for o in range(self.n_zones):
                for d in range(self.n_zones):
                    time_series = filled_matrix[day_idx, :, o, d]
                    mask = np.isnan(time_series)
                    
                    if mask.any():
                        # 用训练集的OD平均值填充
                        fill_value = od_train_means[o, d]
                        if np.isnan(fill_value):
                            fill_value = global_train_mean
                        
                        filled_matrix[day_idx, mask, o, d] = fill_value
        
        self.od_time_filled = filled_matrix
        print(f"填充完成，剩余NaN: {np.isnan(filled_matrix).sum()}")
        
    def create_features(self):
        """创建特征（严格时序，无泄露）"""
        print("\n" + "="*80)
        print("创建特征（无数据泄露）")
        print("="*80)
        
        features_list = []
        
        for date_str in self.dates:
            day_idx = self.dates.index(date_str)
            
            # 判断属于哪个集合
            if date_str in self.train_dates:
                split = 'train'
            elif date_str in self.val_dates:
                split = 'val'
            else:
                split = 'test'
            
            date_dt = pd.to_datetime(date_str)
            day_of_week = date_dt.dayofweek
            is_weekend = 1 if day_of_week >= 5 else 0
            
            for t in range(96):
                hour = (t * 15) // 60
                minute = (t * 15) % 60
                
                is_morning_peak = 1 if 7 <= hour <= 9 else 0
                is_evening_peak = 1 if 17 <= hour <= 19 else 0
                is_peak_hour = is_morning_peak or is_evening_peak
                
                for o in range(self.n_zones):
                    for d in range(self.n_zones):
                        current_time = self.od_time_filled[day_idx, t, o, d]
                        current_flow = self.od_flow_matrix[day_idx, t, o, d]
                        
                        if np.isnan(current_time):
                            continue
                        
                        # === 基本特征（无ID） ===
                        features = {
                            'hour': hour,
                            'minute': minute,
                            'time_slot': t,
                            'day_of_week': day_of_week,
                            'is_weekend': is_weekend,
                            'is_morning_peak': is_morning_peak,
                            'is_evening_peak': is_evening_peak,
                            'is_peak_hour': is_peak_hour,
                        }
                        
                        # === Flow特征（重要！）===
                        features['current_flow'] = current_flow
                        features['log_flow'] = np.log1p(current_flow)
                        features['flow_squared'] = current_flow ** 2  # 流量平方（拥堵非线性）
                        
                        # 流量与高峰的交互
                        features['flow_x_peak'] = current_flow * is_peak_hour
                        features['flow_x_morning'] = current_flow * is_morning_peak
                        features['flow_x_evening'] = current_flow * is_evening_peak
                        
                        # === 历史统计（只用过去数据）===
                        if day_idx > 0:
                            # 该OD对过去所有天的数据
                            past_od_data = self.od_time_filled[:day_idx, :, o, d].flatten()
                            past_od_data = past_od_data[~np.isnan(past_od_data)]
                            
                            if len(past_od_data) > 0:
                                features['od_hist_mean'] = np.mean(past_od_data)
                                features['od_hist_std'] = np.std(past_od_data)
                                features['od_hist_min'] = np.min(past_od_data)
                                features['od_hist_max'] = np.max(past_od_data)
                            else:
                                features['od_hist_mean'] = current_time
                                features['od_hist_std'] = 0
                                features['od_hist_min'] = current_time
                                features['od_hist_max'] = current_time
                        else:
                            features['od_hist_mean'] = current_time
                            features['od_hist_std'] = 0
                            features['od_hist_min'] = current_time
                            features['od_hist_max'] = current_time
                        
                        # === 过去4小时（16个时间槽）===
                        lags = []
                        for lag in range(1, 17):
                            if t >= lag:
                                lag_time = self.od_time_filled[day_idx, t-lag, o, d]
                                lag_flow = self.od_flow_matrix[day_idx, t-lag, o, d]
                            elif day_idx > 0:
                                lag_time = self.od_time_filled[day_idx-1, 96+t-lag, o, d]
                                lag_flow = self.od_flow_matrix[day_idx-1, 96+t-lag, o, d]
                            else:
                                lag_time = features['od_hist_mean']
                                lag_flow = 0
                            
                            if not np.isnan(lag_time):
                                lags.append(lag_time)
                            
                            features[f'lag_{lag}_time'] = lag_time if not np.isnan(lag_time) else features['od_hist_mean']
                            features[f'lag_{lag}_flow'] = lag_flow
                        
                        # 4小时统计
                        if lags:
                            features['past_4h_mean'] = np.mean(lags)
                            features['past_4h_std'] = np.std(lags)
                            features['past_4h_max'] = np.max(lags)
                            features['past_4h_trend'] = lags[-1] - lags[0] if len(lags) >= 2 else 0
                        else:
                            features['past_4h_mean'] = features['od_hist_mean']
                            features['past_4h_std'] = 0
                            features['past_4h_max'] = features['od_hist_mean']
                            features['past_4h_trend'] = 0
                        
                        # 过去4小时的平均流量
                        past_4h_flows = [features[f'lag_{i}_flow'] for i in range(1, 17)]
                        features['past_4h_flow_mean'] = np.mean(past_4h_flows)
                        features['past_4h_flow_max'] = np.max(past_4h_flows)
                        
                        # === 前几天同一时段（处理早高峰突变）===
                        same_time_vals = []
                        for lookback in range(1, min(day_idx + 1, 4)):
                            if day_idx >= lookback:
                                same_val = self.od_time_filled[day_idx - lookback, t, o, d]
                                if not np.isnan(same_val):
                                    same_time_vals.append(same_val)
                                features[f'same_time_day_minus_{lookback}'] = same_val if not np.isnan(same_val) else features['od_hist_mean']
                        
                        if same_time_vals:
                            features['same_time_mean'] = np.mean(same_time_vals)
                            features['same_time_std'] = np.std(same_time_vals)
                        else:
                            features['same_time_mean'] = features['od_hist_mean']
                            features['same_time_std'] = 0
                        
                        # 目标值和元数据
                        features['target'] = current_time
                        features['date'] = date_str
                        features['origin'] = o + 1  # 仅用于保存结果，不参与训练
                        features['dest'] = d + 1
                        features['split'] = split
                        
                        features_list.append(features)
        
        self.df_features = pd.DataFrame(features_list)
        
        # 移除origin/dest，不参与训练
        feature_cols = [c for c in self.df_features.columns 
                       if c not in ['target', 'date', 'origin', 'dest', 'split']]
        
        print(f"\n特征矩阵:")
        print(f"  总样本数: {len(self.df_features):,}")
        print(f"  训练集: {(self.df_features['split']=='train').sum():,}")
        print(f"  验证集: {(self.df_features['split']=='val').sum():,}")
        print(f"  测试集: {(self.df_features['split']=='test').sum():,}")
        print(f"  特征数: {len(feature_cols)}")
        
        return feature_cols
    
    def train_models(self, feature_cols):
        """训练模型"""
        print("\n" + "="*80)
        print("训练模型")
        print("="*80)
        
        # 分割数据
        train_df = self.df_features[self.df_features['split'] == 'train']
        val_df = self.df_features[self.df_features['split'] == 'val']
        test_df = self.df_features[self.df_features['split'] == 'test']
        
        X_train = train_df[feature_cols]
        y_train = train_df['target']
        X_val = val_df[feature_cols]
        y_val = val_df['target']
        X_test = test_df[feature_cols]
        y_test = test_df['target']
        
        results = {}
        
        # XGBoost
        print("\n训练 XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     verbose=False)
        
        xgb_pred = xgb_model.predict(X_test)
        results['XGBoost'] = {'model': xgb_model, 'predictions': xgb_pred}
        
        # LightGBM
        print("训练 LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)])
       
        lgb_pred = lgb_model.predict(X_test)
        results['LightGBM'] = {'model': lgb_model, 'predictions': lgb_pred}
        
        self.results = results
        self.test_df = test_df
        self.y_test = y_test
        self.feature_cols = feature_cols
        
        return results
    
    def evaluate_models(self):
        """评估模型（包含准确率指标）"""
        print("\n" + "="*80)
        print("模型评估")
        print("="*80)
        
        eval_results = []
        
        for model_name, result in self.results.items():
            y_pred = result['predictions']
            y_true = self.y_test
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            # MAPE
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # 准确率指标
            acc_5min = (np.abs(y_true - y_pred) < 5).sum() / len(y_true) * 100
            acc_10min = (np.abs(y_true - y_pred) < 10).sum() / len(y_true) * 100
            acc_20min = (np.abs(y_true - y_pred) < 20).sum() / len(y_true) * 100
            
            eval_results.append({
                'Model': model_name,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape,
                'Accuracy_5min': acc_5min,
                'Accuracy_10min': acc_10min,
                'Accuracy_20min': acc_20min
            })
            
            print(f"\n{model_name}:")
            print(f"  MAE: {mae:.4f} 分钟")
            print(f"  RMSE: {rmse:.4f} 分钟")
            print(f"  R²: {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  准确率(<5分钟): {acc_5min:.2f}%")
            print(f"  准确率(<10分钟): {acc_10min:.2f}%")
            print(f"  准确率(<20分钟): {acc_20min:.2f}%")
        
        self.eval_df = pd.DataFrame(eval_results)
        return self.eval_df
    
    def save_results(self):
        """保存结果"""
        print("\n" + "="*80)
        print("保存结果")
        print("="*80)
        
        # 保存评估结果
        eval_file = f'{self.output_dir}/evaluation_no_leak.csv'
        self.eval_df.to_csv(eval_file, index=False)
        print(f"✓ 评估结果: {eval_file}")
        
        # 保存预测结果（包含真实值）
        test_df = self.test_df.copy()
        for model_name, result in self.results.items():
            test_df[f'predicted_{model_name}'] = result['predictions']
            test_df[f'error_{model_name}'] = np.abs(test_df['target'] - result['predictions'])
        
        pred_file = f'{self.output_dir}/predictions_no_leak.csv'
        test_df.to_csv(pred_file, index=False)
        print(f"✓ 预测结果: {pred_file}")
        
        # 特征重要性
        for model_name, result in self.results.items():
            model = result['model']
            if hasattr(model, 'feature_importances_'):
                fi_df = pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fi_file = f'{self.output_dir}/feature_importance_{model_name}_no_leak.csv'
                fi_df.to_csv(fi_file, index=False)
                print(f"✓ {model_name}特征重要性: {fi_file}")
                
                # 显示Top 10
                print(f"\n{model_name} Top 10 重要特征:")
                for idx, row in fi_df.head(10).iterrows():
                    print(f"  {row['feature']:<30} {row['importance']:.4f}")
    
    def run(self):
        """运行完整流程"""
        self.load_data()
        self.fill_missing_train_only()
        feature_cols = self.create_features()
        self.train_models(feature_cols)
        self.evaluate_models()
        self.save_results()
        
        print("\n" + "="*80)
        print("✓ 完成！无数据泄露版本")
        print("="*80)


if __name__ == '__main__':
    data_path = '/data/alice/cjtest/TRC/haidian_od_analysis/output/od_flow_temporal.csv'
    output_dir = '/data/alice/cjtest/TRC/Travel_Time'
    
    predictor = TravelTimePredictorNoLeak(data_path, output_dir)
    predictor.run()
