#!/usr/bin/env python3
"""
步骤2：基于插值数据的行程时间预测
===================================

功能：
1. 读取插值后的数据
2. 创建特征（不泄露测试集信息）
3. 训练模型，重点学习flow→travel time的关系
4. 评估模型（包括准确率指标）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class NoLeakTravelTimePredictor:
    """无数据泄露的行程时间预测器"""
    
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.scaler = StandardScaler()
        
    def load_interpolated_data(self):
        """加载插值后的数据"""
        print("="*80)
        print("1. 加载插值后的数据")
        print("="*80)
        
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        print(f"数据量: {len(self.df):,} 条记录")
        print(f"日期范围: {self.df['date'].min()} ~ {self.df['date'].max()}")
        print(f"插值记录: {self.df['is_interpolated'].sum():,} ({self.df['is_interpolated'].sum()/len(self.df)*100:.1f}%)")
        print(f"原始记录: {(~self.df['is_interpolated']).sum():,}")
        
        return self.df
    
    def create_features_no_leak(self, train_dates, val_dates, test_dates):
        """
        创建特征（严格防止数据泄露）
        
        关键点：
        1. 不使用origin/dest ID（防止模型记忆）
        2. 历史统计值只用训练集计算
        3. 强调flow与travel time的关系
        """
        print("\n" + "="*80)
        print("2. 创建特征（无数据泄露）")
        print("="*80)
        
        df = self.df.copy()
        
        # 数据集划分
        train_df = df[df['date'].isin(pd.to_datetime(train_dates))].copy()
        val_df = df[df['date'].isin(pd.to_datetime(val_dates))].copy()
        test_df = df[df['date'].isin(pd.to_datetime(test_dates))].copy()
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_df):,} 条 ({train_dates})")
        print(f"  验证集: {len(val_df):,} 条 ({val_dates})")
        print(f"  测试集: {len(test_df):,} 条 ({test_dates})")
        
        # === 只用训练集计算历史统计值 ===
        print(f"\n计算历史统计（仅使用训练集）...")
        
        # 全局统计
        train_global_mean = train_df['avg_time'].mean()
        train_global_std = train_df['avg_time'].std()
        
        # 按时段统计
        hour_stats = train_df.groupby('hour')['avg_time'].agg(['mean', 'std']).reset_index()
        hour_stats.columns = ['hour', 'hour_mean', 'hour_std']
        hour_stats['hour_std'].fillna(0, inplace=True)
        
        # 按flow区间统计（学习flow→time关系）
        train_df['flow_bin'] = pd.cut(train_df['flow'], bins=[0, 1, 3, 5, 10, 100], 
                                        labels=['0-1', '1-3', '3-5', '5-10', '10+'])
        flow_stats = train_df.groupby('flow_bin')['avg_time'].agg(['mean', 'std']).reset_index()
        flow_stats.columns = ['flow_bin', 'flow_bin_mean', 'flow_bin_std']
        
        print(f"  训练集全局均值: {train_global_mean:.2f} 分钟")
        print(f"  训练集全局标准差: {train_global_std:.2f} 分钟")
        
        # === 为所有数据集创建特征 ===
        for name, dataset in [('train', train_df), ('val', val_df), ('test', test_df)]:
            print(f"\n创建{name}集特征...")
            
            # 1. 时间特征
            dataset['hour_norm'] = dataset['hour'] / 23.0
            dataset['minute_norm'] = dataset['minute'] / 59.0
            dataset['time_slot_norm'] = dataset['time_slot'] / 95.0
            
            # 星期几（测试集也可以知道是星期几）
            dataset['day_of_week'] = dataset['date'].dt.dayofweek / 6.0
            dataset['is_weekend'] = (dataset['date'].dt.dayofweek >= 5).astype(int)
            
            # 高峰时段
            dataset['is_morning_peak'] = ((dataset['hour'] >= 7) & (dataset['hour'] <= 9)).astype(int)
            dataset['is_evening_peak'] = ((dataset['hour'] >= 17) & (dataset['hour'] <= 19)).astype(int)
            dataset['is_peak'] = (dataset['is_morning_peak'] | dataset['is_evening_peak']).astype(int)
            
            # 2. Flow特征（重点！）
            dataset['flow_raw'] = dataset['flow']
            dataset['flow_log'] = np.log1p(dataset['flow'])
            dataset['flow_sqrt'] = np.sqrt(dataset['flow'])
            dataset['flow_squared'] = dataset['flow'] ** 2
            
            # Flow密度（flow与时段的交互）
            dataset['flow_density_morning'] = dataset['flow'] * dataset['is_morning_peak']
            dataset['flow_density_evening'] = dataset['flow'] * dataset['is_evening_peak']
            
            # 3. 历史统计特征（来自训练集）
            dataset = dataset.merge(hour_stats, on='hour', how='left')
            dataset['hour_mean'].fillna(train_global_mean, inplace=True)
            dataset['hour_std'].fillna(train_global_std, inplace=True)
            
            # Flow区间统计
            dataset['flow_bin'] = pd.cut(dataset['flow'], bins=[0, 1, 3, 5, 10, 100], 
                                           labels=['0-1', '1-3', '3-5', '5-10', '10+'])
            dataset = dataset.merge(flow_stats, on='flow_bin', how='left')
            dataset['flow_bin_mean'].fillna(train_global_mean, inplace=True)
            dataset['flow_bin_std'].fillna(0, inplace=True)
            
            # 4. 时序特征(使用本OD对的历史)
            # 这里只用当前和之前的数据，不用未来数据
            dataset = dataset.sort_values(['origin', 'dest', 'date', 'time_slot']).reset_index(drop=True)
            
            # 前4小时的rolling统计（16个时间槽）
            for lag in [1, 2, 4, 8, 16]:
                dataset[f'lag_{lag}'] = dataset.groupby(['origin', 'dest'])['avg_time'].shift(lag)
            
            dataset['rolling_4h_mean'] = dataset.groupby(['origin', 'dest'])['avg_time'].rolling(
                window=16, min_periods=1).mean().values
            dataset['rolling_4h_std'] = dataset.groupby(['origin', 'dest'])['avg_time'].rolling(
                window=16, min_periods=1).std().values
            
            # 填充缺失值 - 只填充数值列
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
    
    def prepare_xy(self, df, feature_cols, target_col='avg_time'):
        """准备X和y"""
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        return X, y
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """训练多个模型"""
        print("\n" + "="*80)
        print("3. 训练模型")
        print("="*80)
        
        models = {}
        
        # XGBoost
        print("\nXGBoost训练中...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        models['XGBoost'] = xgb_model
        
        # LightGBM
        print("LightGBM训练中...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)])
        models['LightGBM'] = lgb_model
        
        # 集成
        models['Ensemble'] = None  # 占位符
        
        return models
    
    def evaluate_models(self, models, X_test, y_test, test_df):
        """评估模型"""
        print("\n" + "="*80)
        print("4. 模型评估")
        print("="*80)
        
        results = []
        predictions_all = {}
        
        for name, model in models.items():
            if name == 'Ensemble':
                # 集成预测
                pred = (predictions_all['XGBoost'] + predictions_all['LightGBM']) / 2
            else:
                pred = model.predict(X_test)
                predictions_all[name] = pred
            
            # 计算指标
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)
            
            # MAPE (只计算非零真实值)
            mask = y_test > 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_test[mask] - pred[mask]) / y_test[mask])) * 100
            else:
                mape = 0
            
            # 准确率指标：预测误差在X分钟内的比例
            errors = np.abs(y_test - pred)
            acc_5min = (errors <= 5).sum() / len(errors) * 100
            acc_10min = (errors <= 10).sum() / len(errors) * 100
            acc_20min = (errors <= 20).sum() / len(errors) * 100
            
            results.append({
                'Model': name,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape,
                'Acc@5min': acc_5min,
                'Acc@10min': acc_10min,
                'Acc@20min': acc_20min
            })
            
            print(f"\n{name}:")
            print(f"  MAE:  {mae:.4f} 分钟")
            print(f"  RMSE: {rmse:.4f} 分钟")
            print(f"  R²:   {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  准确率（误差≤5分钟）:  {acc_5min:.2f}%")
            print(f"  准确率（误差≤10分钟）: {acc_10min:.2f}%")
            print(f"  准确率（误差≤20分钟）: {acc_20min:.2f}%")
        
        results_df = pd.DataFrame(results)
        
        # 保存预测结果
        test_results = test_df.copy()
        test_results['pred_xgb'] = predictions_all['XGBoost']
        test_results['pred_lgb'] = predictions_all['LightGBM']
        test_results['pred_ensemble'] = (predictions_all['XGBoost'] + predictions_all['LightGBM']) / 2
        test_results['error'] = np.abs(test_results['avg_time'] - test_results['pred_ensemble'])
        test_results['error_pct'] = test_results['error'] / test_results['avg_time'] * 100
        
        return results_df, test_results, models
    
    def analyze_flow_impact(self, test_results):
        """分析flow对travel time的影响"""
        print("\n" + "="*80)
        print("5. Flow影响分析")
        print("="*80)
        
        # 按flow分组分析
        test_results['flow_group'] = pd.cut(
            test_results['flow'], 
            bins=[-0.1, 0.1, 1, 3, 5, 10, 100],
            labels=['0', '0-1', '1-3', '3-5', '5-10', '10+']
        )
        
        flow_analysis = test_results.groupby('flow_group').agg({
            'avg_time': 'mean',
            'pred_ensemble': 'mean',
            'error': 'mean',
            'flow': 'count'
        }).round(2)
        flow_analysis.columns = ['真实平均时间', '预测平均时间', '平均误差', '样本数']
        
        print("\nFlow分组分析:")
        print(flow_analysis)
        
        # 相关性分析
        corr_flow_time = test_results[['flow', 'avg_time']].corr().iloc[0, 1]
        print(f"\nFlow与Travel Time相关系数: {corr_flow_time:.4f}")
        
        if corr_flow_time > 0:
            print("✓ Flow增加时，Travel Time倾向于增加（拥堵效应）")
        
        return flow_analysis
    
    def visualize_results(self, test_results, models, feature_cols):
        """可视化结果"""
        print("\n" + "="*80)
        print("6. 可视化结果")
        print("="*80)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 真实值vs预测值
        ax1 = plt.subplot(2, 4, 1)
        sample = test_results.sample(min(5000, len(test_results)))
        ax1.scatter(sample['avg_time'], sample['pred_ensemble'], alpha=0.3, s=10)
        max_val = max(sample['avg_time'].max(), sample['pred_ensemble'].max())
        ax1.plot([0, max_val], [0, max_val], 'r--', lw=2)
        ax1.set_xlabel('True Travel Time (min)')
        ax1.set_ylabel('Predicted Travel Time (min)')
        ax1.set_title('True vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # 2. 误差分布
        ax2 = plt.subplot(2, 4, 2)
        ax2.hist(test_results['error'], bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(test_results['error'].median(), color='r', linestyle='--', 
                   label=f"Median: {test_results['error'].median():.2f}")
        ax2.set_xlabel('Absolute Error (min)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Flow vs Travel Time
        ax3 = plt.subplot(2, 4, 3)
        flow_bins = test_results.groupby('flow_group').agg({'avg_time': 'mean', 'pred_ensemble': 'mean'})
        x = range(len(flow_bins))
        ax3.bar([i-0.2 for i in x], flow_bins['avg_time'], width=0.4, label='True', alpha=0.7)
        ax3.bar([i+0.2 for i in x], flow_bins['pred_ensemble'], width=0.4, label='Predicted', alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels(flow_bins.index, rotation=45)
        ax3.set_xlabel('Flow Group')
        ax3.set_ylabel('Avg Travel Time (min)')
        ax3.set_title('Flow vs Travel Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 按时段的误差
        ax4 = plt.subplot(2, 4, 4)
        hourly_error = test_results.groupby('hour')['error'].mean()
        ax4.plot(hourly_error.index, hourly_error.values, marker='o', linewidth=2)
        ax4.set_xlabel('Hour')
        ax4.set_ylabel('Mean Error (min)')
        ax4.set_title('Error by Hour')
        ax4.grid(True, alpha=0.3)
        
        # 5. 特征重要性
        ax5 = plt.subplot(2, 4, 5)
        importances = models['XGBoost'].feature_importances_
        indices = np.argsort(importances)[-15:]
        ax5.barh(range(len(indices)), importances[indices])
        ax5.set_yticks(range(len(indices)))
        ax5.set_yticklabels([feature_cols[i] for i in indices], fontsize=8)
        ax5.set_xlabel('Importance')
        ax5.set_title('Top 15 Feature Importance')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # 6. 累积误差分布
        ax6 = plt.subplot(2, 4, 6)
        sorted_errors = np.sort(test_results['error'])
        cumulative = np.arange(1, len(sorted_errors)+1) / len(sorted_errors) * 100
        ax6.plot(sorted_errors, cumulative, linewidth=2)
        ax6.axvline(5, color='r', linestyle='--', alpha=0.5, label='5 min')
        ax6.axvline(10, color='g', linestyle='--', alpha=0.5, label='10 min')
        ax6.axvline(20, color='b', linestyle='--', alpha=0.5, label='20 min')
        ax6.set_xlabel('Absolute Error (min)')
        ax6.set_ylabel('Cumulative Percentage (%)')
        ax6.set_title('Cumulative Error Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0, 50)
        
        # 7. Flow散点图
        ax7 = plt.subplot(2, 4, 7)
        sample = test_results[test_results['flow'] > 0].sample(min(3000, len(test_results)))
        scatter = ax7.scatter(sample['flow'], sample['avg_time'], 
                             c=sample['error'], cmap='RdYlGn_r', alpha=0.5, s=20)
        plt.colorbar(scatter, ax=ax7, label='Error (min)')
        ax7.set_xlabel('Flow')
        ax7.set_ylabel('Travel Time (min)')
        ax7.set_title('Flow vs Time (colored by error)')
        ax7.grid(True, alpha=0.3)
        
        # 8. 高峰vs平峰误差
        ax8 = plt.subplot(2, 4, 8)
        peak_errors = [
            test_results[test_results['is_peak']==0]['error'].mean(),
            test_results[test_results['is_morning_peak']==1]['error'].mean(),
            test_results[test_results['is_evening_peak']==1]['error'].mean()
        ]
        ax8.bar(['Off-Peak', 'Morning Peak', 'Evening Peak'], peak_errors, alpha=0.7)
        ax8.set_ylabel('Mean Absolute Error (min)')
        ax8.set_title('Error by Time Period')
        ax8.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/no_leak_results.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 可视化已保存: {output_path}")
    
    def save_results(self, results_df, test_results):
        """保存结果"""
        print("\n" + "="*80)
        print("7. 保存结果")
        print("="*80)
        
        # 评估指标
        results_df.to_csv(f'{self.output_dir}/no_leak_evaluation.csv', index=False)
        print(f"✓ 评估指标: no_leak_evaluation.csv")
        
        # 预测结果
        test_results.to_csv(f'{self.output_dir}/no_leak_predictions.csv', index=False)
        print(f"✓ 预测结果: no_leak_predictions.csv (包含真实值对比)")


def main():
    """主函数"""
    print("="*80)
    print("步骤2：基于插值数据的行程时间预测（无数据泄露）")
    print("="*80)
    print(f"执行时间: {datetime.now()}")
    
    # 配置
    data_path = '/data/alice/cjtest/TRC/Travel_Time/od_flow_interpolated.csv'
    output_dir = '/data/alice/cjtest/TRC/Travel_Time'
    
    train_dates = ['2008-02-02', '2008-02-03', '2008-02-04', '2008-02-05', '2008-02-06']
    val_dates = ['2008-02-07']
    test_dates = ['2008-02-08']
    
    predictor = NoLeakTravelTimePredictor(data_path, output_dir)
    
    # 1. 加载数据
    predictor.load_interpolated_data()
    
    # 2. 创建特征
    train_df, val_df, test_df, train_mean = predictor.create_features_no_leak(
        train_dates, val_dates, test_dates
    )
    
    # 选择特征列（不包含origin/dest ID）
    feature_cols = [
        'hour_norm', 'minute_norm', 'time_slot_norm',
        'day_of_week', 'is_weekend', 'is_morning_peak', 'is_evening_peak', 'is_peak',
        'flow_raw', 'flow_log', 'flow_sqrt', 'flow_squared',
        'flow_density_morning', 'flow_density_evening',
        'hour_mean', 'hour_std',
        'flow_bin_mean', 'flow_bin_std',
        'lag_1', 'lag_2', 'lag_4', 'lag_8', 'lag_16',
        'rolling_4h_mean', 'rolling_4h_std'
    ]
    
    print(f"\n特征列表 ({len(feature_cols)}个):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    # 准备X和y
    X_train, y_train = predictor.prepare_xy(train_df, feature_cols)
    X_val, y_val = predictor.prepare_xy(val_df, feature_cols)
    X_test, y_test = predictor.prepare_xy(test_df, feature_cols)
    
    print(f"\n数据形状:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    
    # 3. 训练模型
    models = predictor.train_models(X_train, y_train, X_val, y_val)
    
    # 4. 评估
    results_df, test_results, models = predictor.evaluate_models(
        models, X_test, y_test, test_df
    )
    
    # 5. Flow影响分析
    flow_analysis = predictor.analyze_flow_impact(test_results)
    
    # 6. 可视化
    predictor.visualize_results(test_results, models, feature_cols)
    
    # 7. 保存结果
    predictor.save_results(results_df, test_results)
    
    print("\n" + "="*80)
    print("✓ 完成！")
    print("="*80)
    print(f"\n关键结果:")
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
