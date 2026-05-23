"""
行程时间预测模型
==================

数据集划分：
- 训练集：2008-02-02 ~ 2008-02-06
- 验证集：2008-02-07
- 测试集：2008-02-08

预测目标：avg_time (平均出行时间，单位：分钟)

特征说明：
1. 时间特征：hour, minute, time_slot, day_of_week, is_weekend
2. 空间特征：origin, dest, od_pair
3. 流量特征：flow (出行流量)
4. 历史特征：historical travel time patterns

模型：使用XGBoost和LightGBM集成模型
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TravelTimePrediction:
    """行程时间预测类"""
    
    def __init__(self, data_path, output_dir):
        """
        初始化
        
        参数:
            data_path: 数据文件路径
            output_dir: 输出目录
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.models = {}
        self.feature_importance = {}
        
    def load_data(self):
        """加载数据"""
        print("=" * 60)
        print("1. 加载数据")
        print("=" * 60)
        
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"数据形状: {df.shape}")
        print(f"日期范围: {df['date'].min()} 到 {df['date'].max()}")
        print(f"\n数据预览:\n{df.head()}")
        print(f"\n数据统计:\n{df.describe()}")
        
        return df
    
    def feature_engineering(self, df):
        """特征工程"""
        print("\n" + "=" * 60)
        print("2. 特征工程")
        print("=" * 60)
        
        df = df.copy()
        
        # 时间特征
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=周一, 6=周日
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 时段特征（早高峰、晚高峰、平时）
        df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['is_peak_hour'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)
        
        # OD对特征
        df['od_pair'] = df['origin'].astype(str) + '_' + df['dest'].astype(str)
        
        # 流量特征（对数变换，避免极端值影响）
        df['log_flow'] = np.log1p(df['flow'])
        
        # 历史统计特征（基于训练集计算）
        print("\n创建的新特征:")
        print("- day_of_week: 星期几 (0=周一, 6=周日)")
        print("- is_weekend: 是否周末")
        print("- is_morning_peak: 是否早高峰 (7-9点)")
        print("- is_evening_peak: 是否晚高峰 (17-19点)")
        print("- is_peak_hour: 是否高峰时段")
        print("- od_pair: OD对标识")
        print("- log_flow: 流量对数变换")
        
        return df
    
    def add_historical_features(self, train_df, val_df, test_df):
        """添加历史统计特征"""
        print("\n" + "=" * 60)
        print("3. 添加历史统计特征")
        print("=" * 60)
        
        # 基于OD对的历史平均时间
        od_avg_time = train_df.groupby('od_pair')['avg_time'].agg(['mean', 'std', 'median']).reset_index()
        od_avg_time.columns = ['od_pair', 'od_hist_mean', 'od_hist_std', 'od_hist_median']
        od_avg_time['od_hist_std'].fillna(0, inplace=True)
        
        # 基于时段的历史平均时间
        hour_avg_time = train_df.groupby('hour')['avg_time'].agg(['mean', 'std']).reset_index()
        hour_avg_time.columns = ['hour', 'hour_hist_mean', 'hour_hist_std']
        hour_avg_time['hour_hist_std'].fillna(0, inplace=True)
        
        # 基于起点的历史平均时间
        origin_avg_time = train_df.groupby('origin')['avg_time'].agg(['mean', 'std']).reset_index()
        origin_avg_time.columns = ['origin', 'origin_hist_mean', 'origin_hist_std']
        origin_avg_time['origin_hist_std'].fillna(0, inplace=True)
        
        # 基于终点的历史平均时间
        dest_avg_time = train_df.groupby('dest')['avg_time'].agg(['mean', 'std']).reset_index()
        dest_avg_time.columns = ['dest', 'dest_hist_mean', 'dest_hist_std']
        dest_avg_time['dest_hist_std'].fillna(0, inplace=True)
        
        # 合并历史特征
        train_df = train_df.merge(od_avg_time, on='od_pair', how='left')
        train_df = train_df.merge(hour_avg_time, on='hour', how='left')
        train_df = train_df.merge(origin_avg_time, on='origin', how='left')
        train_df = train_df.merge(dest_avg_time, on='dest', how='left')
        
        val_df = val_df.merge(od_avg_time, on='od_pair', how='left')
        val_df = val_df.merge(hour_avg_time, on='hour', how='left')
        val_df = val_df.merge(origin_avg_time, on='origin', how='left')
        val_df = val_df.merge(dest_avg_time, on='dest', how='left')
        
        test_df = test_df.merge(od_avg_time, on='od_pair', how='left')
        test_df = test_df.merge(hour_avg_time, on='hour', how='left')
        test_df = test_df.merge(origin_avg_time, on='origin', how='left')
        test_df = test_df.merge(dest_avg_time, on='dest', how='left')
        
        # 填充缺失值（验证集和测试集中可能有新的OD对）
        train_mean = train_df['avg_time'].mean()
        train_median = train_df['avg_time'].median()
        
        for df in [train_df, val_df, test_df]:
            df['od_hist_mean'].fillna(train_mean, inplace=True)
            df['od_hist_std'].fillna(0, inplace=True)
            df['od_hist_median'].fillna(train_median, inplace=True)
            df['hour_hist_mean'].fillna(train_mean, inplace=True)
            df['hour_hist_std'].fillna(0, inplace=True)
            df['origin_hist_mean'].fillna(train_mean, inplace=True)
            df['origin_hist_std'].fillna(0, inplace=True)
            df['dest_hist_mean'].fillna(train_mean, inplace=True)
            df['dest_hist_std'].fillna(0, inplace=True)
        
        print("添加的历史特征:")
        print("- OD对历史统计: 均值、标准差、中位数")
        print("- 小时历史统计: 均值、标准差")
        print("- 起点历史统计: 均值、标准差")
        print("- 终点历史统计: 均值、标准差")
        
        return train_df, val_df, test_df
    
    def split_data(self, df):
        """划分数据集"""
        print("\n" + "=" * 60)
        print("4. 划分数据集")
        print("=" * 60)
        
        train_df = df[(df['date'] >= '2008-02-02') & (df['date'] <= '2008-02-06')].copy()
        val_df = df[df['date'] == '2008-02-07'].copy()
        test_df = df[df['date'] == '2008-02-08'].copy()
        
        print(f"训练集: {train_df['date'].min()} ~ {train_df['date'].max()}, 样本数: {len(train_df)}")
        print(f"验证集: {val_df['date'].unique()[0]}, 样本数: {len(val_df)}")
        print(f"测试集: {test_df['date'].unique()[0]}, 样本数: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def prepare_features(self, train_df, val_df, test_df):
        """准备特征和标签"""
        print("\n" + "=" * 60)
        print("5. 准备特征和标签")
        print("=" * 60)
        
        # 选择特征列
        feature_cols = [
            'hour', 'minute', 'time_slot', 'day_of_week', 'is_weekend',
            'is_morning_peak', 'is_evening_peak', 'is_peak_hour',
            'origin', 'dest', 'flow', 'log_flow',
            'od_hist_mean', 'od_hist_std', 'od_hist_median',
            'hour_hist_mean', 'hour_hist_std',
            'origin_hist_mean', 'origin_hist_std',
            'dest_hist_mean', 'dest_hist_std'
        ]
        
        target_col = 'avg_time'
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        print(f"\n特征数量: {len(feature_cols)}")
        print(f"特征列表:\n{feature_cols}")
        print(f"\n训练集特征形状: {X_train.shape}")
        print(f"验证集特征形状: {X_val.shape}")
        print(f"测试集特征形状: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """训练XGBoost模型"""
        print("\n" + "=" * 60)
        print("6. 训练XGBoost模型")
        print("=" * 60)
        
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = xgb.XGBRegressor(**params, eval_metric='rmse')
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # 验证集预测
        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)
        
        print(f"\n验证集性能:")
        print(f"  MAE: {val_mae:.4f} 分钟")
        print(f"  RMSE: {val_rmse:.4f} 分钟")
        print(f"  R²: {val_r2:.4f}")
        
        self.models['xgboost'] = model
        self.feature_importance['xgboost'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """训练LightGBM模型"""
        print("\n" + "=" * 60)
        print("7. 训练LightGBM模型")
        print("=" * 60)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 64,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # 验证集预测
        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)
        
        print(f"\n验证集性能:")
        print(f"  MAE: {val_mae:.4f} 分钟")
        print(f"  RMSE: {val_rmse:.4f} 分钟")
        print(f"  R²: {val_r2:.4f}")
        
        self.models['lightgbm'] = model
        self.feature_importance['lightgbm'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model
    
    def ensemble_predict(self, X):
        """集成预测（XGBoost和LightGBM的平均）"""
        xgb_pred = self.models['xgboost'].predict(X)
        lgb_pred = self.models['lightgbm'].predict(X)
        return (xgb_pred + lgb_pred) / 2
    
    def evaluate(self, X_test, y_test, test_df):
        """评估模型"""
        print("\n" + "=" * 60)
        print("8. 测试集评估")
        print("=" * 60)
        
        # 单模型预测
        xgb_pred = self.models['xgboost'].predict(X_test)
        lgb_pred = self.models['lightgbm'].predict(X_test)
        
        # 集成预测
        ensemble_pred = (xgb_pred + lgb_pred) / 2
        
        # 评估指标
        models_eval = {
            'XGBoost': xgb_pred,
            'LightGBM': lgb_pred,
            'Ensemble': ensemble_pred
        }
        
        results = []
        for model_name, pred in models_eval.items():
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)
            mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
            
            results.append({
                'Model': model_name,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape
            })
            
            print(f"\n{model_name}:")
            print(f"  MAE: {mae:.4f} 分钟")
            print(f"  RMSE: {rmse:.4f} 分钟")
            print(f"  R²: {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
        
        results_df = pd.DataFrame(results)
        
        # 保存预测结果
        test_results = test_df.copy()
        test_results['predicted_time_xgb'] = xgb_pred
        test_results['predicted_time_lgb'] = lgb_pred
        test_results['predicted_time_ensemble'] = ensemble_pred
        test_results['absolute_error'] = np.abs(y_test - ensemble_pred)
        test_results['relative_error'] = test_results['absolute_error'] / y_test
        
        return results_df, test_results
    
    def plot_results(self, test_results, results_df):
        """绘制结果图表"""
        print("\n" + "=" * 60)
        print("9. 绘制结果图表")
        print("=" * 60)
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 真实值 vs 预测值散点图
        ax = axes[0, 0]
        ax.scatter(test_results['avg_time'], test_results['predicted_time_ensemble'], 
                   alpha=0.5, s=10)
        ax.plot([test_results['avg_time'].min(), test_results['avg_time'].max()],
                [test_results['avg_time'].min(), test_results['avg_time'].max()],
                'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Travel Time (min)')
        ax.set_ylabel('Predicted Travel Time (min)')
        ax.set_title('Actual vs Predicted Travel Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 误差分布直方图
        ax = axes[0, 1]
        errors = test_results['avg_time'] - test_results['predicted_time_ensemble']
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Prediction Error (min)')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.grid(True, alpha=0.3)
        
        # 3. 模型比较
        ax = axes[0, 2]
        models = results_df['Model'].values
        mae_values = results_df['MAE'].values
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, mae_values, alpha=0.7)
        ax.set_xlabel('Model')
        ax.set_ylabel('MAE (min)')
        ax.set_title('Model Comparison (MAE)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=15)
        ax.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(mae_values):
            ax.text(i, v + 0.2, f'{v:.2f}', ha='center', va='bottom')
        
        # 4. 按小时的误差分析
        ax = axes[1, 0]
        hourly_error = test_results.groupby('hour')['absolute_error'].mean()
        ax.plot(hourly_error.index, hourly_error.values, marker='o', linewidth=2)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Mean Absolute Error (min)')
        ax.set_title('Prediction Error by Hour')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 2))
        
        # 5. 特征重要性（XGBoost）
        ax = axes[1, 1]
        top_features = self.feature_importance['xgboost'].head(10)
        ax.barh(range(len(top_features)), top_features['importance'].values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importance (XGBoost)')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 6. 相对误差分布
        ax = axes[1, 2]
        relative_errors = test_results['relative_error'] * 100
        ax.hist(relative_errors.clip(upper=100), bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Relative Error (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Relative Error Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f'{self.output_dir}/travel_time_prediction_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {plot_path}")
        plt.close()
        
    def save_results(self, results_df, test_results):
        """保存结果"""
        print("\n" + "=" * 60)
        print("10. 保存结果")
        print("=" * 60)
        
        # 保存模型评估结果
        eval_path = f'{self.output_dir}/model_evaluation.csv'
        results_df.to_csv(eval_path, index=False)
        print(f"模型评估结果已保存: {eval_path}")
        
        # 保存测试集预测结果
        test_path = f'{self.output_dir}/test_predictions.csv'
        test_results.to_csv(test_path, index=False)
        print(f"测试集预测结果已保存: {test_path}")
        
        # 保存特征重要性
        for model_name, fi in self.feature_importance.items():
            fi_path = f'{self.output_dir}/feature_importance_{model_name}.csv'
            fi.to_csv(fi_path, index=False)
            print(f"{model_name}特征重要性已保存: {fi_path}")
        
        # 保存模型摘要
        summary_path = f'{self.output_dir}/model_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("行程时间预测模型总结\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. 数据集划分\n")
            f.write(f"   训练集: 2008-02-02 ~ 2008-02-06\n")
            f.write(f"   验证集: 2008-02-07\n")
            f.write(f"   测试集: 2008-02-08\n\n")
            
            f.write("2. 输入特征（共21个）\n")
            f.write("   时间特征:\n")
            f.write("     - hour: 小时 (0-23)\n")
            f.write("     - minute: 分钟 (0-59)\n")
            f.write("     - time_slot: 时间槽\n")
            f.write("     - day_of_week: 星期几 (0=周一, 6=周日)\n")
            f.write("     - is_weekend: 是否周末\n")
            f.write("     - is_morning_peak: 是否早高峰 (7-9点)\n")
            f.write("     - is_evening_peak: 是否晚高峰 (17-19点)\n")
            f.write("     - is_peak_hour: 是否高峰时段\n\n")
            
            f.write("   空间特征:\n")
            f.write("     - origin: 起点区域ID\n")
            f.write("     - dest: 终点区域ID\n\n")
            
            f.write("   流量特征:\n")
            f.write("     - flow: 出行流量\n")
            f.write("     - log_flow: 流量对数变换\n\n")
            
            f.write("   历史统计特征:\n")
            f.write("     - od_hist_mean: OD对历史平均时间\n")
            f.write("     - od_hist_std: OD对历史标准差\n")
            f.write("     - od_hist_median: OD对历史中位数\n")
            f.write("     - hour_hist_mean: 小时历史平均时间\n")
            f.write("     - hour_hist_std: 小时历史标准差\n")
            f.write("     - origin_hist_mean: 起点历史平均时间\n")
            f.write("     - origin_hist_std: 起点历史标准差\n")
            f.write("     - dest_hist_mean: 终点历史平均时间\n")
            f.write("     - dest_hist_std: 终点历史标准差\n\n")
            
            f.write("3. 输出\n")
            f.write("   预测目标: avg_time (平均出行时间，单位：分钟)\n\n")
            
            f.write("4. 模型\n")
            f.write("   - XGBoost回归模型\n")
            f.write("   - LightGBM回归模型\n")
            f.write("   - 集成模型（两个模型的平均值）\n\n")
            
            f.write("5. 预测方法\n")
            f.write("   基于机器学习的时序回归预测:\n")
            f.write("   (1) 利用历史OD流量数据训练模型\n")
            f.write("   (2) 提取时间、空间、流量和历史统计特征\n")
            f.write("   (3) 使用梯度提升树模型学习特征与出行时间的关系\n")
            f.write("   (4) 对新数据进行预测\n\n")
            
            f.write("6. 测试集性能\n")
            f.write(results_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("7. Top 10 重要特征 (XGBoost)\n")
            f.write(self.feature_importance['xgboost'].head(10).to_string(index=False))
            f.write("\n\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"模型摘要已保存: {summary_path}")
        
    def run(self):
        """运行完整流程"""
        print("\n" + "=" * 80)
        print("行程时间预测模型")
        print("=" * 80)
        
        # 1. 加载数据
        df = self.load_data()
        
        # 2. 特征工程
        df = self.feature_engineering(df)
        
        # 3. 划分数据集
        train_df, val_df, test_df = self.split_data(df)
        
        # 4. 添加历史统计特征
        train_df, val_df, test_df = self.add_historical_features(train_df, val_df, test_df)
        
        # 5. 准备特征和标签
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = \
            self.prepare_features(train_df, val_df, test_df)
        
        # 6. 训练XGBoost
        self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # 7. 训练LightGBM
        self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        # 8. 评估
        results_df, test_results = self.evaluate(X_test, y_test, test_df)
        
        # 9. 绘制结果
        self.plot_results(test_results, results_df)
        
        # 10. 保存结果
        self.save_results(results_df, test_results)
        
        print("\n" + "=" * 80)
        print("完成！所有结果已保存到:", self.output_dir)
        print("=" * 80)


if __name__ == '__main__':
    # 设置路径
    data_path = '/data/alice/cjtest/TRC/haidian_od_analysis/output/od_flow_temporal.csv'
    output_dir = '/data/alice/cjtest/TRC/Travel_Time'
    
    # 创建预测器并运行
    predictor = TravelTimePrediction(data_path, output_dir)
    predictor.run()
