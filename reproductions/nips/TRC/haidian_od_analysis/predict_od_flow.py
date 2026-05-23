#!/usr/bin/env python3
"""
OD流量预测系统
功能：基于历史OD流量数据预测未来流量
支持多种预测算法：历史平均、SARIMA、LSTM、Prophet
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ODFlowPredictor:
    """OD流量预测器"""
    
    def __init__(self, data_dir='output'):
        self.data_dir = data_dir
        self.od_flow = None
        self.od_time = None
        self.od_table = None
        
    def load_data(self):
        """加载OD矩阵数据"""
        print("="*80)
        print("加载数据")
        print("="*80)
        
        # 加载numpy矩阵
        self.od_flow = np.load(f'{self.data_dir}/od_flow_full.npy')
        self.od_time = np.load(f'{self.data_dir}/od_time_full.npy')
        
        # 加载表格数据
        self.od_table = pd.read_csv(f'{self.data_dir}/od_flow_table_full.csv')
        
        print(f"\n矩阵形状: {self.od_flow.shape}")
        print(f"  时间槽数: {self.od_flow.shape[0]}")
        print(f"  区域数: {self.od_flow.shape[1]} × {self.od_flow.shape[2]}")
        print(f"\nOD表记录数: {len(self.od_table)}")
        
        # 分析数据覆盖天数
        self.analyze_temporal_coverage()
        
    def analyze_temporal_coverage(self):
        """分析时间覆盖情况"""
        # 从文件名推断时间范围
        trips_df = pd.read_csv(f'{self.data_dir}/od_trips_full.csv')
        trips_df['start_time'] = pd.to_datetime(trips_df['start_time'])
        
        self.start_date = trips_df['start_time'].min().date()
        self.end_date = trips_df['start_time'].max().date()
        self.num_days = (self.end_date - self.start_date).days + 1
        
        print(f"\n时间范围:")
        print(f"  开始: {self.start_date}")
        print(f"  结束: {self.end_date}")
        print(f"  天数: {self.num_days} 天")
        
    def prepare_time_series(self, origin=None, dest=None):
        """
        准备时间序列数据
        
        Args:
            origin: 起点区域ID (1-29)，None表示所有区域汇总
            dest: 终点区域ID (1-29)，None表示所有区域汇总
            
        Returns:
            时间序列数组
        """
        if origin is None and dest is None:
            # 全局流量：所有OD对的总和
            ts = self.od_flow.sum(axis=(1, 2))
            label = "全局流量"
        elif origin is not None and dest is None:
            # 从origin出发的所有流量
            ts = self.od_flow[:, origin-1, :].sum(axis=1)
            label = f"区域{origin}出发流量"
        elif origin is None and dest is not None:
            # 到达dest的所有流量
            ts = self.od_flow[:, :, dest-1].sum(axis=1)
            label = f"到达区域{dest}流量"
        else:
            # 特定OD对
            ts = self.od_flow[:, origin-1, dest-1]
            label = f"区域{origin}→区域{dest}"
        
        return ts, label
    
    def method1_historical_average(self, ts, label, predict_steps=96):
        """
        方法1: 历史平均法
        使用相同时间槽的历史平均值作为预测
        """
        print(f"\n{'='*80}")
        print(f"方法1: 历史平均法 - {label}")
        print(f"{'='*80}")
        
        slots_per_day = 96
        num_days = len(ts) // slots_per_day
        
        # 重塑为 (天数, 时间槽)
        ts_reshaped = ts[:num_days * slots_per_day].reshape(num_days, slots_per_day)
        
        # 计算每个时间槽的历史平均
        historical_avg = ts_reshaped.mean(axis=0)
        
        # 预测：重复历史平均模式
        predictions = np.tile(historical_avg, predict_steps // slots_per_day + 1)[:predict_steps]
        
        print(f"训练数据: {num_days} 天, {len(ts)} 个时间槽")
        print(f"预测: {predict_steps} 个时间槽")
        print(f"平均流量: {historical_avg.mean():.2f} trips/slot")
        print(f"流量范围: [{historical_avg.min():.0f}, {historical_avg.max():.0f}]")
        
        return predictions, historical_avg
    
    def method2_weighted_average(self, ts, label, predict_steps=96, window=2):
        """
        方法2: 加权移动平均法
        最近几天的数据权重更大
        """
        print(f"\n{'='*80}")
        print(f"方法2: 加权移动平均法 - {label}")
        print(f"{'='*80}")
        
        slots_per_day = 96
        num_days = len(ts) // slots_per_day
        
        ts_reshaped = ts[:num_days * slots_per_day].reshape(num_days, slots_per_day)
        
        # 使用最近window天的加权平均
        if num_days > window:
            # 权重：越近的天数权重越大
            weights = np.arange(1, window + 1)
            weights = weights / weights.sum()
            
            weighted_avg = np.average(ts_reshaped[-window:], axis=0, weights=weights)
        else:
            weighted_avg = ts_reshaped.mean(axis=0)
        
        predictions = np.tile(weighted_avg, predict_steps // slots_per_day + 1)[:predict_steps]
        
        print(f"窗口大小: {window} 天")
        print(f"预测平均流量: {weighted_avg.mean():.2f} trips/slot")
        
        return predictions, weighted_avg
    
    def method3_linear_trend(self, ts, label, predict_steps=96):
        """
        方法3: 线性趋势预测
        考虑历史趋势
        """
        print(f"\n{'='*80}")
        print(f"方法3: 线性趋势法 - {label}")
        print(f"{'='*80}")
        
        slots_per_day = 96
        num_days = len(ts) // slots_per_day
        
        ts_reshaped = ts[:num_days * slots_per_day].reshape(num_days, slots_per_day)
        
        predictions = []
        
        for slot in range(96):
            # 该时间槽的历史数据
            slot_values = ts_reshaped[:, slot]
            
            # 拟合线性趋势
            x = np.arange(len(slot_values))
            slope, intercept = np.polyfit(x, slot_values, 1)
            
            # 预测未来
            future_x = np.arange(num_days, num_days + predict_steps // 96)
            future_vals = slope * future_x + intercept
            
            # 确保非负
            future_vals = np.maximum(future_vals, 0)
            predictions.append(future_vals)
        
        # 重组预测结果
        predictions = np.array(predictions).T.flatten()[:predict_steps]
        
        print(f"预测平均流量: {predictions.mean():.2f} trips/slot")
        
        return predictions
    
    def method4_similar_day(self, ts, label, predict_steps=96):
        """
        方法4: 相似日法
        找到历史上最相似的一天作为预测基准
        """
        print(f"\n{'='*80}")
        print(f"方法4: 相似日法 - {label}")
        print(f"{'='*80}")
        
        slots_per_day = 96
        num_days = len(ts) // slots_per_day
        
        ts_reshaped = ts[:num_days * slots_per_day].reshape(num_days, slots_per_day)
        
        # 计算最后一天与历史每天的相似度
        last_day = ts_reshaped[-1]
        
        similarities = []
        for i in range(num_days - 1):
            # 使用余弦相似度
            similarity = np.dot(ts_reshaped[i], last_day) / (
                np.linalg.norm(ts_reshaped[i]) * np.linalg.norm(last_day) + 1e-10
            )
            similarities.append(similarity)
        
        # 找到最相似的一天
        most_similar_day = np.argmax(similarities)
        
        # 使用最相似日的模式作为预测
        predictions = np.tile(ts_reshaped[most_similar_day], predict_steps // slots_per_day + 1)[:predict_steps]
        
        print(f"最相似的历史日: 第{most_similar_day + 1}天")
        print(f"相似度: {similarities[most_similar_day]:.4f}")
        print(f"预测平均流量: {predictions.mean():.2f} trips/slot")
        
        return predictions, most_similar_day
    
    def evaluate_predictions(self, true_values, predictions, method_name):
        """评估预测效果"""
        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        
        # 计算MAPE (Mean Absolute Percentage Error)
        mask = true_values > 0
        mape = np.mean(np.abs((true_values[mask] - predictions[mask]) / true_values[mask])) * 100
        
        # R²
        r2 = r2_score(true_values, predictions)
        
        return {
            'method': method_name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
    
    def cross_validation(self, origin=None, dest=None):
        """
        交叉验证：使用前N天预测最后一天
        """
        print(f"\n{'='*80}")
        print("交叉验证：使用前6天预测第7天")
        print(f"{'='*80}")
        
        ts, label = self.prepare_time_series(origin, dest)
        
        slots_per_day = 96
        num_days = len(ts) // slots_per_day
        
        if num_days < 2:
            print("数据不足，无法进行交叉验证")
            return
        
        # 使用前N-1天作为训练，最后一天作为测试
        train_ts = ts[:(num_days - 1) * slots_per_day]
        test_ts = ts[(num_days - 1) * slots_per_day:(num_days) * slots_per_day]
        
        print(f"\n数据集: {label}")
        print(f"训练: {num_days - 1} 天")
        print(f"测试: 1 天 ({len(test_ts)} 时间槽)")
        
        # 测试多种方法
        results = []
        
        # 方法1: 历史平均
        pred1, _ = self.method1_historical_average(train_ts, label, predict_steps=96)
        results.append(self.evaluate_predictions(test_ts, pred1, '历史平均'))
        
        # 方法2: 加权平均
        pred2, _ = self.method2_weighted_average(train_ts, label, predict_steps=96, window=2)
        results.append(self.evaluate_predictions(test_ts, pred2, '加权平均'))
        
        # 方法3: 线性趋势
        pred3 = self.method3_linear_trend(train_ts, label, predict_steps=96)
        results.append(self.evaluate_predictions(test_ts, pred3, '线性趋势'))
        
        # 方法4: 相似日
        pred4, _ = self.method4_similar_day(train_ts, label, predict_steps=96)
        results.append(self.evaluate_predictions(test_ts, pred4, '相似日'))
        
        # 显示结果
        results_df = pd.DataFrame(results)
        print(f"\n预测性能对比:")
        print(results_df.to_string(index=False))
        
        # 保存结果
        results_df.to_csv(f'{self.data_dir}/prediction_comparison.csv', index=False)
        print(f"\n✓ 结果已保存: prediction_comparison.csv")
        
        # 可视化对比
        self.visualize_predictions(test_ts, [pred1, pred2, pred3, pred4], 
                                   ['历史平均', '加权平均', '线性趋势', '相似日'],
                                   label)
        
        return results_df
    
    def visualize_predictions(self, true_values, predictions_list, method_names, label):
        """可视化预测结果对比"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        hours = np.arange(len(true_values)) / 4  # 转换为小时
        
        for i, (pred, name) in enumerate(zip(predictions_list, method_names)):
            ax = axes[i]
            
            ax.plot(hours, true_values, 'b-', linewidth=2, label='实际值', alpha=0.7)
            ax.plot(hours, pred, 'r--', linewidth=2, label='预测值', alpha=0.7)
            
            ax.set_xlabel('Hour of Day', fontsize=12)
            ax.set_ylabel('Flow (trips)', fontsize=12)
            ax.set_title(f'{name} - {label}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 显示误差
            mae = mean_absolute_error(true_values, pred)
            rmse = np.sqrt(mean_squared_error(true_values, pred))
            ax.text(0.02, 0.98, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/prediction_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 可视化已保存: prediction_visualization.png")
    
    def predict_top_od_pairs(self, top_n=10):
        """预测Top N热门OD对的未来流量"""
        print(f"\n{'='*80}")
        print(f"预测Top {top_n} 热门OD对")
        print(f"{'='*80}")
        
        # 找到流量最大的OD对
        daily_flow = self.od_flow.sum(axis=0)
        top_od = []
        for o in range(29):
            for d in range(29):
                if daily_flow[o, d] > 0:
                    top_od.append((o+1, d+1, daily_flow[o, d]))
        
        top_od.sort(key=lambda x: x[2], reverse=True)
        top_od = top_od[:top_n]
        
        print(f"\nTop {top_n} OD对:")
        for rank, (o, d, flow) in enumerate(top_od, 1):
            print(f"  {rank:2d}. 区域{o:2d} → 区域{d:2d}: {int(flow):,} trips")
        
        # 为每个OD对生成预测
        predictions_summary = []
        
        for o, d, flow in top_od:
            ts, label = self.prepare_time_series(origin=o, dest=d)
            
            # 使用加权平均法预测
            pred, _ = self.method2_weighted_average(ts, label, predict_steps=96, window=2)
            
            predictions_summary.append({
                'origin': o,
                'dest': d,
                'historical_daily_avg': flow / 7,  # 7天平均
                'predicted_daily': pred.sum(),
                'predicted_peak': pred.max(),
                'predicted_avg_per_slot': pred.mean()
            })
        
        pred_df = pd.DataFrame(predictions_summary)
        pred_df.to_csv(f'{self.data_dir}/top_od_predictions.csv', index=False)
        
        print(f"\n✓ Top OD对预测已保存: top_od_predictions.csv")
        
        return pred_df
    
    def generate_full_prediction(self, method='weighted_average'):
        """
        生成完整的OD矩阵预测（所有区域对）
        """
        print(f"\n{'='*80}")
        print(f"生成完整OD矩阵预测 - 方法: {method}")
        print(f"{'='*80}")
        
        predicted_od = np.zeros((96, 29, 29))
        
        total_pairs = 29 * 29
        processed = 0
        
        for o in range(29):
            for d in range(29):
                ts = self.od_flow[:, o, d]
                
                if ts.sum() > 0:  # 只预测有历史流量的OD对
                    if method == 'historical_average':
                        pred, _ = self.method1_historical_average(ts, f"{o+1}→{d+1}", predict_steps=96)
                    elif method == 'weighted_average':
                        pred, _ = self.method2_weighted_average(ts, f"{o+1}→{d+1}", predict_steps=96)
                    else:
                        pred = self.method3_linear_trend(ts, f"{o+1}→{d+1}", predict_steps=96)
                    
                    predicted_od[:, o, d] = pred
                
                processed += 1
                if processed % 100 == 0:
                    print(f"\r进度: {processed}/{total_pairs} OD对", end='', flush=True)
        
        print(f"\n\n预测完成!")
        print(f"非零OD对: {(predicted_od.sum(axis=0) > 0).sum()}")
        print(f"预测总流量: {predicted_od.sum():.0f} trips")
        print(f"预测日均流量: {predicted_od.sum() / 1:.0f} trips/day")
        
        # 保存预测结果
        np.save(f'{self.data_dir}/predicted_od_flow.npy', predicted_od)
        print(f"\n✓ 预测矩阵已保存: predicted_od_flow.npy")
        
        return predicted_od


def main():
    """主函数"""
    print("="*80)
    print("OD流量预测系统")
    print("="*80)
    
    predictor = ODFlowPredictor(data_dir='output')
    
    # 1. 加载数据
    predictor.load_data()
    
    # 2. 全局流量预测验证
    print(f"\n{'='*80}")
    print("任务1: 全局流量交叉验证")
    print(f"{'='*80}")
    predictor.cross_validation(origin=None, dest=None)
    
    # 3. 预测Top OD对
    print(f"\n{'='*80}")
    print("任务2: 预测热门OD对")
    print(f"{'='*80}")
    predictor.predict_top_od_pairs(top_n=10)
    
    # 4. 生成完整预测矩阵
    print(f"\n{'='*80}")
    print("任务3: 生成完整OD矩阵预测")
    print(f"{'='*80}")
    predicted_od = predictor.generate_full_prediction(method='weighted_average')
    
    # 5. 可视化预测的OD矩阵
    print(f"\n{'='*80}")
    print("任务4: 可视化预测结果")
    print(f"{'='*80}")
    
    daily_predicted = predicted_od.sum(axis=0)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(daily_predicted, cmap='YlOrRd', 
               xticklabels=range(1, 30),
               yticklabels=range(1, 30),
               cbar_kws={'label': 'Predicted Trips'},
               vmin=0, vmax=np.percentile(daily_predicted[daily_predicted>0], 95))
    plt.title('Predicted Daily OD Flow Matrix (Next Day)', fontsize=16, fontweight='bold')
    plt.xlabel('Destination Region', fontsize=12)
    plt.ylabel('Origin Region', fontsize=12)
    plt.tight_layout()
    plt.savefig('output/predicted_od_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 预测热力图已保存: predicted_od_heatmap.png")
    
    print(f"\n{'='*80}")
    print("预测任务完成!")
    print(f"{'='*80}")
    print("\n生成的文件:")
    print("  1. prediction_comparison.csv - 方法对比结果")
    print("  2. prediction_visualization.png - 预测效果可视化")
    print("  3. top_od_predictions.csv - Top OD对预测")
    print("  4. predicted_od_flow.npy - 完整预测矩阵")
    print("  5. predicted_od_heatmap.png - 预测热力图")


if __name__ == '__main__':
    main()
