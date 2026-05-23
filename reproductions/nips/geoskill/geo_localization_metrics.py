"""
地理定位精度 Metrics 公式计算
用于与 coauthor 对比的标准 metrics 实现
"""

import math
from typing import List, Dict, Tuple


class GeoLocalizationMetrics:
    """定位精度相关的所有 metrics 计算"""
    
    # ============== 基础距离计算 ==============
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Haversine 公式：计算地球表面两点间的大圆距离
        
        公式：
        a = sin²(Δφ/2) + cos(φ1) * cos(φ2) * sin²(Δλ/2)
        c = 2 * atan2(√a, √(1−a))
        d = R * c
        
        其中：
        - φ: 纬度 (degrees)
        - λ: 经度 (degrees)  
        - R: 地球半径 ≈ 6371 km
        - d: 距离 (km)
        
        Args:
            lat1, lon1: 真实位置 (Ground Truth)
            lat2, lon2: 预测位置 (Prediction)
            
        Returns:
            距离 (km)
        """
        R = 6371.0  # 地球平均半径 (km)
        
        # 转换为弧度
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine 公式
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance_km = R * c
        
        return distance_km
    
    
    # ============== 单样本精度指标 ==============
    
    @staticmethod
    def accuracy_at_threshold(distance_km: float, threshold_km: float) -> int:
        """
        判断是否在阈值内
        
        公式：
        Acc@X = 1 if d ≤ X else 0
        
        其中：
        - d: 预测与真值的距离
        - X: 距离阈值
        """
        return 1 if distance_km <= threshold_km else 0
    
    
    # ============== 批量样本精度指标 ==============
    
    @staticmethod
    def median_error(distances_km: List[float]) -> float:
        """
        中位数误差
        
        公式：
        Median Error = median({d₁, d₂, ..., dₙ})
        
        定义：
        - 当排序后样本数为奇数时，取中间值
        - 当排序后样本数为偶数时，取中间两个值的平均数
        """
        if not distances_km:
            return None
        sorted_distances = sorted(distances_km)
        n = len(sorted_distances)
        if n % 2 == 1:
            return sorted_distances[n // 2]
        else:
            return (sorted_distances[n // 2 - 1] + sorted_distances[n // 2]) / 2.0
    
    
    @staticmethod
    def accuracy_at_k(distances_km: List[float], threshold_km: float) -> float:
        """
        距离精度 (Accuracy @ k)
        
        公式：
        Acc@Xkm = (count(dᵢ ≤ X)) / N
        
        其中：
        - dᵢ: 第 i 个样本的预测距离错误
        - X: 距离阈值 (km)
        - N: 总样本数
        """
        if not distances_km:
            return 0.0
        correct = sum(1 for d in distances_km if d <= threshold_km)
        return correct / len(distances_km)
    
    
    @staticmethod
    def mean_error(distances_km: List[float]) -> float:
        """
        平均误差距离
        
        公式：
        Mean Error = (Σᵢ dᵢ) / N
        
        其中：
        - dᵢ: 第 i 个样本的距离
        - N: 总样本数
        """
        if not distances_km:
            return None
        return sum(distances_km) / len(distances_km)
    
    
    # ============== 分布统计指标 ==============
    
    @staticmethod
    def coverage_rate(valid_predictions: int, total_samples: int) -> float:
        """
        预测覆盖率
        
        公式：
        Coverage = count(valid predictions) / N
        
        其中：
        - "有效预测" 指成功输出定位坐标的样本
        - N: 总样本数
        """
        if total_samples == 0:
            return 0.0
        return valid_predictions / total_samples
    
    
    @staticmethod
    def standard_deviation(distances_km: List[float]) -> float:
        """
        误差距离的标准差
        
        公式：
        σ = √[(Σ(dᵢ - μ)²) / N]
        
        其中：
        - μ: 平均距离
        - N: 样本数
        """
        if not distances_km or len(distances_km) < 2:
            return None
        mean = sum(distances_km) / len(distances_km)
        variance = sum((d - mean) ** 2 for d in distances_km) / len(distances_km)
        return math.sqrt(variance)
    
    
    # ============== 多阈值评估 ==============
    
    @staticmethod
    def compute_accuracy_across_thresholds(
        distances_km: List[float],
        thresholds_km: List[float] = None
    ) -> Dict[str, float]:
        """
        计算多个阈值下的精度
        
        公式（对每个阈值 X）：
        Acc@Xkm = (Σ 1[dᵢ ≤ X]) / N
        
        默认阈值：
        - 1 km (街道级)
        - 10 km (社区级)
        - 25 km (城市级)
        - 100 km (地区级)
        - 200 km (州级)
        - 750 km (国家级)
        - 2500 km (大洲级)
        """
        if thresholds_km is None:
            thresholds_km = [1, 10, 25, 100, 200, 750, 2500]
        
        results = {}
        for threshold in thresholds_km:
            acc = GeoLocalizationMetrics.accuracy_at_k(distances_km, threshold)
            results[f"Acc@{threshold}km"] = acc
        
        return results
    
    
    @staticmethod
    def comprehensive_metrics(
        distances_km: List[float],
        valid_count: int = None,
        total_count: int = None
    ) -> Dict:
        """
        计算完整的 metrics 汇总报告
        
        包含的指标：
        1. 中位数误差: median(d)
        2. 平均误差: mean(d)
        3. 标准差: std(d)
        4. 多阈值精度: Acc@{1,10,25,100,200,750,2500}km
        5. 预测覆盖率 (可选): valid_count / total_count
        """
        if not distances_km:
            return {}
        
        # 如果未提供有效样本计数，则假设所有样本都有效
        if valid_count is None:
            valid_count = len(distances_km)
        if total_count is None:
            total_count = len(distances_km)
        
        metrics = {
            "total_samples": total_count,
            "valid_predictions": valid_count,
            "coverage_rate": GeoLocalizationMetrics.coverage_rate(valid_count, total_count),
            "median_error_km": GeoLocalizationMetrics.median_error(distances_km),
            "mean_error_km": GeoLocalizationMetrics.mean_error(distances_km),
            "std_error_km": GeoLocalizationMetrics.standard_deviation(distances_km),
        }
        
        # 添加多阈值精度
        threshold_accuracies = GeoLocalizationMetrics.compute_accuracy_across_thresholds(distances_km)
        metrics.update(threshold_accuracies)
        
        return metrics


# ============== 使用示例 ==============

if __name__ == "__main__":
    # 示例数据：预测定位距离真值的距离 (km)
    example_distances = [0.5, 2.3, 5.1, 8.7, 15.4, 25.0, 50.2, 100.5, 150.0, 2000.0]
    
    print("=" * 60)
    print("地理定位精度 Metrics 计算示例")
    print("=" * 60)
    
    # 1. 基础指标
    print("\n[1] 基础指标:")
    print(f"中位数误差: {GeoLocalizationMetrics.median_error(example_distances):.2f} km")
    print(f"平均误差: {GeoLocalizationMetrics.mean_error(example_distances):.2f} km")
    print(f"标准差: {GeoLocalizationMetrics.standard_deviation(example_distances):.2f} km")
    
    # 2. 多阈值精度
    print("\n[2] 多阈值精度 (Accuracy@Xkm):")
    accuracies = GeoLocalizationMetrics.compute_accuracy_across_thresholds(example_distances)
    for threshold_name, accuracy in accuracies.items():
        print(f"{threshold_name}: {accuracy:.1%}")
    
    # 3. 完整报告
    print("\n[3] 完整 Metrics 报告:")
    comprehensive = GeoLocalizationMetrics.comprehensive_metrics(
        example_distances,
        valid_count=10,
        total_count=10
    )
    for key, value in comprehensive.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # 4. Haversine 距离计算示例
    print("\n[4] Haversine 距离计算示例:")
    # 北京坐标
    dist = GeoLocalizationMetrics.haversine_distance(39.9042, 116.4074, 40.7128, -74.0060)
    print(f"北京到纽约的距离: {dist:.2f} km")
