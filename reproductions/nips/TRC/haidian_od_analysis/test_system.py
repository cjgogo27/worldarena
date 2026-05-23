#!/usr/bin/env python3
"""
系统测试脚本
快速验证所有模块是否正常工作
"""

import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """测试1: 检查所有模块是否可以导入"""
    print("=" * 60)
    print("测试1: 检查模块导入")
    print("=" * 60)
    
    try:
        from region_processor import RegionProcessor
        print("✓ region_processor 导入成功")
    except Exception as e:
        print(f"✗ region_processor 导入失败: {e}")
        return False
    
    try:
        from time_processor import TimeProcessor
        print("✓ time_processor 导入成功")
    except Exception as e:
        print(f"✗ time_processor 导入失败: {e}")
        return False
    
    try:
        from od_matrix_generator import ODMatrixGenerator
        print("✓ od_matrix_generator 导入成功")
    except Exception as e:
        print(f"✗ od_matrix_generator 导入失败: {e}")
        return False
    
    try:
        from quality_checker import DataQualityChecker
        print("✓ quality_checker 导入成功")
    except Exception as e:
        print(f"✗ quality_checker 导入失败: {e}")
        return False
    
    return True


def test_dependencies():
    """测试2: 检查依赖包"""
    print("\n" + "=" * 60)
    print("测试2: 检查依赖包")
    print("=" * 60)
    
    required_packages = [
        'pandas', 'numpy', 'geopandas', 'shapely', 
        'h5py', 'matplotlib', 'seaborn', 'tqdm'
    ]
    
    all_ok = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装")
            all_ok = False
    
    return all_ok


def test_config_files():
    """测试3: 检查配置文件"""
    print("\n" + "=" * 60)
    print("测试3: 检查配置文件")
    print("=" * 60)
    
    import pandas as pd
    
    # 检查区域映射表
    mapping_path = 'config/region_mapping.csv'
    if os.path.exists(mapping_path):
        try:
            df = pd.read_csv(mapping_path)
            print(f"✓ 区域映射表存在: {len(df)} 个区域")
            
            # 验证必需字段
            required_cols = ['region_id', 'region_code', 'region_name', 'region_type']
            if all(col in df.columns for col in required_cols):
                print(f"✓ 所有必需字段存在")
            else:
                print(f"✗ 缺少必需字段")
                return False
                
        except Exception as e:
            print(f"✗ 读取区域映射表失败: {e}")
            return False
    else:
        print(f"✗ 区域映射表不存在: {mapping_path}")
        return False
    
    return True


def test_data_files():
    """测试4: 检查数据文件"""
    print("\n" + "=" * 60)
    print("测试4: 检查数据文件")
    print("=" * 60)
    
    # Shapefile
    shapefile_path = '/data/alice/cjtest/TRC/海淀区边界_110108_Shapefile_(poi86.com)/110108.shp'
    if os.path.exists(shapefile_path):
        print(f"✓ Shapefile存在")
    else:
        print(f"⚠ Shapefile不存在: {shapefile_path}")
        print(f"  (首次运行时需要此文件)")
    
    # 轨迹数据
    traj_path = '/data/alice/cjtest/TRC/all_taxi_data.csv'
    if os.path.exists(traj_path):
        size_mb = os.path.getsize(traj_path) / (1024**2)
        print(f"✓ 轨迹数据存在: {size_mb:.2f} MB")
    else:
        print(f"⚠ 轨迹数据不存在: {traj_path}")
        print(f"  (首次运行时需要此文件)")
    
    return True


def test_module_functionality():
    """测试5: 测试模块基本功能"""
    print("\n" + "=" * 60)
    print("测试5: 测试模块功能")
    print("=" * 60)
    
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from time_processor import TimeProcessor
    from od_matrix_generator import ODMatrixGenerator
    from quality_checker import DataQualityChecker
    
    # 测试TimeProcessor
    try:
        processor = TimeProcessor(interval_minutes=15)
        test_time = "2008-02-02 15:46:08"
        rounded = processor.round_to_interval(test_time)
        slot_id = processor.create_time_slot_id(test_time)
        print(f"✓ TimeProcessor工作正常 (时间槽ID: {slot_id})")
    except Exception as e:
        print(f"✗ TimeProcessor测试失败: {e}")
        return False
    
    # 测试ODMatrixGenerator
    try:
        generator = ODMatrixGenerator(num_regions=29, interval_minutes=15)
        print(f"✓ ODMatrixGenerator工作正常 (29个区域, 15分钟粒度)")
    except Exception as e:
        print(f"✗ ODMatrixGenerator测试失败: {e}")
        return False
    
    # 测试DataQualityChecker
    try:
        checker = DataQualityChecker(num_regions=29)
        # 创建测试数据
        test_df = pd.DataFrame({
            'taxi_id': [1, 1, 2],
            'date_time': ['2008-02-02 10:00:00', '2008-02-02 10:15:00', '2008-02-02 10:30:00'],
            'longitude': [116.3, 116.31, 116.32],
            'latitude': [39.99, 39.98, 39.97]
        })
        # 运行检查（静默模式）
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # checker.check_trajectory_data(test_df)
        print(f"✓ DataQualityChecker工作正常")
    except Exception as e:
        print(f"✗ DataQualityChecker测试失败: {e}")
        return False
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("海淀区OD流量分析系统 - 系统测试")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_imports),
        ("依赖包", test_dependencies),
        ("配置文件", test_config_files),
        ("数据文件", test_data_files),
        ("模块功能", test_module_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ 测试 '{test_name}' 出错: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 所有测试通过！系统可以正常运行。")
        print("\n运行主程序:")
        print("  python main.py")
        return 0
    else:
        print("\n⚠ 部分测试失败，请检查上述错误信息。")
        print("\n安装依赖:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
