#!/usr/bin/env python3
"""
分步执行脚本 - 可以单独执行每个步骤
"""

import os
import sys

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import HaidianODAnalysisPipeline

def main():
    """分步执行主程序"""
    
    # 配置参数
    config = {
        # 路径配置
        'shapefile_path': '/data/alice/cjtest/TRC/海淀区边界_110108_Shapefile_(poi86.com)/110108.shp',
        'region_mapping_path': '/data/alice/cjtest/TRC/haidian_od_analysis/config/region_mapping.csv',
        'trajectory_path': '/data/alice/cjtest/TRC/all_taxi_data.csv',
        'output_dir': '/data/alice/cjtest/TRC/haidian_od_analysis/output',
        
        # 参数配置
        'num_regions': 29,  # 22街道 + 7镇
        'interval_minutes': 15,  # 15分钟时间粒度
        'trip_time_threshold': 30,  # 出行时间阈值（分钟）
    }
    
    # 创建流程实例
    pipeline = HaidianODAnalysisPipeline(config)
    
    print("\n" + "=" * 80)
    print("分步执行模式")
    print("=" * 80)
    
    while True:
        print("\n请选择要执行的步骤:")
        print("  1. 加载区域数据")
        print("  2. 加载轨迹数据")
        print("  3. 空间映射 (轨迹点 -> 区域)")
        print("  4. 时间映射 (时间 -> 15分钟时间槽)")
        print("  5. 提取出行并生成OD矩阵")
        print("  6. 保存最终结果")
        print("  7. 执行所有剩余步骤")
        print("  0. 退出")
        
        choice = input("\n请输入选项 (0-7): ").strip()
        
        if choice == '0':
            print("退出程序")
            break
        elif choice == '1':
            print("\n开始执行步骤1...")
            pipeline.step1_load_and_prepare_regions()
            print("\n步骤1完成！按Enter继续...")
            input()
        elif choice == '2':
            print("\n开始执行步骤2...")
            pipeline.step2_load_and_filter_trajectory()
            print("\n步骤2完成！按Enter继续...")
            input()
        elif choice == '3':
            print("\n开始执行步骤3...")
            pipeline.step3_spatial_mapping()
            print("\n步骤3完成！按Enter继续...")
            input()
        elif choice == '4':
            print("\n开始执行步骤4...")
            pipeline.step4_temporal_mapping()
            print("\n步骤4完成！按Enter继续...")
            input()
        elif choice == '5':
            print("\n开始执行步骤5...")
            print("警告: 这一步可能需要较长时间，请耐心等待...")
            pipeline.step5_extract_trips_and_generate_od()
            print("\n步骤5完成！按Enter继续...")
            input()
        elif choice == '6':
            print("\n开始执行步骤6...")
            pipeline.step6_save_results()
            print("\n步骤6完成！按Enter继续...")
            input()
        elif choice == '7':
            print("\n开始执行所有步骤...")
            pipeline.run()
            break
        else:
            print("无效选项，请重新输入")

if __name__ == "__main__":
    main()
