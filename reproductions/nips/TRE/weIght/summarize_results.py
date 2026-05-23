import pandas as pd
import os
from pathlib import Path

# 设置结果目录
results_dir = r"results"

# 获取所有Excel文件(排除汇总文件本身)，保持原始文件系统顺序
excel_files = [f for f in os.listdir(results_dir) 
               if f.endswith('.xlsx') and f != '结果汇总.xlsx']

# 不排序，保持文件系统的原始顺序

print(f"找到 {len(excel_files)} 个Excel文件")

# 存储所有结果
all_results = []

# 读取每个Excel文件的第二行(即第一行数据,不包括表头)
for excel_file in excel_files:
    file_path = os.path.join(results_dir, excel_file)
    print(f"正在读取: {excel_file}")
    
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 获取第一行数据(索引为0,表头后的第一行,即Excel中的第二行)
        if len(df) >= 1:
            first_data_row = df.iloc[0:1].copy()  # 获取第一行数据,保持DataFrame格式
            # 添加文件名作为标识
            first_data_row.insert(0, '文件名', excel_file)
            all_results.append(first_data_row)
            print(f"  成功读取数据")
        else:
            print(f"  警告: {excel_file} 没有数据")
    
    except Exception as e:
        print(f"  错误: 无法读取 {excel_file}: {str(e)}")

# 合并所有结果
if all_results:
    summary_df = pd.concat(all_results, ignore_index=True)
    
    # 保存到汇总文件
    output_path = os.path.join(results_dir, '结果汇总.xlsx')
    summary_df.to_excel(output_path, index=False)
    
    print(f"\n成功! 已将 {len(all_results)} 个文件的结果汇总到: {output_path}")
    print(f"汇总表包含 {len(summary_df)} 行数据")
    print(f"\n列名: {list(summary_df.columns)}")
else:
    print("错误: 没有成功读取任何数据")
