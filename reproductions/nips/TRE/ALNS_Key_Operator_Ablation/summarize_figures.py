import pandas as pd
import os
from pathlib import Path

# 设置Figures目录
figures_dir = r"E:\TRE\ALNS_Key_Operator_Ablation\Figures"

# 获取所有实验文件夹
experiment_folders = [f for f in os.listdir(figures_dir) 
                     if os.path.isdir(os.path.join(figures_dir, f)) and f.startswith('experiment')]

print(f"找到 {len(experiment_folders)} 个实验文件夹")

# 存储所有结果
all_results = []

# 遍历每个实验文件夹
for exp_folder in experiment_folders:
    exp_path = os.path.join(figures_dir, exp_folder)
    
    # 查找percentage子文件夹
    sub_folders = [f for f in os.listdir(exp_path) 
                   if os.path.isdir(os.path.join(exp_path, f)) and 'percentage' in f]
    
    for sub_folder in sub_folders:
        sub_path = os.path.join(exp_path, sub_folder)
        
        # 查找obj_record开头的Excel文件
        files = [f for f in os.listdir(sub_path) 
                if f.startswith('obj_record') and f.endswith('.xlsx')]
        
        for excel_file in files:
            file_path = os.path.join(sub_path, excel_file)
            print(f"正在读取: {exp_folder}/{sub_folder}/{excel_file}")
            
            try:
                # 读取Excel文件
                df = pd.read_excel(file_path)
                
                # 只提取最后一行
                if len(df) >= 1:
                    last_row = df.iloc[-1:].copy()  # 获取最后一行，保持DataFrame格式
                    
                    # 添加标识列
                    last_row.insert(0, '实验编号', exp_folder)
                    last_row.insert(1, '子文件夹', sub_folder)
                    last_row.insert(2, '文件名', excel_file)
                    
                    all_results.append(last_row)
                    print(f"  成功读取最后一行数据 (共 {len(df)} 行)")
                else:
                    print(f"  警告: {excel_file} 没有数据")
            
            except Exception as e:
                print(f"  错误: 无法读取 {excel_file}: {str(e)}")

# 合并所有结果
if all_results:
    summary_df = pd.concat(all_results, ignore_index=True)
    
    # 保存到汇总文件
    output_path = os.path.join(figures_dir, 'overall.xlsx')
    summary_df.to_excel(output_path, index=False)
    
    print(f"\n成功! 已将 {len(all_results)} 个文件的结果汇总到: {output_path}")
    print(f"汇总表包含 {len(summary_df)} 行数据")
    print(f"\n列名: {list(summary_df.columns)}")
else:
    print("错误: 没有成功读取任何数据")
