import pandas as pd

# 基础文件路径
base_path = r"D:\Pycharm\py\code\code_gurobi\results\exps_record_all_parallelexp60625000"

# 创建一个空的DataFrame来存储合并后的结果
combined_df = None

# 遍历11个文件
for i in range(1, 10):
    # 构建完整的文件路径
    file_path = base_path + str(i) + "parallel3.xlsx"
    
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    if combined_df is None:
        # 第一个文件，直接作为基础
        combined_df = df
    else:
        # 从第二个文件开始，只添加数据行（不包含表头）
        combined_df = pd.concat([combined_df, df.iloc[0:]], ignore_index=True)

# 保存合并后的文件
output_path = r"D:\Pycharm\py\code\code_gurobi\results\combined_results1.xlsx"
combined_df.to_excel(output_path, index=False)

print("文件合并完成！")