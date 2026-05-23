import pandas as pd

# 创建两个空的DataFrame来存储所有结果的合并
final_combined_df1 = None
final_combined_df2 = None

# 遍历不同的基础路径
for base_num in range(1, 7):
    # 构建基础路径
    base_folder = rf"D:\Pycharm\py\code\codes_ALNS-{base_num}"
    base_path1 = rf"{base_folder}\单位成本和载具空间利用率\结果_ALNS{base_num}_60625000"
    base_path2 = rf"{base_folder}\results\exps_record_all_parallelexp60625000"
    
    # 创建临时DataFrame来存储当前路径的合并结果
    combined_df1 = None
    combined_df2 = None
    
    # 遍历5个文件
    for i in range(1, 6):
        try:
            # 构建完整的文件路径
            file_path1 = base_path1 + str(i) + ".xlsx"
            file_path2 = base_path2 + str(i) + "parallel3.xlsx"
            
            # 读取第一组Excel文件
            df1 = pd.read_excel(file_path1)
            if combined_df1 is None:
                combined_df1 = df1
            else:
                combined_df1 = pd.concat([combined_df1, df1.iloc[0:]], ignore_index=True)
            
            # 读取第二组Excel文件
            df2 = pd.read_excel(file_path2)
            if combined_df2 is None:
                combined_df2 = df2
            else:
                combined_df2 = pd.concat([combined_df2, df2.iloc[0:]], ignore_index=True)
        except Exception as e:
            print(f"处理文件时出错 (base_num={base_num}, i={i}): {str(e)}")
            continue
    
    # 将当前路径的结果添加到最终结果中
    if combined_df1 is not None:
        if final_combined_df1 is None:
            final_combined_df1 = combined_df1
        else:
            final_combined_df1 = pd.concat([final_combined_df1, combined_df1], ignore_index=True)
    
    if combined_df2 is not None:
        if final_combined_df2 is None:
            final_combined_df2 = combined_df2
        else:
            final_combined_df2 = pd.concat([final_combined_df2, combined_df2], ignore_index=True)

# 保存最终合并的文件
if final_combined_df1 is not None:
    output_path1 = r"D:\Pycharm\py\code\codes_ALNS-1\单位成本和载具空间利用率\final_combined_results1.xlsx"
    final_combined_df1.to_excel(output_path1, index=False)

if final_combined_df2 is not None:
    output_path2 = r"D:\Pycharm\py\code\codes_ALNS-1\results\final_combined_results2.xlsx"
    final_combined_df2.to_excel(output_path2, index=False)

print("所有文件夹的数据都已合并完成！")