import pandas as pd
import os

def process_experiment(exp_num):
    # 读取 Excel 文件
    routes_df = pd.read_excel(f"D:\Pycharm\py\code\code_gurobi\\Figures\\experiment{exp_num}\\percentage[0, 1]parallel_number3\\routes_matchpercentage[0, 1]parallel_number3{exp_num}.xlsx", header=None)
    intermodal_df = pd.read_excel(f"D:\Pycharm\py\code\code_gurobi\\Figures\\experiment{exp_num}\\Intermodal_EGS_data_all.xlsx", sheet_name='R_10', header=None)
    exp_df = pd.read_excel(f"D:\Pycharm\py\code\code_gurobi\\results\\exps_record_all_parallelexp{exp_num}parallel3.xlsx", sheet_name='exps_record')
    obj_df = pd.read_excel(f"D:\Pycharm\py\code\code_gurobi\\Figures\\experiment{exp_num}\\percentage[0, 1]parallel_number3\\obj_recordpercentage[0, 1]parallel_number3{exp_num}.xlsx", sheet_name='obj_record_best')

    # 初始化 qr 的和
    qr_sum = 0

    # 遍历第 2 行的第 1 到 10 列
    for i in range(1, 11):
        if pd.notna(routes_df.iloc[1, i]):
            qr_sum += intermodal_df.iloc[i, 6]  # 假设 qr 在第 6 列

    # 去除列名中的空格
    exp_df.columns = exp_df.columns.str.strip()

    # 计算 'calculated_value'
    exp_df['calculated_value'] = (
        exp_df['number_used_vehicles'] * exp_df['barge_seved_r_portion'] * 160 +
        exp_df['number_used_vehicles'] * exp_df['train_seved_r_portion'] * 240 +
        exp_df['number_used_vehicles'] * exp_df['truck_seved_r_portion'] * 60
    ) / 100

    # 计算载具空间利用率
    vehicle_utilization = qr_sum / exp_df['calculated_value'].iloc[-1]

    # 去除列名中的空格
    obj_df.columns = obj_df.columns.str.strip()

    # 计算 'cost_per_distance'
    obj_df['cost_per_distance'] = obj_df['overall_cost'] / obj_df['overall_distance']

    # 计算单位kg公里的成本
    cost_per_distance_per_kg = obj_df['cost_per_distance'].iloc[-1] / qr_sum

    #单位碳税（per_ton_km)
    carbon_tax_per_ton = (exp_df['best_emission_cost'].iloc[-1]*1000 / obj_df['cost_per_distance'].iloc[-1]) / qr_sum

    #单位存储成本(per kg)
    unit_storage_cost = exp_df['best_storage_cost'] / qr_sum

    #运输(per kg)
    unit_transit_cost = exp_df['best_request_cost'] / qr_sum

    #延误
    unit_delay_penalty = exp_df['best_delay_penalty'] / qr_sum

    served_requests = obj_df['served_requests'].iloc[-1]

    # 创建 DataFrame 保存结果
    results_df = pd.DataFrame({
        'unit cost': [cost_per_distance_per_kg],
        'load factor': [vehicle_utilization],
        'served_requests': [served_requests],
        'carbon_tax_per_ton': [carbon_tax_per_ton],
        'unit_storage_cost': [unit_storage_cost.iloc[-1]],
        'unit_transit_cost': [unit_transit_cost.iloc[-1]],
        'unit_delay_penalty': [unit_delay_penalty.iloc[-1]]
    })

    # 定义保存路径
    output_folder = r"D:\Pycharm\py\code\code_gurobi\单位成本和载具空间利用率"
    output_file = os.path.join(output_folder, f"结果_{exp_num}.xlsx")

    # 确保文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 保存结果到 Excel 文件
    results_df.to_excel(output_file, index=False)
    print(f"结果已保存到 Excel 文件中：{output_file}")

def main():
    # 遍历实验编号
    for i in range(1, 10):
        exp_num = f"60625000{i}"
        print(f"处理实验 {exp_num}")
        try:
            process_experiment(exp_num)
        except Exception as e:
            print(f"处理实验 {exp_num} 时出错：{str(e)}")

if __name__ == "__main__":
    main()