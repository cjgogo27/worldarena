import pandas as pd
import os

# 定义实验编号和对应的sheet名称映射
exp_sheet_mapping = {
    '6062500035': 'R_10',
}

# 遍历ALNS文件夹
for alns_num in range(1, 2):
    print(f"\n处理 ALNS-{alns_num} 文件夹...")
    
    for exp_id, sheet_name in exp_sheet_mapping.items():
        # 读取 Excel 文件
        try:
            routes_df = pd.read_excel(fr"E:\TRE\Vehicle_Load\Figures\experiment6062500035\percentage[0, 0]parallel_number3\routes_matchpercentage[0, 0]parallel_number36062500035.xlsx", header=None)
            intermodal_df = pd.read_excel(fr"E:\TRE\Vehicle_Load\Intermodal_EGS_data_all.xlsx", sheet_name=sheet_name, header=None)
            exp_df = pd.read_excel(fr"E:\TRE\Vehicle_Load\results\exps_record_all_parallelexp6062500035parallel3.xlsx", sheet_name='exps_record')
            obj_df = pd.read_excel(fr"E:\TRE\Vehicle_Load\Figures\experiment6062500035\percentage[0, 0]parallel_number3\obj_recordpercentage[0, 0]parallel_number36062500035.xlsx", sheet_name='obj_record_best')
        except FileNotFoundError as e:
            print(f"警告：在处理 ALNS-{alns_num} 的实验 {exp_id} 时找不到文件：{str(e)}")
            continue

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
            'alns_version': [f'ALNS-{alns_num}'],
            'experiment_id': [exp_id],
            'sheet_name': [sheet_name],
            'unit cost': [cost_per_distance_per_kg],
            'load factor': [vehicle_utilization],
            'served_requests': [served_requests],
            'carbon_tax_per_ton': [carbon_tax_per_ton],
            'unit_storage_cost': [unit_storage_cost.iloc[-1]],
            'unit_transit_cost': [unit_transit_cost.iloc[-1]],
            'unit_delay_penalty': [unit_delay_penalty.iloc[-1]]
        })

        # 定义保存路径
        output_folder = fr"E:\TRE\Vehicle_Load\单位成本和载具空间利用率"
        output_file = os.path.join(output_folder, f"结果_ALNS{alns_num}_{exp_id}.xlsx")

        # 确保文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 保存结果到 Excel 文件
        results_df.to_excel(output_file, index=False)

        print(f"实验 ALNS-{alns_num}_{exp_id} 的结果已保存到：{output_file}")