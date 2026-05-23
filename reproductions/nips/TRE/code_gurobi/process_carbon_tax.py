import pandas as pd
import os

def process_experiment(exp_num):
    try:
        # 构建文件路径
        file_path = f"D:/Pycharm/py/code/codes_ALNS/单位成本和载具空间利用率/结果{exp_num}.xlsx"
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None
            
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 获取carbon_tax_per_ton值
        if 'carbon_tax_per_ton' in df.columns:
            carbon_tax = df['carbon_tax_per_ton'].iloc[0]
            # 转换为字符串并获取最后一位
            last_digit = str(carbon_tax)[-1]
            return last_digit
        else:
            print(f"实验{exp_num}中未找到carbon_tax_per_ton列")
            return None
            
    except Exception as e:
        print(f"处理实验{exp_num}时发生错误: {str(e)}")
        return None

def main():
    results = {}
    # 遍历实验编号
    for i in range(1, 10):
        exp_num = f"60625000{i}"
        result = process_experiment(exp_num)
        if result is not None:
            results[exp_num] = result
    
    # 打印结果
    print("\n处理结果:")
    print("-" * 30)
    for exp_num, last_digit in results.items():
        print(f"实验 {exp_num}: carbon_tax_per_ton 的最后一位是 {last_digit}")

if __name__ == "__main__":
    main() 