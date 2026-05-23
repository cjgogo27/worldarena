import os
import shutil

# 源文件夹路径
source_dir = "/root/autodl-tmp/AesFA-main/output/main/256"

# 目标文件夹路径
content_dir = "/root/autodl-tmp/AesFA-main/data_evl/content"
style_dir = "/root/autodl-tmp/AesFA-main/data_evl/style"
tar_dir = "/root/autodl-tmp/AesFA-main/data_evl/tar"

# 创建目标文件夹（如果不存在）
os.makedirs(content_dir, exist_ok=True)
os.makedirs(style_dir, exist_ok=True)
os.makedirs(tar_dir, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_dir):
    # 构建完整的文件路径
    source_path = os.path.join(source_dir, filename)
    
    # 只处理文件，不处理文件夹
    if os.path.isfile(source_path):
        # 根据文件名中的关键词进行分类
        if "_content_style_" in filename:
            target_path = os.path.join(content_dir, filename)
            shutil.move(source_path, target_path)
            print(f"移动文件: {filename} -> content 目录")
        elif "_style_style" in filename:
            target_path = os.path.join(style_dir, filename)
            shutil.move(source_path, target_path)
            print(f"移动文件: {filename} -> style 目录")
        elif "_stylized_style" in filename:
            target_path = os.path.join(tar_dir, filename)
            shutil.move(source_path, target_path)
            print(f"移动文件: {filename} -> tar 目录")

print("文件分类移动完成！")
