import torch
from PIL import Image
from torchvision import transforms
from metrics.extranet import ExtraVGG19
from rl.function import calc_content_loss, calc_style_loss
from pathlib import Path
import glob
import os
import time  # 用于记录计算时间


# 配置（完全复用推理脚本的参数）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vgg_path = "./metrics/vgg_normalised.pth"
content_weight = 10.0  # 训练时用，推理打印的是未加权值
style_weight = 3.0
size = 256  # 与推理脚本的--content_size一致
crop = False  # 关键：与推理脚本的--crop默认值一致（False）

# 图像文件夹路径 - 根据实际情况修改
content_dir = "/data/mayue/cjy/BAGEL/data_evl/content"  # 内容图文件夹
style_dir = "/data/mayue/cjy/BAGEL/data_evl/style"      # 风格图文件夹
stylized_dir = "/data/mayue/cjy/BAGEL/output/batch_results"  # 风格化结果文件夹

# 支持的图像格式
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

# --------------------------
# 1. 图像预处理：完全复制test_transform逻辑
# --------------------------
def test_transform(size, crop=True):
    transform_list = []
    if size != 0:
        # 强制缩放到正方形（元组参数）
        transform_list.append(transforms.Resize((size, size)))
    if crop and size != 0:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

# --------------------------
# 2. 加载单张图像
# --------------------------
def load_image(path, transform):
    try:
        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0)  # 添加batch维度
        return img.to(device)
    except Exception as e:
        print(f"加载图像 {path} 失败: {e}")
        return None

# --------------------------
# 3. 获取文件夹中的所有图像路径
# --------------------------
def get_image_paths(folder):
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(folder, ext)))
    # 按文件名排序，确保内容图、风格图和结果图能正确对应
    image_paths.sort()
    return image_paths

# --------------------------
# 4. 计算单组图像的损失
# --------------------------
def calculate_single_loss(content_img, style_img, stylized_img, extranet):
    with torch.no_grad():
        # 提取特征
        content_feats = extranet(content_img)
        style_feats = extranet(style_img)
        stylized_feats = extranet(stylized_img)

        content_loss = 0.0
        style_loss = 0.0
        
        for i in range(len(stylized_feats)):
            # 调整风格特征尺寸以匹配风格化特征
            if stylized_feats[i].shape != style_feats[i].shape:
                style_feats_i = transforms.Resize(stylized_feats[i].shape[2:])(style_feats[i])
            else:
                style_feats_i = style_feats[i]

            # 内容损失：仅计算relu4_1（倒数第2层）
            if i == len(stylized_feats) - 2:
                content_loss = calc_content_loss(stylized_feats[i], content_feats[i])

            # 风格损失：计算前4层（0-3）
            if i < 4:
                if i == 0:
                    style_loss = calc_style_loss(stylized_feats[i], style_feats_i)
                else:
                    style_loss += calc_style_loss(stylized_feats[i], style_feats_i)

        total_loss = content_loss + style_loss
        
        return {
            'content_loss': content_loss.item(),
            'style_loss': style_loss.item(),
            'total_loss': total_loss.item()
        }

# --------------------------
# 5. 保存损失结果到文件
# --------------------------
def save_loss_results(save_dir, all_losses, content_paths, style_paths, stylized_paths, valid_indices):
    # 确保保存目录存在（自动创建stylized_dir/loss）
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "batch_loss_results.txt")
    
    # 获取当前时间（用于记录）
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    with open(save_path, "w", encoding="utf-8") as f:
        # 写入文件头（基本信息）
        f.write("=" * 60 + "\n")
        f.write(f"批量损失计算结果 - 生成时间: {current_time}\n")
        f.write(f"使用设备: {device}\n")
        f.write(f"图像尺寸: {size}x{size}\n")
        f.write(f"内容图文件夹: {content_dir}\n")
        f.write(f"风格图文件夹: {style_dir}\n")
        f.write(f"风格化结果文件夹: {stylized_dir}\n")
        f.write("=" * 60 + "\n")
        
        # 写入表头（每组细节）
        f.write(f"{'序号':<4} | {'内容图':<20} | {'风格图':<20} | {'风格化图':<20} | {'内容损失':<12} | {'风格损失':<12} | {'总损失':<12}\n")
        f.write("-" * 120 + "\n")
        
        # 写入每组的详细结果
        for idx, (loss, valid_idx) in enumerate(zip(all_losses, valid_indices)):
            # 获取图像文件名（仅保留文件名，去掉路径）
            content_name = os.path.basename(content_paths[valid_idx])
            style_name = os.path.basename(style_paths[valid_idx])
            stylized_name = os.path.basename(stylized_paths[valid_idx])
            
            # 写入一行数据（对齐格式）
            f.write(f"{idx + 1:<4} | {content_name:<20} | {style_name:<20} | {stylized_name:<20} | {loss['content_loss']:<12.6f} | {loss['style_loss']:<12.6f} | {loss['total_loss']:<12.6f}\n")
        
        # 写入平均值
        f.write("\n" + "=" * 60 + "\n")
        avg_content = sum(l['content_loss'] for l in all_losses) / len(all_losses)
        avg_style = sum(l['style_loss'] for l in all_losses) / len(all_losses)
        avg_total = sum(l['total_loss'] for l in all_losses) / len(all_losses)
        f.write(f"有效组数: {len(all_losses)} 组\n")
        f.write(f"平均内容损失: {avg_content:.6f}\n")
        f.write(f"平均风格损失: {avg_style:.6f}\n")
        f.write(f"平均总损失: {avg_total:.6f}\n")
        f.write("=" * 60 + "\n")
    
    print(f"\n损失结果已保存至: {save_path}")

# --------------------------
# 6. 批量处理主函数
# --------------------------
def batch_calculate_loss():
    # 初始化特征提取网络
    extranet = ExtraVGG19(vgg_path).to(device)
    extranet.eval()
    print(f"使用设备: {device}")

    # 获取所有图像路径
    content_paths = get_image_paths(content_dir)
    style_paths = get_image_paths(style_dir)
    stylized_paths = get_image_paths(stylized_dir)
    
    # 确保三组图像数量一致
    min_count = min(len(content_paths), len(style_paths), len(stylized_paths))
    if min_count == 0:
        print("没有找到足够的图像进行处理")
        return
    
    print(f"找到 {len(content_paths)} 张内容图, {len(style_paths)} 张风格图, {len(stylized_paths)} 张风格化结果图")
    print(f"将处理 {min_count} 组图像")

    # 初始化预处理
    transform = test_transform(size=size, crop=crop)
    
    # 存储所有损失结果和有效索引（用于对应图像路径）
    all_losses = []
    valid_indices = []  # 记录有效组的原始索引（避免跳过的组对应错误）
    
    # 批量处理图像
    for i in range(min_count):
        # 打印进度
        if (i + 1) % 5 == 0 or i == min_count - 1:
            print(f"处理进度: {i + 1}/{min_count}")
        
        # 获取图像路径
        content_path = content_paths[i]
        style_path = style_paths[i]
        stylized_path = stylized_paths[i]
        
        # 加载图像
        content_img = load_image(content_path, transform)
        style_img = load_image(style_path, transform)
        stylized_img = load_image(stylized_path, transform)
        
        # 检查图像是否加载成功
        if content_img is None or style_img is None or stylized_img is None:
            print(f"跳过第 {i + 1} 组图像，因为加载失败")
            continue
        
        # 计算损失
        losses = calculate_single_loss(content_img, style_img, stylized_img, extranet)
        all_losses.append(losses)
        valid_indices.append(i)  # 记录当前有效组的原始索引
        
        # 可选：打印每组的损失
        # print(f"第 {i + 1} 组 - 内容损失: {losses['content_loss']:.6f}, 风格损失: {losses['style_loss']:.6f}, 总损失: {losses['total_loss']:.6f}")
    
    # 计算并打印平均值
    if all_losses:
        avg_content = sum(l['content_loss'] for l in all_losses) / len(all_losses)
        avg_style = sum(l['style_loss'] for l in all_losses) / len(all_losses)
        avg_total = sum(l['total_loss'] for l in all_losses) / len(all_losses)
        
        print("\n" + "=" * 50)
        print(f"共处理 {len(all_losses)} 组有效图像")
        print(f"平均内容损失: {avg_content:.6f}")
        print(f"平均风格损失: {avg_style:.6f}")
        print(f"平均总损失: {avg_total:.6f}")
        print("=" * 50)
        
        # 保存结果到 stylized_dir/loss 文件夹
        loss_save_dir = os.path.join(stylized_dir, "loss")  # 固定在stylized_dir下创建loss子文件夹
        save_loss_results(loss_save_dir, all_losses, content_paths, style_paths, stylized_paths, valid_indices)
    else:
        print("没有有效的损失计算结果")
        return

if __name__ == "__main__":
    batch_calculate_loss()