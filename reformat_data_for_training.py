#!/usr/bin/env python3
"""
将转换后的数据格式调整为 train_wan_i2v_lora.py 脚本期望的格式
输入: metadata.json + train/ 目录
输出: train.jsonl + images/ + videos/
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import List
import sys

DATASET_DIR = Path("/data/alice/cjtest/datasets/worldarena_wan_i2v_clean50")
TRAIN_DIR = DATASET_DIR / "train"
IMAGES_DIR = DATASET_DIR / "images"
VIDEOS_DIR = DATASET_DIR / "videos"


def setup_directories():
    """创建必要的目录"""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ 创建目录: images/, videos/")


def extract_first_frame_ffmpeg(video_path: Path, output_image: Path) -> bool:
    """使用 ffmpeg 提取视频第一帧"""
    try:
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", "select=eq(n\\,0)",
            "-q:v", "2",
            "-y",
            str(output_image)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and output_image.exists()
    except:
        return False


def extract_first_frame_opencv(video_path: Path, output_image: Path) -> bool:
    """使用 opencv 提取视频第一帧（备选方案）"""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite(str(output_image), frame)
            return output_image.exists()
    except:
        pass
    return False


def extract_first_frame_imageio(video_path: Path, output_image: Path) -> bool:
    """使用 imageio 提取视频第一帧（备选方案）"""
    try:
        import imageio
        from PIL import Image
        reader = imageio.get_reader(str(video_path))
        frame = reader.get_data(0)
        reader.close()
        img = Image.fromarray(frame)
        img.save(output_image)
        return output_image.exists()
    except:
        pass
    return False


def extract_first_frames():
    """提取所有视频的第一帧"""
    print("\n📸 提取视频第一帧...")
    
    metadata_file = DATASET_DIR / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    success_count = 0
    failed_count = 0
    
    for i, sample in enumerate(metadata, 1):
        video_file = sample['video_file']
        video_path = TRAIN_DIR / video_file
        
        # 生成输出图片文件名
        image_file = video_file.replace('.mp4', '.png')
        image_path = IMAGES_DIR / image_file
        
        # 如果已存在，跳过
        if image_path.exists():
            success_count += 1
            if i % 100 == 0:
                print(f"  进度: {i}/{len(metadata)} (已存在)")
            continue
        
        # 尝试多种方法提取第一帧
        extracted = False
        
        # 优先使用 ffmpeg
        if extract_first_frame_ffmpeg(video_path, image_path):
            extracted = True
        # 备选：opencv
        elif extract_first_frame_opencv(video_path, image_path):
            extracted = True
        # 备选：imageio
        elif extract_first_frame_imageio(video_path, image_path):
            extracted = True
        
        if extracted:
            success_count += 1
        else:
            failed_count += 1
            print(f"  ⚠ 提取失败: {video_file}")
        
        if i % 100 == 0:
            print(f"  进度: {i}/{len(metadata)}")
    
    print(f"  ✓ 提取完成: {success_count}/{len(metadata)} 成功")
    if failed_count > 0:
        print(f"  ⚠ 失败: {failed_count}")
    
    return success_count


def link_or_copy_videos():
    """将视频链接或复制到 videos/ 目录"""
    print("\n🎬 处理视频文件...")
    
    metadata_file = DATASET_DIR / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    success_count = 0
    
    for i, sample in enumerate(metadata, 1):
        video_file = sample['video_file']
        src = TRAIN_DIR / video_file
        dst = VIDEOS_DIR / video_file
        
        # 如果已存在（符号链接或复制），跳过
        if dst.exists():
            success_count += 1
            if i % 100 == 0:
                print(f"  进度: {i}/{len(metadata)} (已存在)")
            continue
        
        try:
            # 优先使用符号链接（节省空间）
            if not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                except (OSError, NotImplementedError):
                    # 如果符号链接失败，则复制文件
                    shutil.copy2(src, dst)
            success_count += 1
        except Exception as e:
            print(f"  ⚠ 处理失败: {video_file} - {e}")
        
        if i % 100 == 0:
            print(f"  进度: {i}/{len(metadata)}")
    
    print(f"  ✓ 处理完成: {success_count}/{len(metadata)}")


def create_jsonl():
    """从 metadata.json 创建 train.jsonl"""
    print("\n📝 生成 train.jsonl...")
    
    metadata_file = DATASET_DIR / "metadata.json"
    jsonl_file = DATASET_DIR / "train.jsonl"
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # 清空或创建 jsonl 文件
    with open(jsonl_file, 'w') as f:
        for sample in metadata:
            video_file = sample['video_file']
            image_file = video_file.replace('.mp4', '.png')
            instruction = sample['instruction']
            
            # 创建 jsonl 记录
            record = {
                "prompt": instruction,
                "image": image_file,
                "video": video_file
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"  ✓ 生成完成: {len(metadata)} 行数据")


def verify_data():
    """验证数据格式是否完整"""
    print("\n✅ 验证数据格式...")
    
    jsonl_file = DATASET_DIR / "train.jsonl"
    
    # 检查 train.jsonl
    if not jsonl_file.exists():
        print("  ✗ train.jsonl 不存在")
        return False
    
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()
    
    print(f"  ✓ train.jsonl: {len(lines)} 行")
    
    # 检查目录
    image_count = len(list(IMAGES_DIR.glob('*.png')))
    video_count = len(list(VIDEOS_DIR.glob('*.mp4')))
    
    print(f"  ✓ images/: {image_count} 张图片")
    print(f"  ✓ videos/: {video_count} 个视频")
    
    # 检查第一行格式
    if lines:
        first_record = json.loads(lines[0])
        required_keys = {'prompt', 'image', 'video'}
        if required_keys.issubset(first_record.keys()):
            print(f"  ✓ JSONL 格式正确")
            print(f"\n    示例记录:")
            print(f"    prompt: {first_record['prompt'][:80]}...")
            print(f"    image: {first_record['image']}")
            print(f"    video: {first_record['video']}")
        else:
            print(f"  ✗ JSONL 格式错误：缺少 {required_keys - set(first_record.keys())}")
            return False
    
    # 最终检查
    all_valid = image_count > 0 and video_count > 0 and len(lines) > 0
    
    if all_valid:
        print(f"\n✓ 数据格式验证完成！可以开始训练。")
    else:
        print(f"\n⚠ 数据格式验证失败")
    
    return all_valid


def main():
    print("=" * 100)
    print("数据格式转换器：metadata.json + train/ → train.jsonl + images/ + videos/")
    print("=" * 100)
    print()
    
    # 步骤 1: 创建目录
    setup_directories()
    
    # 步骤 2: 提取第一帧
    image_count = extract_first_frames()
    
    # 步骤 3: 处理视频文件
    link_or_copy_videos()
    
    # 步骤 4: 生成 JSONL
    create_jsonl()
    
    # 步骤 5: 验证
    verify_data()
    
    print()
    print("=" * 100)
    print("✓ 数据准备完成！")
    print(f"  数据集目录: {DATASET_DIR}")
    print("  可以使用以下命令启动训练:")
    print()
    print("  python3 /data/alice/cjtest/model_repros/wan_sft_workspace/train_wan_i2v_lora.py \\")
    print("    --model-path /path/to/Wan2.1-I2V-14B-480P \\")
    print(f"    --dataset-dir {DATASET_DIR} \\")
    print("    --output-dir /path/to/output \\")
    print("    --num-frames 121 \\")
    print("    --batch-size 1 \\")
    print("    --max-steps 330")
    print("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ 被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
