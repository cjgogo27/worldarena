from huggingface_hub import snapshot_download
import os
import time
import sys

save_dir = "models/BAGEL-7B-MoT"
repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
cache_dir = save_dir + "/cache"

# 确保目录存在
os.makedirs(save_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# 设置更稳定的 hf_transfer 参数
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_TRANSFER_CONCURRENCY"] = "2"  # 进一步降低并发数，提高稳定性
os.environ["HF_TRANSFER_CHUNK_SIZE"] = "8388608"  # 8MB chunks，更保守的设置
os.environ["HF_TRANSFER_RETRIES"] = "5"  # hf_transfer 内部重试次数

def download_with_retry(max_retries=5):  # 增加重试次数
    """带重试机制的下载函数，配置更稳定的参数"""
    for attempt in range(max_retries):
        try:
            print(f"开始下载 {repo_id}... (尝试 {attempt + 1}/{max_retries})")
            
            snapshot_download(
                cache_dir=cache_dir,
                local_dir=save_dir,
                repo_id=repo_id,
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
                max_workers=1,  # 单线程更稳定
                tqdm_class=None,  # 简化进度显示
                ignore_patterns=["*.git*"],  # 忽略git文件
            )
            print("下载完成!")
            return True
            
        except RuntimeError as e:
            if "hf_transfer" in str(e):
                print(f"hf_transfer 错误 (尝试 {attempt + 1}): {e}")
                print("尝试使用更保守的设置...")
            else:
                print(f"下载失败 (尝试 {attempt + 1}): {e}")
        except Exception as e:
            print(f"未预期的错误 (尝试 {attempt + 1}): {e}")
            
        if attempt < max_retries - 1:
            wait_time = min((attempt + 1) * 3, 15)  # 最多等待15秒
            print(f"等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
        else:
            print("所有重试都失败了。建议检查:")
            print("1. 网络连接是否稳定")
            print("2. 磁盘空间是否充足")
            print("3. 或尝试设置 HF_HUB_ENABLE_HF_TRANSFER=0 禁用 hf_transfer")
            return False

if __name__ == "__main__":
    success = download_with_retry()
    if not success:
        sys.exit(1)
