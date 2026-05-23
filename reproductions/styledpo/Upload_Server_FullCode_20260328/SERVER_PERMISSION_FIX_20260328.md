# Server Permission Fix（EACCES unlink）

你看到的错误：
- `Error: EACCES: permission denied, unlink .../benchmark/style_benchmark_...`

本质原因：目标目录里已有同名文件，但当前用户对这些旧文件没有删除权限（常见于之前用 root/sudo 解压过一次）。

## 推荐修复步骤（服务器）

### 方案 A（推荐）：删除旧目录后重新解压
```bash
cd /data/alice/cjtest
# 如果你有 sudo 权限
sudo rm -rf Upload_Server_FullCode_20260328
unzip Style-DPO-v2.8-fullcode-20260328.zip
```

### 方案 B：修正目录所有权后覆盖
```bash
cd /data/alice/cjtest
sudo chown -R $USER:$USER Upload_Server_FullCode_20260328
chmod -R u+rwX Upload_Server_FullCode_20260328
# 然后再解压覆盖
unzip -o Style-DPO-v2.8-fullcode-20260328.zip
```

### 方案 C（无 sudo）：解压到新目录再切换
```bash
cd /data/alice/cjtest
mkdir -p Upload_Server_FullCode_20260328_new
unzip Style-DPO-v2.8-fullcode-20260328.zip -d Upload_Server_FullCode_20260328_new
# 验证后切换使用新目录
```

## 解压后快速校验
```bash
cd /data/alice/cjtest/Upload_Server_FullCode_20260328
python - <<'PY'
from pathlib import Path
root = Path('benchmark/style_benchmark_40x10')
subs = [p for p in root.iterdir() if p.is_dir()]
img_ext = {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
counts = [sum(1 for f in s.iterdir() if f.is_file() and f.suffix.lower() in img_ext) for s in subs]
print('styles=', len(subs), 'total=', sum(counts), 'min=', min(counts), 'max=', max(counts))
PY
```

预期输出：
- `styles= 40 total= 400 min= 10 max= 10`
