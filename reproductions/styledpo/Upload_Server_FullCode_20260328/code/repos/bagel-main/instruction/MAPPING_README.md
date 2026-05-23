# 图片重命名映射说明

## 📋 概述

已成功将 `/data/mayue/cjy/BAGEL/outputtext` 中的 35 张图片按照新的映射规则重命名并复制到 `/data/mayue/cjy/BAGEL/instruction` 目录。

## 🔄 映射规则

### Prompt 映射（旧 → 新）

旧的文件命名基于以下顺序：
- **style01** = 卡通风格（鲜艳色彩） → **新 prompt4**
- **style02** = 黑白风格 → **新 prompt3**
- **style03** = 铅笔素描风格 → **新 prompt1**
- **style04** = 日落暖色调 → **新 prompt2**
- **style05** = 卡通风格（紫粉色） → **新 prompt5**

新的 prompt 顺序：
```
prompt1: Convert this image into pencil sketch style.
prompt2: Make this image style into the sunset style with warm tone.
prompt3: Transform the image into a black and white style.
prompt4: Transform the image into a vibrant, cartoon-style illustration with bold colors and simplified shapes, enhancing the contrast and saturation for a lively, animated look
prompt5: Transform the image into a vibrant, cartoonish style with bright purple and pink colors, enhancing the contrast and adding a whimsical background.
```

### Content 映射

```
content1 → bridge.png
content2 → farm2.jpg
content3 → indoor.png
content4 → Lighthouse on the coast.png
content5 → s5.jpg
content6 → s6.jpg
content7 → tower.jpg
```

## 📁 文件结构

### 源目录结构
```
outputtext/
├── prompt_01/
│   ├── bridge_style01.png
│   ├── farm2_style01.png
│   ├── ... (7张图片)
├── prompt_02/
│   ├── bridge_style02.png
│   ├── ... (7张图片)
├── ... (prompt_03, 04, 05)
```

### 目标目录结构
```
instruction/
├── instruction.txt (新顺序的5个prompt)
├── content1_prompt1.png (bridge + 铅笔素描)
├── content1_prompt2.png (bridge + 日落)
├── content1_prompt3.png (bridge + 黑白)
├── content1_prompt4.png (bridge + 卡通鲜艳)
├── content1_prompt5.png (bridge + 卡通紫粉)
├── ... (共35张图片)
└── content7_prompt5.png (tower + 卡通紫粉)
```

## ✅ 重命名示例

| 原文件名 | 新文件名 | 说明 |
|---------|---------|------|
| `prompt_01/bridge_style01.png` | `content1_prompt4.png` | bridge + 卡通鲜艳 |
| `prompt_02/farm2_style02.png` | `content2_prompt3.png` | farm2 + 黑白 |
| `prompt_03/indoor_style03.png` | `content3_prompt1.png` | indoor + 铅笔素描 |
| `prompt_04/s5_style04.png` | `content5_prompt2.png` | s5 + 日落 |
| `prompt_05/tower_style05.png` | `content7_prompt5.png` | tower + 卡通紫粉 |

## 🎯 完整映射表

### content1 (bridge.png)
- `prompt_03/bridge_style03.png` → `content1_prompt1.png` (铅笔素描)
- `prompt_04/bridge_style04.png` → `content1_prompt2.png` (日落)
- `prompt_02/bridge_style02.png` → `content1_prompt3.png` (黑白)
- `prompt_01/bridge_style01.png` → `content1_prompt4.png` (卡通鲜艳)
- `prompt_05/bridge_style05.png` → `content1_prompt5.png` (卡通紫粉)

### content2 (farm2.jpg)
- `prompt_03/farm2_style03.png` → `content2_prompt1.png` (铅笔素描)
- `prompt_04/farm2_style04.png` → `content2_prompt2.png` (日落)
- `prompt_02/farm2_style02.png` → `content2_prompt3.png` (黑白)
- `prompt_01/farm2_style01.png` → `content2_prompt4.png` (卡通鲜艳)
- `prompt_05/farm2_style05.png` → `content2_prompt5.png` (卡通紫粉)

### content3 (indoor.png)
- `prompt_03/indoor_style03.png` → `content3_prompt1.png` (铅笔素描)
- `prompt_04/indoor_style04.png` → `content3_prompt2.png` (日落)
- `prompt_02/indoor_style02.png` → `content3_prompt3.png` (黑白)
- `prompt_01/indoor_style01.png` → `content3_prompt4.png` (卡通鲜艳)
- `prompt_05/indoor_style05.png` → `content3_prompt5.png` (卡通紫粉)

### content4 (Lighthouse on the coast.png)
- `prompt_03/Lighthouse on the coast_style03.png` → `content4_prompt1.png` (铅笔素描)
- `prompt_04/Lighthouse on the coast_style04.png` → `content4_prompt2.png` (日落)
- `prompt_02/Lighthouse on the coast_style02.png` → `content4_prompt3.png` (黑白)
- `prompt_01/Lighthouse on the coast_style01.png` → `content4_prompt4.png` (卡通鲜艳)
- `prompt_05/Lighthouse on the coast_style05.png` → `content4_prompt5.png` (卡通紫粉)

### content5 (s5.jpg)
- `prompt_03/s5_style03.png` → `content5_prompt1.png` (铅笔素描)
- `prompt_04/s5_style04.png` → `content5_prompt2.png` (日落)
- `prompt_02/s5_style02.png` → `content5_prompt3.png` (黑白)
- `prompt_01/s5_style01.png` → `content5_prompt4.png` (卡通鲜艳)
- `prompt_05/s5_style05.png` → `content5_prompt5.png` (卡通紫粉)

### content6 (s6.jpg)
- `prompt_03/s6_style03.png` → `content6_prompt1.png` (铅笔素描)
- `prompt_04/s6_style04.png` → `content6_prompt2.png` (日落)
- `prompt_02/s6_style02.png` → `content6_prompt3.png` (黑白)
- `prompt_01/s6_style01.png` → `content6_prompt4.png` (卡通鲜艳)
- `prompt_05/s6_style05.png` → `content6_prompt5.png` (卡通紫粉)

### content7 (tower.jpg)
- `prompt_03/tower_style03.png` → `content7_prompt1.png` (铅笔素描)
- `prompt_04/tower_style04.png` → `content7_prompt2.png` (日落)
- `prompt_02/tower_style02.png` → `content7_prompt3.png` (黑白)
- `prompt_01/tower_style01.png` → `content7_prompt4.png` (卡通鲜艳)
- `prompt_05/tower_style05.png` → `content7_prompt5.png` (卡通紫粉)

## ✅ 验证结果

- ✅ 总共处理: **35 张图片**
- ✅ 成功复制: **35 张图片**
- ✅ 错误: **0 个**
- ✅ 文件命名: `content{1-7}_prompt{1-5}.png`
- ✅ 保存位置: `/data/mayue/cjy/BAGEL/instruction/`

## 📝 使用说明

现在你可以使用这些重命名后的图片进行评估：
- 图片位置: `/data/mayue/cjy/BAGEL/instruction/`
- Prompt 文件: `/data/mayue/cjy/BAGEL/instruction/instruction.txt`
- 命名格式: `content{X}_prompt{Y}.png`，其中 X=1-7，Y=1-5

这个新的目录结构与 HQ-Edit 评估脚本的映射配置完全匹配！
