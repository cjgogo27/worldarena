# 海淀区OD分析 - 后台运行指南

## 修改说明

### 1. 坐标范围优化
已将硬编码的经纬度范围改为**从Shapefile动态读取**：

**原来（硬编码）：**
```python
经度: 116.0 - 116.5
纬度: 39.9 - 40.2
```

**现在（动态读取 + 1km缓冲区）：**
```python
实际边界: 
  经度: 116.042532°E - 116.388829°E
  纬度: 39.885475°N - 40.159687°N

使用范围（+0.01°缓冲 ≈ 1km）:
  经度: 116.032532°E - 116.398829°E  
  纬度: 39.875475°N - 40.169687°N
```

**优点：**
- ✅ 自动适应Shapefile实际边界
- ✅ 不会遗漏边界附近的有效数据点
- ✅ 避免过度过滤（原方案南侧切掉0.01°）
- ✅ 小缓冲区考虑GPS定位误差

---

## 2. 后台运行方案

### 启动后台任务
```bash
cd /data/alice/cjtest/TRC/haidian_od_analysis
./run_background.sh
```

**输出示例：**
```
========================================
海淀区OD分析 - 后台运行
========================================
Screen会话名: haidian_od_analysis
日志文件: /data/alice/cjtest/TRC/haidian_od_analysis/logs/run_20260203_143025.log
开始时间: Mon Feb  3 14:30:25 CST 2026
========================================
✓ 后台任务已启动成功

常用命令:
  查看实时日志:   tail -f /data/alice/.../logs/run_20260203_143025.log
  进入screen会话: screen -r haidian_od_analysis
  退出screen会话: Ctrl+A then D (不要Ctrl+C，会终止程序)
  查看所有会话:   screen -ls
  终止任务:       screen -X -S haidian_od_analysis quit

预计运行时间: 2-4小时 (取决于数据量)
```

---

### 检查运行进度
```bash
./check_progress.sh
```

**输出示例：**
```
========================================
海淀区OD分析 - 进度检查
========================================
当前时间: Mon Feb  3 15:15:30 CST 2026

✓ 后台任务正在运行
  Screen会话: haidian_od_analysis

最新日志: /data/alice/.../logs/run_20260203_143025.log
----------------------------------------
最后20行输出:
----------------------------------------
步骤 5/6: 提取出行记录并生成OD矩阵
正在提取出行记录...
开始从轨迹数据提取出行记录...
原始轨迹点数: 2111817
过滤后轨迹点数: 2111817
正在处理 9719 辆车的轨迹...
Extracting trips:  15%|████          | 1500/9719 [12:30<1:23:15]
  [1500/9719] vehicles | Trips: 45231 | Avg: 0.50s/veh | ETA: 68.3min
----------------------------------------

输出文件检查:
----------------------------------------
✓ step1: 3 个文件
✓ step2: 2 个文件
✓ step3: 3 个文件
✓ step4: 2 个文件
  step5: 未生成
  step6: 未生成
========================================
```

---

### 查看实时日志
```bash
# 最新日志文件路径（由check_progress.sh提供）
tail -f /data/alice/cjtest/TRC/haidian_od_analysis/logs/run_20260203_143025.log

# 按 Ctrl+C 退出日志查看（不影响后台任务）
```

---

### 进入交互式会话（高级操作）
```bash
screen -r haidian_od_analysis

# 在screen会话内：
#   - 可以看到程序实时输出
#   - Ctrl+C 可以中断程序
#   - Ctrl+A then D 可以退出会话（程序继续运行）
```

---

### 停止后台任务
```bash
# 方法1: 使用脚本命令
screen -X -S haidian_od_analysis quit

# 方法2: 进入会话后按 Ctrl+C
screen -r haidian_od_analysis
# 然后按 Ctrl+C
```

---

## 3. 文件说明

### 新增文件
- **`run_background.sh`**: 后台启动脚本
- **`check_progress.sh`**: 进度检查脚本
- **`logs/run_YYYYMMDD_HHMMSS.log`**: 运行日志（自动生成）

### 输出目录结构
```
haidian_od_analysis/
├── logs/                    # 新增：日志目录
│   └── run_20260203_143025.log
├── output/
│   ├── step1_regions.csv
│   ├── step1_regions.geojson
│   ├── step2_trajectory_sample.csv
│   ├── step3_trajectory_with_regions.csv
│   ├── step4_trajectory_with_time.csv
│   ├── step5_trips.csv       # 待生成
│   ├── step6_od_matrices.h5  # 待生成
│   └── visualizations/
├── run_background.sh         # 新增
├── check_progress.sh         # 新增
└── main.py                   # 已修改：动态读取边界
```

---

## 4. 常见问题

### Q1: 如何确认任务是否在运行？
```bash
./check_progress.sh
# 或
screen -ls
```

### Q2: 任务意外中断怎么办？
- 查看最新日志文件，找到错误信息
- 修复问题后重新运行 `./run_background.sh`

### Q3: 如何查看历史日志？
```bash
ls -lth logs/  # 按时间倒序列出所有日志
cat logs/run_20260203_143025.log  # 查看特定日志
```

### Q4: screen会话消失了？
```bash
screen -ls  # 检查所有会话
# 如果没有会话，说明程序已完成或异常退出，查看日志确认
```

---

## 5. 推荐工作流程

```bash
# 1. 启动后台任务
cd /data/alice/cjtest/TRC/haidian_od_analysis
./run_background.sh

# 2. 定期检查进度（建议每30分钟）
./check_progress.sh

# 3. 查看实时日志（可选）
tail -f logs/run_*.log

# 4. 任务完成后检查结果
./check_progress.sh
ls -lh output/step*.csv
```

---

## 6. 性能预估

- **步骤1-4**: 5-10分钟
- **步骤5（trip提取）**: 2-3小时（9719辆车）
- **步骤6（保存结果）**: 5-10分钟
- **总计**: 约2.5-4小时

进度更新频率：
- 每处理10辆车更新进度
- 每秒刷新tqdm进度条
- 进度日志实时保存到文件
