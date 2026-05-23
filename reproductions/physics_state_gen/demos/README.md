# Particle/Coordinate Diffusion Mini Demos

这组 demo 对应你给的 3 个实验方向，每个实验都在独立文件夹里，执行后会自动训练并输出静态图与动画。

## 目录

- `demo1_1d_particle_flow/`：方案一，1D 粒子流，观察 spectral bias（平滑分布 vs 离散尖峰）
- `demo2_2d_shape_morph/`：方案二，2D 点云从圆形连续形变到三角形
- `demo3_coordinate_digit/`：方案三，坐标版“手写数字 8”点云生成

## 快速运行

在项目根目录执行：

```bash
python demos/demo1_1d_particle_flow/run.py
python demos/demo2_2d_shape_morph/run.py
python demos/demo3_coordinate_digit/run.py
```

每个 demo 会在各自目录下生成：

- `outputs/*.png`：可视化图
- `outputs/*.gif`：粒子/点云演化动画
- `outputs/*.mp4`：若本机可用 `ffmpeg`，则额外导出 MP4
- `outputs/metrics.json`：关键指标

## 依赖

- Python 3.9+
- torch
- numpy
- matplotlib
