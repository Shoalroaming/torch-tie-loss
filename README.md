# 光强传输方程损失

基于PyTorch的光强传输方程损失函数，支持GPU加速。

## 功能特性

- 光强传输方程
- 多种损失类型
- 自动维度处理
- GPU加速计算
- 内存优化

## 文件结构

```
.
├── tieloss.py          # TIE损失函数实现
├── memory_test.py      # 内存使用测试
└── optimize_test.py    # 相位重建优化测试 
```

## 核心类

### TIELoss

```python
TIELoss(
    image_size,         # 图像尺寸（正方形）
    wavelength_m=632.8e-9,  # 波长 [m]
    pixel_size_m=8e-6,  # 像素尺寸 [m]
    loss_type='mse',    # 损失类型：'mse', 'mae', 'rmse'
)
```

## 使用示例

### 内存测试

```bash
python test_memory.py
```

测试不同图像尺寸和批大小下的内存使用情况。

### 相位重建优化

```bash
python test_optimize.py
```

执行相位重建优化实验，测试三种策略下的重建效果。实验包含可视化结果和量化评估（MSE、MAE、相关系数）。

## 输入输出

### 输入
- 相位图，光强图，目标光强导数图（维度为(H,W)、(B,H,W) 或 (B,1,H,W)）
- 波长、像素尺寸等物理参数

### 输出
- 标量损失值，表示预测相位对应的光强导数与目标导数的差异

## 依赖

- PyTorch
- NumPy（仅测试脚本）
- matplotlib（仅测试脚本）
- SciPy（仅测试脚本）
