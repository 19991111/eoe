# _experimental - 实验性功能目录

此目录存放当前未启用但未来可能使用的实验性功能。

## 当前内容

### gpu_accel/
GPU加速模块 - 使用PyTorch在NVIDIA A100上加速EOE计算

**状态**: 🧪 实验中 - 未集成

**用途**:
- 神经网络前向/反向传播GPU加速
- 大规模Agent并行计算
- 环境渲染加速

**启用条件**:
- CUDA可用
- 需要集成到 environment_v12.py
- 需要性能基准测试验证

**使用方式**:
```python
from _experimental.gpu_accel import gpu_accel
```

---

## 添加新实验功能

1. 在此目录创建子目录
2. 添加 README.md 说明用途
3. 更新本文件