"""
GPU加速模块 - 使用PyTorch在NVIDIA A100上加速EOE计算
"""

import torch
import numpy as np
from typing import Tuple, Optional

# GPU设备配置
GPU_DEVICE = torch.device('cuda:3')  # 使用GPU 3

# 缓存的模型和张量
_cached_model: Optional[torch.nn.Module] = None
_cached_weights: Optional[dict] = None


def get_gpu_device():
    """获取GPU设备"""
    return GPU_DEVICE


def is_gpu_available():
    """检查GPU是否可用"""
    return torch.cuda.is_available() and torch.cuda.device_count() > 3


def toroidal_distance_gpu(x1: torch.Tensor, y1: torch.Tensor, 
                          x2: torch.Tensor, y2: torch.Tensor,
                          width: float, height: float) -> torch.Tensor:
    """
    GPU加速的环形世界距离计算
    
    使用PyTorch广播机制一次性计算所有距离
    """
    # 确保在GPU上
    if not x1.is_cuda:
        x1 = x1.to(GPU_DEVICE)
    if not y1.is_cuda:
        y1 = y1.to(GPU_DEVICE)
    if not x2.is_cuda:
        x2 = x2.to(GPU_DEVICE)
    if not y2.is_cuda:
        y2 = y2.to(GPU_DEVICE)
    
    # 计算差值
    dx = x2 - x1
    dy = y2 - y1
    
    # 环形世界最短路径
    dx = dx - width * torch.floor(dx / width + 0.5)
    dy = dy - height * torch.floor(dy / height + 0.5)
    
    # 返回距离
    return torch.sqrt(dx**2 + dy**2)


def compute_distances_matrix_gpu(agent_x: np.ndarray, agent_y: np.ndarray,
                                  food_x: np.ndarray, food_y: np.ndarray,
                                  width: float, height: float) -> np.ndarray:
    """
    GPU加速的距离矩阵计算
    
    输入: NumPy数组
    输出: NumPy距离矩阵 (从GPU回传)
    """
    # 转换为PyTorch张量
    ax = torch.tensor(agent_x, dtype=torch.float32, device=GPU_DEVICE)
    ay = torch.tensor(agent_y, dtype=torch.float32, device=GPU_DEVICE)
    fx = torch.tensor(food_x, dtype=torch.float32, device=GPU_DEVICE)
    fy = torch.tensor(food_y, dtype=torch.float32, device=GPU_DEVICE)
    
    # 广播计算距离矩阵
    # shape: (n_food, n_agents)
    diff_x = fx[:, None] - ax[None, :]
    diff_y = fy[:, None] - ay[None, :]
    
    # 环形世界处理
    diff_x = diff_x - width * torch.floor(diff_x / width + 0.5)
    diff_y = diff_y - height * torch.floor(diff_y / height + 0.5)
    
    # 计算距离
    dists = torch.sqrt(diff_x**2 + diff_y**2)
    
    # 返回NumPy数组
    return dists.cpu().numpy()


def batch_forward_gpu(node_activations: np.ndarray, edge_weights: np.ndarray,
                       edge_sources: np.ndarray, edge_targets: np.ndarray,
                       node_types: np.ndarray) -> np.ndarray:
    """
    GPU加速的神经网络前向传播
    
    这是一个简化的批处理版本，用于测试
    完整版本需要考虑节点类型（DELAY, META等）
    """
    n_nodes = len(node_activations)
    n_edges = len(edge_weights)
    
    # 转换为PyTorch张量
    acts = torch.tensor(node_activations, dtype=torch.float32, device=GPU_DEVICE)
    weights = torch.tensor(edge_weights, dtype=torch.float32, device=GPU_DEVICE)
    sources = torch.tensor(edge_sources, dtype=torch.long, device=GPU_DEVICE)
    targets = torch.tensor(edge_targets, dtype=torch.long, device=GPU_DEVICE)
    
    # 初始化输出
    output = acts.clone()
    
    # 简单的矩阵乘法模拟
    for i in range(n_edges):
        src = sources[i]
        tgt = targets[i]
        output[tgt] = output[tgt] + acts[src] * weights[i]
    
    # ReLU激活
    output = torch.relu(output)
    
    return output.cpu().numpy()


def sensor_fusion_gpu(sensor_values: list, fusion_weights: list) -> np.ndarray:
    """
    GPU加速的传感器融合
    
    输入: 多个传感器的值和融合权重
    输出: 融合后的传感器值
    """
    sensors = [torch.tensor(s, dtype=torch.float32, device=GPU_DEVICE) for s in sensor_values]
    weights = torch.tensor(fusion_weights, dtype=torch.float32, device=GPU_DEVICE)
    
    # 加权平均
    result = torch.zeros_like(sensors[0])
    for s, w in zip(sensors, weights):
        result = result + s * w
    
    return result.cpu().numpy()


# ============================================================
# 性能测试
# ============================================================
def benchmark_gpu_distance(n_agents=100, n_food=100, n_iters=1000):
    """测试GPU距离计算性能"""
    import time
    
    # 随机数据
    agent_x = np.random.rand(n_agents) * 100
    agent_y = np.random.rand(n_agents) * 100
    food_x = np.random.rand(n_food) * 100
    food_y = np.random.rand(n_food) * 100
    
    # CPU版本
    t0 = time.time()
    for _ in range(n_iters):
        diff_x = food_x[:, None] - agent_x[None, :]
        diff_y = food_y[:, None] - agent_y[None, :]
        diff_x = diff_x - 100 * np.floor(diff_x / 100 + 0.5)
        diff_y = diff_y - 100 * np.floor(diff_y / 100 + 0.5)
        dists = np.sqrt(diff_x**2 + diff_y**2)
    cpu_time = time.time() - t0
    
    # GPU版本
    t0 = time.time()
    for _ in range(n_iters):
        dists = compute_distances_matrix_gpu(agent_x, agent_y, food_x, food_y, 100, 100)
    gpu_time = time.time() - t0
    
    print(f"距离矩阵计算 ({n_agents} agents × {n_food} food):")
    print(f"  CPU: {cpu_time*1000:.1f}ms ({n_iters}次)")
    print(f"  GPU: {gpu_time*1000:.1f}ms ({n_iters}次)")
    print(f"  加速比: {cpu_time/gpu_time:.1f}×")
    
    return cpu_time, gpu_time


if __name__ == "__main__":
    print("=== GPU性能测试 ===")
    if is_gpu_available():
        print(f"GPU设备: {torch.cuda.get_device_name(3)}")
        benchmark_gpu_distance()
    else:
        print("GPU不可用")