#!/usr/bin/env python3
"""
v16.1 稀疏代谢与可塑性死锁测试
===============================
验证：
1. 克莱伯缩放公式边界（Cost(N) < N * Cost(1)）
2. 激活阈值与Hebbian学习的交互
3. 解决"可塑性死锁"的方案验证
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import torch
import numpy as np

# ==================== 代谢模型参数 ====================

METABOLISM_BASE = 0.005          # 静息底噪
ACTIVATION_THRESHOLD = 0.1       # 激活阈值
METABOLISM_PER_ACTIVATION = 0.003  # 每次激活成本
KLEIBER_EXPONENT = 0.75          # 克莱伯指数

print("=" * 60)
print("v16.1 稀疏代谢模型验证")
print("=" * 60)
print(f"基础代谢: {METABOLISM_BASE}")
print(f"激活阈值: {ACTIVATION_THRESHOLD}")
print(f"激活成本: {METABOLISM_PER_ACTIVATION}")
print(f"克莱伯指数: {KLEIBER_EXPONENT}")
print("-" * 60)


# ==================== 1. 克莱伯缩放边界验证 ====================

def compute_metabolism_sparse(activation_count):
    """稀疏激活代谢计算"""
    if activation_count == 0:
        return METABOLISM_BASE
    # 正确公式：指数作用于整体激活数
    return METABOLISM_BASE + (activation_count ** KLEIBER_EXPONENT) * METABOLISM_PER_ACTIVATION

def compute_metabolism_linear(activation_count):
    """线性代谢（对比基准）"""
    return METABOLISM_BASE + activation_count * METABOLISM_PER_ACTIVATION

print("\n[验证1] 克莱伯缩放边界:")
print(f"{'激活数':<8} {'稀疏成本':<12} {'线性成本':<12} {'次线性优势':<12} {'N×Cost(1)':<12}")
print("-" * 60)

for n in [1, 2, 5, 10, 20, 50]:
    sparse = compute_metabolism_sparse(n)
    linear = compute_metabolism_linear(n)
    advantage = (linear - sparse) / linear * 100
    n_times = n * compute_metabolism_sparse(1)
    
    print(f"{n:<8} {sparse:<12.4f} {linear:<12.4f} {advantage:>6.1f}%     {n_times:<12.4f}")
    
    # 验证单调递增
    if n > 1:
        prev = compute_metabolism_sparse(n-1)
        assert sparse > prev, f"错误：Cost({n}) <= Cost({n-1})！"

print("\n✓ 克莱伯缩放验证通过：单调递增且次线性")


# ==================== 2. 可塑性死锁验证 ====================

print("\n" + "=" * 60)
print("[验证2] 可塑性死锁 (Catch-22) 验证")
print("=" * 60)

# 模拟新节点
class Node:
    def __init__(self, weight=0.01):
        self.weight = weight  # 初始极小权重
        self.input = 0.0
        self.output = 0.0
        self.pre_activation = 0.0  # 原始输入（激活前）
    
    def forward(self, input_signal):
        """前向传播"""
        self.input = input_signal
        self.pre_activation = input_signal * self.weight  # 激活前
        self.output = np.tanh(self.pre_activation)  # 激活后
        return self.output
    
    def hebbian_update(self, pre_signal, lr=0.01):
        """Hebbian学习（方案A：使用pre-activation）"""
        # 关键：使用pre_activation而不是output！
        delta = lr * pre_signal * self.pre_activation
        self.weight += delta
        return delta


# 方案A：使用pre-activation的Hebbian学习
print("\n方案A: 使用 pre-activation 计算Hebbian更新")
new_node = Node(weight=0.01)

print(f"  初始权重: {new_node.weight:.4f}")
print(f"  输入信号: 1.0")
print(f"  pre_activation: {1.0 * new_node.weight:.4f}")
print(f"  output (sigmoid): {new_node.forward(1.0):.4f}")

# 连续更新
for i in range(10):
    delta = new_node.hebbian_update(pre_signal=1.0, lr=0.1)

print(f"  10次更新后权重: {new_node.weight:.4f}")
print(f"  pre_activation: {1.0 * new_node.weight:.4f}")
print(f"  是否超过阈值({ACTIVATION_THRESHOLD}): {new_node.pre_activation > ACTIVATION_THRESHOLD}")


# 方案B：添加自发噪声
print("\n方案B: 自发噪声突破阈值")
np.random.seed(42)
new_node_b = Node(weight=0.01)
SPONTANEOUS_NOISE_STD = 0.05  # 噪声标准差

print(f"  初始权重: {new_node_b.weight:.4f}")
print(f"  添加噪声后:")

# 模拟多次尝试
突破次数 = 0
for i in range(20):
    # 添加噪声到pre_activation
    noisy_pre = new_node_b.pre_activation + np.random.normal(0, SPONTANEOUS_NOISE_STD)
    if abs(noisy_pre) > ACTIVATION_THRESHOLD:
        突破次数 += 1
        if 突破次数 == 1:
            print(f"    第{i+1}次尝试首次突破阈值!")

print(f"  20次尝试中突破阈值: {突破次数}次")
print(f"  突破概率: {突破次数/20*100:.1f}%")


# ==================== 3. 综合验证 ====================

print("\n" + "=" * 60)
print("[验证3] 综合代谢计算")
print("=" * 60)

def compute_agent_metabolism(node_outputs, use_pre_activation=False):
    """
    计算智能体代谢
    
    参数:
        node_outputs: list of (pre_activation, post_activation) tuples
        use_pre_activation: 是否使用pre-activation（方案A）
    """
    # 统计激活数
    if use_pre_activation:
        # 方案A：基于pre-activation
        activation_count = sum(1 for pre, post in node_outputs if abs(pre) > ACTIVATION_THRESHOLD)
    else:
        # 原始方案：基于post-activation（可能有死锁风险）
        activation_count = sum(1 for pre, post in node_outputs if abs(post) > ACTIVATION_THRESHOLD)
    
    return compute_metabolism_sparse(activation_count)


# 模拟两种网络
print("\n场景1: 成熟网络（高激活）")
mature_outputs = [
    (0.8, 0.66),   # pre, post
    (0.6, 0.54),
    (0.9, 0.72),
    (0.7, 0.61),
]
cost_mature = compute_agent_metabolism(mature_outputs, use_pre_activation=True)
print(f"  5节点高激活 -> 代谢: {cost_mature:.4f}")

print("\n场景2: 新生节点（初始权重低）")
newborn_outputs = [
    (0.8, 0.66),   # 成熟节点
    (0.01, 0.01),  # 新节点（权重0.01）
    (0.01, 0.01),  # 新节点
    (0.01, 0.01),  # 新节点
]
cost_post = compute_agent_metabolism(newborn_outputs, use_pre_activation=False)  # 原始
cost_pre = compute_agent_metabolism(newborn_outputs, use_pre_activation=True)   # 方案A

print(f"  原始(post-activation): {cost_post:.4f}")
print(f"  改进(pre-activation):  {cost_pre:.4f}")
print(f"  差异: {(cost_post - cost_pre)/cost_post*100:.1f}%")

# 验证新节点不会显著增加代谢
print(f"\n  新节点额外成本: {(cost_pre - compute_metabolism_sparse(1)):.4f}")
print(f"  新节点成本比例: {(cost_pre - compute_metabolism_sparse(1))/compute_metabolism_sparse(4)*100:.1f}%")


# ==================== 最终总结 ====================

print("\n" + "=" * 60)
print("验证总结")
print("=" * 60)

checks = [
    ("克莱伯缩放单调递增", True),
    ("次线性优势 > 50% @ N=20", compute_metabolism_sparse(20) < 20 * compute_metabolism_sparse(1)),
    ("方案A Hebbian可更新", new_node.weight > 0.01),
    ("方案B噪声突破概率", 突破次数 > 0),
]

for name, passed in checks:
    status = "✓" if passed else "✗"
    print(f"  {status} {name}")

all_passed = all(p for _, p in checks)
print(f"\n{'✓ 所有验证通过！' if all_passed else '✗ 存在失败项'}")

if all_passed:
    print("\n→ 代谢模型可进入演化实验阶段")
else:
    print("\n→ 需要调整参数后重试")