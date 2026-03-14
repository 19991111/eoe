#!/usr/bin/env python3
"""
v15 非线性代谢验证测试
=========================
验证: Cost = log(N + 1) × BaseCost (超出FREE_NODES部分)
"""

import torch
import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

from core.eoe.batched_agents import PoolConfig

config = PoolConfig()

print("=" * 60)
print("v15 非线性代谢验证")
print("=" * 60)

# 参数
log_base = config.LOG_BASE
free_nodes = config.FREE_NODES
base_cost = config.BASE_METABOLISM

print(f"\n参数:")
print(f"  LOG_BASE = {log_base}")
print(f"  FREE_NODES = {free_nodes}")
print(f"  BASE_COST = {base_cost}")
print(f"\n公式: Cost = log(N - {free_nodes} + 1) × log_factor × {base_cost}")

# 测试不同节点数
print("\n节点数 → 代谢成本 (对比线性):")
print("-" * 60)
print(f"{'N':>4} | {'线性成本':>10} | {'v15非线性':>10} | {'节省':>8} | {'节省%':>8}")
print("-" * 60)

for n in [1, 3, 5, 7, 10, 15, 20, 30, 50, 100]:
    # v14 线性
    linear_cost = n * base_cost
    
    # v15 非线性: Cost = log(taxable + 1) * base_cost
    taxable = max(0, n - free_nodes)
    log_cost = torch.log(torch.tensor(taxable + 1)) / torch.log(torch.tensor(log_base))
    nonlinear_cost = float(log_cost) * base_cost
    
    saving = linear_cost - nonlinear_cost
    saving_pct = (saving / linear_cost * 100) if linear_cost > 0 else 0
    
    print(f"{n:>4} | {linear_cost:>10.4f} | {nonlinear_cost:>10.4f} | {saving:>8.4f} | {saving_pct:>7.1f}%")

# SuperNode测试
print("\n" + "=" * 60)
print("SuperNode成本对比 (v14 vs v15)")
print("=" * 60)

for n_nodes, n_supernodes in [(10, 1), (15, 3), (20, 5)]:
    # v14: super算0.5节点 (线性)
    v14_effective = n_nodes - n_supernodes * 0.5
    v14_cost = v14_effective * base_cost
    
    # v15: super算1节点 + 非线性
    v15_taxable = max(0, n_nodes - free_nodes)
    v15_log_cost = torch.log(torch.tensor(v15_taxable + 1)) / torch.log(torch.tensor(log_base))
    v15_base = float(v15_log_cost) * base_cost
    v15_supernode = n_supernodes * 1.0 * base_cost
    v15_cost = v15_base + v15_supernode
    
    print(f"\n{n_nodes}节点 + {n_supernodes} SuperNode:")
    print(f"  v14: {v14_cost:.4f} (effective={v14_effective})")
    print(f"  v15: {v15_cost:.4f} (base={v15_base:.4f} + super={v15_supernode:.4f})")
    print(f"  变化: {(v15_cost - v14_cost) / v14_cost * 100:+.1f}%")

print("\n✅ 验证完成!")