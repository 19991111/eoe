#!/usr/bin/env python3
"""
能量审计测试脚本
================
用随机乱跑的初始种群空转10000步,验证系统能量是否守恒

运行方式:
    PYTHONPATH=. python scripts/test_energy_audit.py
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import os
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import numpy as np
import time

from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.energy_audit import EnergyAuditHook, create_energy_audit_hook


def run_energy_audit_test(
    steps: int = 10000,
    initial_pop: int = 50,
    device: str = 'cuda:0'
):
    """运行能量守恒测试"""
    
    print("=" * 60)
    print("🔋 EOE 能量守恒测试")
    print("=" * 60)
    print(f"  步数:     {steps}")
    print(f"  初始种群: {initial_pop}")
    print(f"  设备:     {device}")
    print("=" * 60)
    
    # 1. 创建环境 (简化配置,只开必要的)
    env = EnvironmentGPU(
        width=100.0,
        height=100.0,
        resolution=1.0,
        device=device,
        energy_field_enabled=True,      # EPF场
        impedance_field_enabled=False,
        stigmergy_field_enabled=False,   # ISF场 (可选)
        flickering_energy_enabled=False,  # 暂时关闭闪烁
        danger_field_enabled=False,
        matter_grid_enabled=False,
        wind_field_enabled=False,
    )
    
    # 2. 创建配置 (使用最小配置)
    config = PoolConfig()
    config.HEBBIAN_ENABLED = False       # 关闭学习,简化
    config.NONLINEAR_METABOLISM = True
    config.SPARSE_ACTIVATION = True
    
    # 3. 创建 Agent 池
    max_agents = min(initial_pop * 3, 80)
    
    agents = BatchedAgents(
        initial_population=initial_pop,
        max_agents=max_agents,
        env_width=100.0,
        env_height=100.0,
        device=device,
        init_energy=50.0,
        config=config,
        env=env
    )
    
    # 使用寒武纪初始化
    genomes = agents._create_cambrian_genomes(agents.alive_mask.sum().item())
    agents.set_brains(genomes)
    
    # 4. 创建能量审计钩子 (标准模式)
    audit_hook = create_energy_audit_hook("standard", device=device)
    
    # 初始化审计 (记录初始能量)
    audit_hook.initialize(env, agents)
    
    print(f"\n🚀 开始模拟...")
    start_time = time.time()
    
    # 5. 运行模拟
    for step in range(steps):
        # 环境步进
        env.step()
        
        # Agent 步进
        step_stats = agents.step(env=env, dt=0.1)
        
        # 执行能量审计 (每 audit_interval 步)
        # 传递死亡agent能量用于追踪
        dead_agent_energies = None
        if 'dead_indices' in step_stats:
            dead_indices = step_stats['dead_indices']
            if len(dead_indices) > 0:
                dead_agent_energies = agents.state.energies[dead_indices]
        
        result = audit_hook.audit(
            env, agents, step + 1,
            dead_agent_energies=dead_agent_energies
        )
        
        # 定期报告
        if (step + 1) % 2000 == 0:
            elapsed = time.time() - start_time
            speed = (step + 1) / elapsed
            n_alive = agents.alive_mask.sum().item()
            
            print(f"  Step {step+1:5d} | "
                  f"存活: {n_alive:3d} | "
                  f"速度: {speed:.1f}步/秒")
        
        # 种群灭绝则停止
        if agents.alive_mask.sum().item() == 0:
            print(f"\n⚠️ 种群在 Step {step+1} 灭绝!")
            break
    
    elapsed = time.time() - start_time
    
    # 6. 打印审计总结
    print(f"\n模拟完成: {step+1} 步, 耗时 {elapsed:.1f}秒")
    audit_hook.print_summary()
    
    # 7. 返回结果
    stats = audit_hook.get_statistics()
    
    if stats['failed_audits'] == 0:
        print("\n✅ 测试通过: 能量守恒验证成功!")
        return True
    else:
        print(f"\n❌ 测试失败: {stats['failed_audits']} 次守恒破缺")
        print(f"   最大相对误差: {stats['max_relative_error']:.2e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EOE 能量守恒测试")
    parser.add_argument("--steps", type=int, default=10000, help="模拟步数")
    parser.add_argument("--pop", type=int, default=50, help="初始种群")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--strict", action="store_true", help="严格模式")
    
    args = parser.parse_args()
    
    success = run_energy_audit_test(
        steps=args.steps,
        initial_pop=args.pop,
        device=args.device
    )
    
    sys.exit(0 if success else 1)