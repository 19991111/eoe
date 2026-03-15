#!/usr/bin/env python
"""
能量守恒完整测试
================
使用基础稳定配置验证EOE物理引擎的能量守恒

配置:
- 环境: v14风格基础能量场 (静止,无闪烁)
- 代谢: 标准配置
- 审计: strict模式 (每100步)
- 步数: 5000步
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import torch
import time
import os

os.environ['PYTHONUNBUFFERED'] = '1'

from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.energy_audit import EnergyAuditHook


def run_full_audit_test(
    steps: int = 5000,
    initial_pop: int = 50,
    device: str = 'cuda:0'
):
    """运行完整能量守恒测试"""
    
    print("=" * 70)
    print("🔋 EOE 完整能量守恒测试 (v14基础配置)")
    print("=" * 70)
    print(f"  步数:     {steps}")
    print(f"  初始种群: {initial_pop}")
    print(f"  设备:     {device}")
    print(f"  审计模式: strict (每100步)")
    print("=" * 70)
    
    # ===== 1. 基础环境配置 =====
    env = EnvironmentGPU(
        width=100.0,
        height=100.0,
        resolution=1.0,
        device=device,
        # 基础场
        energy_field_enabled=True,      # EPF能量场
        impedance_field_enabled=False,   # 关闭阻抗场
        stigmergy_field_enabled=False,   # 关闭压痕场
        flickering_energy_enabled=False, # 关闭闪烁(静止能量源)
        danger_field_enabled=False,      # 关闭危险场
        matter_grid_enabled=False,       # 关闭物质场
        wind_field_enabled=False,        # 关闭风场
    )
    print("✅ 环境初始化完成 (基础静止能量场)")
    
    # ===== 2. 标准代谢配置 =====
    config = PoolConfig()
    
    # 基础代谢 - 适中,让种群能存活
    config.BASE_METABOLISM = 0.02
    config.BASAL_COST = 0.02
    
    # 学习 - 保持开启
    config.HEBBIAN_ENABLED = True
    config.HEBBIAN_BASE_LR = 0.01
    
    # 代谢模型
    config.NONLINEAR_METABOLISM = True
    config.SPARSE_ACTIVATION = True
    config.LOG_BASE = 2.0
    config.FREE_NODES = 5
    
    # 繁衍
    config.REPRODUCE_ENABLED = True
    config.REPRODUCE_THRESHOLD = 60.0   # 60能量可繁衍
    config.REPRODUCE_COST = 30.0        # 繁衍消耗30
    config.REPRODUCE_OFFSPRING_ENERGY = 20.0  # 子代初始20
    
    # 关闭复杂机制
    config.AGE_ENABLED = False          # 关闭年龄
    config.T_MAZE_ENABLED = False       # 关闭T迷宫
    
    print("✅ 配置加载完成 (标准代谢+繁衍)")
    
    # ===== 3. 创建Agent池 =====
    max_agents = min(initial_pop * 4, 100)
    
    agents = BatchedAgents(
        initial_population=initial_pop,
        max_agents=max_agents,
        env_width=100.0,
        env_height=100.0,
        device=device,
        init_energy=50.0,        # 初始能量
        config=config,
        env=env
    )
    
    # 寒武纪初始化
    genomes = agents._create_cambrian_genomes(agents.alive_mask.sum().item())
    agents.set_brains(genomes)
    
    print(f"✅ Agent池初始化完成 (种群: {initial_pop})")
    
    # ===== 4. 创建能量审计 (strict模式) =====
    audit = EnergyAuditHook(
        tolerance=1e-5,        # 1e-5 严格阈值
        audit_interval=100,    # 每100步审计
        device=device,
        verbose=True
    )
    
    # 初始化审计
    initial_snapshot = audit.initialize(env, agents)
    print(f"✅ 能量审计初始化")
    print(f"   初始总能量: {initial_snapshot.total:.6f}")
    
    print("\n" + "-" * 70)
    print("🚀 开始模拟...")
    print("-" * 70)
    
    start_time = time.time()
    last_print = 0
    
    # ===== 5. 主循环 =====
    for step in range(steps):
        # 环境步进
        env.step()
        
        # Agent步进
        step_stats = agents.step(env=env, dt=0.1)
        
        # 获取死亡信息
        dead_agent_energies = None
        if 'dead_indices' in step_stats and len(step_stats['dead_indices']) > 0:
            dead_indices = step_stats['dead_indices']
            dead_agent_energies = agents.state.energies[dead_indices]
        
        # 执行能量审计
        result = audit.audit(
            env, agents, step + 1,
            dead_agent_energies=dead_agent_energies
        )
        
        # 定期报告
        if (step + 1) % 500 == 0 or step + 1 == steps:
            elapsed = time.time() - start_time
            speed = (step + 1) / elapsed if elapsed > 0 else 0
            n_alive = agents.alive_mask.sum().item()
            
            # 获取最新能量快照
            current = audit.snapshots[-1]
            
            print(f"  Step {step+1:5d} | "
                  f"存活: {n_alive:3d} | "
                  f"总能量: {current.total:.4f} | "
                  f"速度: {speed:.1f}步/秒")
            
            # 检查种群状态
            if n_alive == 0:
                print(f"\n⚠️ 种群在 Step {step+1} 灭绝!")
                break
        
        # 检查是否需要停止
        if agents.alive_mask.sum().item() == 0:
            break
    
    elapsed = time.time() - start_time
    
    # ===== 6. 结果报告 =====
    print("\n" + "=" * 70)
    print("📊 模拟完成")
    print("=" * 70)
    print(f"  运行步数:   {step + 1}")
    print(f"  存活人数:   {agents.alive_mask.sum().item()}")
    print(f"  模拟耗时:   {elapsed:.1f}秒")
    print(f"  平均速度:   {(step+1)/elapsed:.1f}步/秒")
    
    # 打印审计总结
    audit.print_summary()
    
    # 获取统计数据
    stats = audit.get_statistics()
    
    # 返回结果
    return {
        'success': stats['failed_audits'] == 0,
        'stats': stats,
        'final_step': step + 1,
        'final_pop': agents.alive_mask.sum().item()
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EOE 完整能量守恒测试")
    parser.add_argument("--steps", type=int, default=5000, help="模拟步数")
    parser.add_argument("--pop", type=int, default=50, help="初始种群")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    
    args = parser.parse_args()
    
    result = run_full_audit_test(
        steps=args.steps,
        initial_pop=args.pop,
        device=args.device
    )
    
    print("\n" + "=" * 70)
    if result['success']:
        print("✅ 测试通过: 能量守恒验证成功!")
    else:
        print(f"❌ 测试失败: {result['stats']['failed_audits']} 次守恒破缺")
    print("=" * 70)