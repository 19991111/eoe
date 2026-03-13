#!/usr/bin/env python3
"""
v0.0 GPU 仿真主入口
====================
统一、简洁的仿真启动脚本

Usage:
    # 命令行运行
    python main_v13_gpu.py --agents 1000 --steps 1500
    
    # 代码中使用
    from main_v13_gpu import run_simulation
    history = run_simulation(n_agents=1000, lifespan=1500)
"""

import argparse
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.eoe.batch import Simulation


def run_simulation(
    n_agents: int = 100,
    lifespan: int = 1500,
    env_size: float = 100.0,
    device: str = 'cuda:0',
    verbose: bool = True,
    save_history: str = None
):
    """
    运行GPU仿真
    
    Args:
        n_agents: Agent数量
        lifespan: 生命周期步数
        env_size: 环境大小
        device: 计算设备
        verbose: 输出详情
        save_history: 保存历史路径
        
    Returns:
        List[SimState]: 仿真历史
    """
    sim = Simulation(
        n_agents=n_agents,
        env_width=env_size,
        env_height=env_size,
        lifespan=lifespan,
        device=device
    )
    
    history = sim.run(verbose=verbose)
    
    if save_history:
        import json
        data = {
            'config': {
                'n_agents': n_agents,
                'lifespan': lifespan,
                'env_size': env_size,
                'device': device
            },
            'history': [
                {
                    'step': s.step,
                    'alive': s.alive_count,
                    'energy': s.total_energy,
                    'mean_energy': s.mean_energy
                }
                for s in history
            ]
        }
        with open(save_history, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 历史已保存到: {save_history}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description='v0.0 GPU 仿真')
    
    parser.add_argument('--agents', type=int, default=100, help='Agent数量')
    parser.add_argument('--steps', type=int, default=1500, help='生命周期步数')
    parser.add_argument('--size', type=float, default=100.0, help='环境大小')
    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备')
    parser.add_argument('--save', type=str, default=None, help='保存历史')
    parser.add_argument('--quiet', action='store_true', help='静默模式')
    
    args = parser.parse_args()
    
    run_simulation(
        n_agents=args.agents,
        lifespan=args.steps,
        env_size=args.size,
        device=args.device,
        verbose=not args.quiet,
        save_history=args.save
    )


if __name__ == '__main__':
    main()