"""
v0.0 四场协同压力测试 (Co-field Stress Test)

三个验证场景:
A. 路径最优化 (EPF + KIF + ISF)
B. 季节性信号套利 (EPF + ESF + ISF)  
C. 防御性领地化 (EPF + KIF + ISF)

运行示例:
    python scripts/test_v13_cofield_stress.py --scenario A
    python scripts/test_v13_cofield_stress.py --scenario B
    python scripts/test_v13_cofield_stress.py --scenario C
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.eoe.environment import Environment, EnergySource
from core.eoe.agent import Agent
from core.eoe.population import Population


def setup_scenario_A() -> dict:
    """
    场景 A: 路径最优化实验 (EPF + KIF + ISF)
    
    设置:
    - 高能源在 (80, 50)
    - 路径1 (直线): 高阻抗 (Z=50)
    - 路径2 (迂回): 低阻抗 (Z=1)
    """
    print("=" * 60)
    print("场景 A: 路径最优化 (EPF + KIF + ISF)")
    print("=" * 60)
    print("目标: 智能体应选择低阻抗迂回路，形成ISF高速公路")
    
    # 创建环境
    env = Environment(
        width=100, height=100,
        energy_field_enabled=True,
        field_resolution=2.0,
        field_diffusion_rate=0.1,
        field_decay_rate=0.001,
        field_initial_energy=0.1,
        # 禁用其他场
        impedance_field_enabled=False,
        stigmergy_field_enabled=False,
        stress_field_enabled=False,
        n_food=0,
        metabolic_alpha=0.001,
        metabolic_beta=0.01,
    )
    
    # 设置单一高能源 (80, 50)
    env.energy_field.sources = [
        EnergySource(80, 50, injection_rate=2.0, radius=20)
    ]
    env.energy_field.field.fill(0.1)
    
    # 手动设置阻抗场 (路径1高阻抗，路径2低阻抗)
    env.impedance_field_enabled = True
    from core.eoe.kinetic_impedance import KineticImpedanceField
    env.impedance_field = KineticImpedanceField(
        width=100, height=100,
        resolution=2.0,
        seed=42,
        noise_scale=0.0,  # 关闭噪声
        obstacle_density=0.0
    )
    from core.eoe.kinetic_impedance import KineticImpedanceLaw
    env.kinetic_impedance_law = KineticImpedanceLaw(
        move_cost_coeff=0.1,
        repulsion_coeff=0.0
    )
    
    # 手动设置高阻抗区域 (路径1: 直线)
    # 从 (10, 50) 到 (70, 50) 的矩形区域 = 高阻抗
    for ix in range(20, 70):
        for iy in range(40, 60):
            if 0 <= ix < env.impedance_field.grid_width and 0 <= iy < env.impedance_field.grid_height:
                env.impedance_field.field[ix, iy] = 50.0  # 高阻抗
    
    # 低阻抗路径2已由base_impedance=1.0处理
    
    # 启用压痕场
    env.stigmergy_field_enabled = True
    from core.eoe.stigmergy_field import StigmergyField, StigmergyLaw
    env.stigmergy_field = StigmergyField(
        width=100, height=100,
        resolution=2.0,
        diffusion_rate=0.1,
        decay_rate=0.98,
        base_deposit=0.2,
        signal_energy_cost=0.01
    )
    env.stigmergy_law = StigmergyLaw(
        base_deposit=0.2,
        signal_energy_cost=0.01
    )
    
    return {
        'env': env,
        'name': 'Scenario A: Path Optimization',
        'description': '高能源(80,50)，路径1高阻抗，路径2低阻抗'
    }


def setup_scenario_B() -> dict:
    """
    场景 B: 季节性信号套利 (EPF + ESF + ISF)
    
    设置:
    - ESF 冬季: 代谢翻倍，能量源输出减半
    - ISF 信号代价中等
    """
    print("=" * 60)
    print("场景 B: 季节性信号套利 (EPF + ESF + ISF)")
    print("=" * 60)
    print("目标: 观察高压期智能体减少探索，先驱者释放信号引导同伴")
    
    env = Environment(
        width=100, height=100,
        energy_field_enabled=True,
        field_resolution=2.0,
        field_diffusion_rate=0.1,
        field_decay_rate=0.001,
        field_initial_energy=0.1,
        # 禁用KIF
        impedance_field_enabled=False,
        # 启用ISF
        stigmergy_field_enabled=True,
        stigmergy_diffusion_rate=0.1,
        stigmergy_decay_rate=0.98,
        stigmergy_signal_cost=0.02,
        # 启用ESF (应力场)
        stress_field_enabled=True,
        stress_resolution=4.0,
        stress_temp_period=100,  # 短周期便于观察
        n_food=0,
        metabolic_alpha=0.002,
        metabolic_beta=0.02,
    )
    
    # 设置能量源
    env.energy_field.sources = [
        EnergySource(30, 30, injection_rate=0.8, radius=15),
        EnergySource(70, 70, injection_rate=0.8, radius=15),
    ]
    env.energy_field.field.fill(0.1)
    
    return {
        'env': env,
        'name': 'Scenario B: Seasonal Arbitrage',
        'description': 'ESF冬季高压，ISF引导向残余能量点'
    }


def setup_scenario_C() -> dict:
    """
    场景 C: 防御性领地化 (EPF + KIF + ISF)
    
    设置:
    - KIF 闭环围墙，只留小缺口
    - 围墙内高浓度EPF
    """
    print("=" * 60)
    print("场景 C: 防御性领地化 (EPF + KIF + ISF)")
    print("=" * 60)
    print("目标: 观察缺口处ISF堆积，验证渗透率自动切换")
    
    env = Environment(
        width=100, height=100,
        energy_field_enabled=True,
        field_resolution=2.0,
        field_diffusion_rate=0.1,
        field_decay_rate=0.001,
        field_initial_energy=0.1,
        # 启用KIF
        impedance_field_enabled=True,
        impedance_resolution=2.0,
        impedance_noise_scale=0.0,
        impedance_obstacle_density=0.0,
        # 启用ISF
        stigmergy_field_enabled=True,
        stigmergy_diffusion_rate=0.1,
        stigmergy_decay_rate=0.98,
        stigmergy_signal_cost=0.01,
        # 禁用ESF
        stress_field_enabled=False,
        n_food=0,
        metabolic_alpha=0.002,
        metabolic_beta=0.02,
    )
    
    # 设置围墙 (闭环方形，只留一个小缺口)
    # 围墙位置: (30,30) 到 (70,70)，缺口在右边中间
    wall_x1, wall_y1 = 30, 30
    wall_x2, wall_y2 = 70, 70
    gap_y = 55  # 缺口位置
    
    for ix in range(env.impedance_field.grid_width):
        for iy in range(env.impedance_field.grid_height):
            wx = ix * env.impedance_field.resolution
            wy = iy * env.impedance_field.resolution
            
            # 上墙
            if wall_y1 <= wy <= wall_y1 + 5 and wall_x1 <= wx <= wall_x2:
                env.impedance_field.field[ix, iy] = 100.0
            # 下墙
            if wall_y2 - 5 <= wy <= wall_y2 and wall_x1 <= wx <= wall_x2:
                if not (wall_x2 <= wx <= wall_x2 + 10 and gap_y - 5 <= wy <= gap_y + 5):  # 缺口
                    env.impedance_field.field[ix, iy] = 100.0
            # 左墙
            if wall_x1 <= wx <= wall_x1 + 5 and wall_y1 <= wy <= wall_y2:
                env.impedance_field.field[ix, iy] = 100.0
            # 右墙 (除缺口外)
            if wall_x2 - 5 <= wx <= wall_x2 and wall_y1 <= wy <= wall_y2:
                if not (gap_y - 5 <= wy <= gap_y + 5):  # 缺口
                    env.impedance_field.field[ix, iy] = 100.0
    
    # 围墙内设置高能源
    env.energy_field.sources = [
        EnergySource(50, 50, injection_rate=3.0, radius=15)
    ]
    env.energy_field.field.fill(0.1)
    
    return {
        'env': env,
        'name': 'Scenario C: Defensive Territorialization',
        'description': 'KIF闭环围墙，内置高能EPF，缺口处观察ISF堆积'
    }


def run_simulation(env: Environment, n_agents: int = 20, n_steps: int = 500):
    """运行模拟并记录数据"""
    print(f"\n初始化 {n_agents} 个智能体...")
    
    # 创建初始种群
    agents = []
    for i in range(n_agents):
        # 从左侧进入
        agent = Agent(
            agent_id=i,
            x=10 + np.random.uniform(-5, 5),
            y=50 + np.random.uniform(-10, 10),
            theta=0.0
        )
        agent.internal_energy = 150.0
        agents.append(agent)
        env.add_agent(agent)
    
    # 记录数据
    fitness_history = []
    energy_history = []
    alive_history = []
    
    print(f"运行 {n_steps} 步模拟...")
    
    for step in range(n_steps):
        env.step()
        
        # 记录
        alive = sum(1 for a in env.agents if a.is_alive)
        avg_energy = np.mean([a.internal_energy for a in env.agents if a.is_alive]) if alive > 0 else 0
        avg_fitness = np.mean([a.fitness for a in env.agents if a.is_alive]) if alive > 0 else 0
        
        fitness_history.append(avg_fitness)
        energy_history.append(avg_energy)
        alive_history.append(alive)
        
        if step % 50 == 0:
            print(f"  Step {step}: alive={alive}, energy={avg_energy:.1f}, fitness={avg_fitness:.1f}")
    
    return {
        'fitness': fitness_history,
        'energy': energy_history,
        'alive': alive_history,
        'final_positions': [(a.x, a.y) for a in env.agents if a.is_alive]
    }


def visualize_results(scenario: dict, results: dict):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(scenario['name'])
    
    env = scenario['env']
    
    # 1. 能量场热力图
    ax1 = axes[0, 0]
    if env.energy_field:
        im1 = ax1.imshow(env.energy_field.get_heatmap_data(), origin='lower', cmap='hot')
        ax1.set_title('Energy Field (EPF)')
        plt.colorbar(im1, ax=ax1)
    
    # 2. 阻抗场热力图
    ax2 = axes[0, 1]
    if env.impedance_field:
        im2 = ax2.imshow(env.impedance_field.get_heatmap_data(), origin='lower', cmap='Blues')
        ax2.set_title('Impedance Field (KIF)')
        plt.colorbar(im2, ax=ax2)
    
    # 3. 压痕场热力图
    ax3 = axes[1, 0]
    if env.stigmergy_field:
        im3 = ax3.imshow(env.stigmergy_field.get_heatmap_data(), origin='lower', cmap='Greens')
        ax3.set_title('Stigmergy Field (ISF)')
        plt.colorbar(im3, ax=ax3)
        # 标记智能体最终位置
        for x, y in results['final_positions']:
            ax3.plot(x, y, 'r.', markersize=3, alpha=0.5)
    
    # 4. 适应度曲线
    ax4 = axes[1, 1]
    ax4.plot(results['fitness'], label='Avg Fitness', alpha=0.7)
    ax4.plot(results['energy'], label='Avg Energy', alpha=0.7)
    ax4.plot(results['alive'], label='Alive Count', alpha=0.7)
    ax4.set_title('Simulation Metrics')
    ax4.set_xlabel('Step')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"v13_cofield_{scenario['name'].split(':')[0].replace(' ', '_')}.png", dpi=150)
    print(f"保存图表: v13_cofield_{scenario['name'].split(':')[0].replace(' ', '_')}.png")


def main():
    parser = argparse.ArgumentParser(description='v0.0 四场协同压力测试')
    parser.add_argument('--scenario', type=str, default='A', 
                       choices=['A', 'B', 'C', 'ALL'],
                       help='测试场景: A/B/C/ALL')
    parser.add_argument('--agents', type=int, default=20, help='智能体数量')
    parser.add_argument('--steps', type=int, default=500, help='模拟步数')
    
    args = parser.parse_args()
    
    scenarios = []
    if args.scenario in ['A', 'ALL']:
        scenarios.append(('A', setup_scenario_A))
    if args.scenario in ['B', 'ALL']:
        scenarios.append(('B', setup_scenario_B))
    if args.scenario in ['C', 'ALL']:
        scenarios.append(('C', setup_scenario_C))
    
    for scenario_id, setup_fn in scenarios:
        scenario = setup_fn()
        results = run_simulation(scenario['env'], args.agents, args.steps)
        visualize_results(scenario, results)
        print(f"\n场景 {scenario_id} 完成!")
    
    print("\n✅ 所有场景测试完成!")


if __name__ == "__main__":
    main()