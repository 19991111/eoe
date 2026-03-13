#!/usr/bin/env python3
"""
阶段四 v11.0: 红皇后竞争 (Red Queen Competition)

核心策略: 高压筛选
- v11.0机制制造极端生存压力
- 96%+死亡率是正常的（筛选最优秀个体）
- 存活者大脑保存并用于下一轮初始化

运行策略:
- 每轮40代（或存活者<3时提前结束）
- 保存每轮冠军大脑
- 用冠军大脑初始化下一轮
"""

import sys
import json
import numpy as np

# 强制unbuffered输出
sys.stdout = sys.stderr

sys.path.insert(0, '.')

from core.eoe.population import Population


def load_champion(path):
    """加载冠军大脑"""
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None


def save_champion(brain_data, path):
    """保存冠军大脑"""
    with open(path, 'w') as f:
        json.dump(brain_data, f, indent=2)


def run_round(pop, round_num, max_gens=40):
    """运行一轮演化"""
    print(f"\n{'='*60}")
    print(f"Round {round_num}: 开始演化")
    print(f"{'='*60}")
    
    best_fitness_overall = 0
    best_gen = 0
    round_best_brain = None
    
    for gen in range(max_gens):
        # 应用动态环境压力 (包含v11.0季节波动)
        pop.environment.apply_dynamic_pressure(pop.generation)
        
        # 运行一代
        stats = pop.epoch(verbose=False)
        
        # 获取存活Agent
        agents = pop.environment.agents
        alive_agents = [a for a in agents if a.is_alive]
        
        if not alive_agents:
            print(f"Gen {gen}: 全部死亡!")
            break
        
        # 统计
        best = max(alive_agents, key=lambda a: a.fitness)
        total_stored = sum(a.food_stored for a in alive_agents)
        total_eaten = sum(a.food_eaten for a in alive_agents)
        
        if best.fitness > best_fitness_overall:
            best_fitness_overall = best.fitness
            best_gen = gen
            round_best_brain = best.genome.to_dict()
        
        # 每5代输出
        if gen % 5 == 0 or gen == max_gens - 1:
            print(f"Gen {gen:3d} | "
                  f"Alive: {len(alive_agents):2d} | "
                  f"Best Fit: {best.fitness:8.1f} | "
                  f"Stored: {total_stored:3d} | "
                  f"Eaten: {total_eaten:3d}")
        
        # 存活者少于2时提前结束
        if len(alive_agents) < 2:
            print(f"存活者仅剩{len(alive_agents)}人，提前结束")
            break
        
        # 繁殖下一代
        pop.reproduce(verbose=False)
    
    # 保存本轮冠军
    if round_best_brain:
        champ_path = f'champions/stage4_v110_round{round_num}_champ.json'
        save_champion(round_best_brain, champ_path)
        print(f"\n本轮冠军已保存: {champ_path}")
        print(f"  最高适应度: {best_fitness_overall:.1f} (Gen {best_gen})")
    
    return round_best_brain, best_fitness_overall


def main():
    n_rounds = 5  # 运行5轮筛选
    
    # 加载阶段三冠军作为起始
    brain_path = 'champions/stage3_champion.json'
    current_brain = load_champion(brain_path)
    if current_brain:
        print(f"加载初始大脑: {brain_path}")
    else:
        print("警告: 无法加载阶段三冠军，使用随机初始化")
    
    print("=" * 60)
    print("阶段四 v11.0: 红皇后竞争 - 高压筛选模式")
    print("=" * 60)
    print("v11.0机制: 代谢熵增 + 端口干涉 + 季节波动")
    print("策略: 每轮筛选存活冠军，用于下一轮初始化")
    print("=" * 60)
    
    for round_num in range(1, n_rounds + 1):
        # 创建种群
        pop = Population(
            population_size=15,  # 较小种群加快筛选
            elite_ratio=0.3,     # 较高精英率
            env_width=50.0,
            env_height=50.0,
            lifespan=60,
            n_food=15,
            food_energy=80.0,
            respawn_food=True,
            metabolic_alpha=0.003,
            metabolic_beta=0.003,
            seasonal_cycle=True,
            season_length=35,
            winter_food_multiplier=0.0,
            winter_metabolic_multiplier=1.2,
            immediate_eating=True,
            disable_food_escape=False,
            use_champion=True if current_brain else False,
            champion_brain=current_brain,
            pure_survival_mode=True,
            # v11.0 新机制参数
            energy_decay_k=0.0001,
            port_interference_gamma=2.0,
            season_jitter=0.10,
            nest_tax=0.10
        )
        pop._init_population()
        
        # 启用阶段三特性
        pop.environment.enable_fatigue_system(
            enabled=True,
            max_fatigue=80.0,
            fatigue_build_rate=0.15,
            sleep_danger_prob=0.95,
            enable_wakeup_hunger=True,
            enable_sleep_drop=True
        )
        
        pop.environment.enable_thermal_sanctuary(
            enabled=True,
            summer_temp=28.0,
            winter_temp=-10.0,
            food_heat=15.0,
            nest_insulation=0.02
        )
        
        pop.environment.food_escape_enabled = True
        
        print(f"\nv11.0机制配置:")
        print(f"  能量衰减k: {pop.environment.energy_decay_k}")
        print(f"  端口干涉γ: {pop.environment.port_interference_gamma}")
        print(f"  季节波动率: {pop.environment.season_jitter}")
        print(f"  入库税率: {pop.environment.nest_tax}")
        
        # 运行本轮
        round_brain, round_best = run_round(pop, round_num, max_gens=40)
        
        if round_brain:
            current_brain = round_brain
        else:
            print("本轮无存活者，使用上一轮冠军继续")
    
    # 最终总结
    print("\n" + "=" * 60)
    print("阶段四 v11.0 实验完成")
    print("=" * 60)
    
    # 保存最终冠军
    if current_brain:
        save_champion(current_brain, 'champions/stage4_v110_final_champion.json')
        print("最终冠军: champions/stage4_v110_final_champion.json")


if __name__ == '__main__':
    main()