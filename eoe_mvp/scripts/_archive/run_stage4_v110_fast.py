#!/usr/bin/env python3
"""
阶段四 v11.0: 快速迭代测试

目标: 快速验证机制 + 迭代优化
- 更小的种群(10)
- 更短的寿命(30帧)
- 更少的食物(10个)
- 快速迭代
"""

import sys
import json

sys.stdout = sys.stderr
sys.path.insert(0, '.')

from core.eoe.population import Population


def main():
    n_rounds = 10  # 10轮快速迭代
    
    # 加载阶段三冠军
    try:
        with open('champions/stage3_champion.json') as f:
            current_brain = json.load(f)
        print(f"加载初始大脑: stage3_champion.json")
    except:
        current_brain = None
        print("警告: 使用随机初始化")
    
    print("=" * 60)
    print("阶段四 v11.0: 快速迭代模式")
    print("=" * 60)
    
    for round_num in range(1, n_rounds + 1):
        # 小种群 + 短寿命 = 快速迭代
        pop = Population(
            population_size=10,
            elite_ratio=0.4,     # 更高精英率
            env_width=40.0,
            env_height=40.0,
            lifespan=30,         # 短寿命
            n_food=10,
            food_energy=80.0,
            respawn_food=True,
            metabolic_alpha=0.003,
            metabolic_beta=0.003,
            seasonal_cycle=True,
            season_length=25,    # 短季节
            winter_food_multiplier=0.0,
            winter_metabolic_multiplier=1.2,
            immediate_eating=True,
            disable_food_escape=True,  # 禁用逃逸加速
            use_champion=True if current_brain else False,
            champion_brain=current_brain,
            pure_survival_mode=True,
            energy_decay_k=0.0001,
            port_interference_gamma=2.0,
            season_jitter=0.10,
            nest_tax=0.10
        )
        pop._init_population()
        
        pop.environment.enable_fatigue_system(
            enabled=True,
            max_fatigue=60.0,
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
        
        # 运行20代或存活者<2
        best_fit = 0
        best_brain = None
        
        for gen in range(20):
            pop.environment.apply_dynamic_pressure(pop.generation)
            stats = pop.epoch(verbose=False)
            
            alive = [a for a in pop.environment.agents if a.is_alive]
            if not alive:
                print(f"R{round_num} Gen{gen}: 全部死亡")
                break
            
            # 使用压力梯度熔炉选择真冠军
            true_champion = pop.select_true_champion(alive)
            total_stored = sum(a.food_stored for a in alive)
            
            # 更新英雄冢
            pop.update_hall_of_fame(true_champion)
            
            # 使用真冠军的适应度
            if true_champion.fitness > best_fit:
                best_fit = true_champion.fitness
                best_brain = true_champion.genome.to_dict()
            
            if gen % 5 == 0:
                env_p = pop._calculate_environmental_pressure()
                print(f"R{round_num} Gen{gen}: 存活{len(alive):2d} | 真冠军={true_champion.fitness:7.1f} | Stored={total_stored:3d} | 压力={env_p:.2f}")
            
            if len(alive) < 2:
                print(f"R{round_num}: 提前结束 (存活{len(alive)})")
                break
            
            pop.reproduce(verbose=False)
        
        # 保存本轮冠军
        if best_brain:
            with open(f'champions/stage4_v110_r{round_num}.json', 'w') as f:
                json.dump(best_brain, f, indent=2)
            current_brain = best_brain
            print(f"  → 保存冠军, fitness={best_fit:.1f}")
    
    print("\n" + "=" * 60)
    print("快速迭代完成")
    print("最终冠军: champions/stage4_v110_r10.json")
    print("=" * 60)


if __name__ == '__main__':
    main()