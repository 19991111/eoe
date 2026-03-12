#!/usr/bin/env python3
"""
阶段三: 高级生理与微观管理 (Advanced Physiology & Micro-management)

目标: 演化出"自我调节 (Homeostasis)"的智慧

考核指标:
1. 劳逸结合 (Work-Rest Cycle) - 主动休息
2. 趋温性行为 (Thermotaxis) - 冬天在庇护所
3. 代谢精算 (Metabolic Accounting) - 囤脂肪

使用阶段二冠军脑初始化
"""

import sys
import json
import numpy as np
sys.path.insert(0, '.')

from core.eoe.population import Population


def main():
    # 加载阶段二冠军大脑
    brain_path = 'champions/stage1_best_brain.json'
    with open(brain_path) as f:
        brain_data = json.load(f)
    
    print("=" * 60)
    print("阶段三: 高级生理与微观管理")
    print("=" * 60)
    print("启用: 疲劳系统 + 热力学庇护所")
    print("目标: 自我调节智慧")
    print("=" * 60)
    
    # 创建种群
    pop = Population(
        population_size=20,
        elite_ratio=0.25,
        env_width=50.0,
        env_height=50.0,
        lifespan=80,
        n_food=20,
        food_energy=80.0,
        respawn_food=True,
        metabolic_alpha=0.003,
        metabolic_beta=0.003,
        seasonal_cycle=True,
        season_length=35,
        winter_food_multiplier=0.0,
        winter_metabolic_multiplier=1.0,
        immediate_eating=True,
        disable_food_escape=False,
        use_champion=True,
        champion_brain=brain_data,
        pure_survival_mode=True
    )
    pop._init_population()
    
    # 启用阶段三特性
    pop.environment.enable_fatigue_system(
        enabled=True,
        max_fatigue=80.0,
        fatigue_build_rate=0.15,  # 较低，避免死亡螺旋
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
    
    print(f"\n环境配置:")
    print(f"  疲劳系统: {pop.environment.fatigue_system_enabled}")
    print(f"  热力学庇护所: {pop.environment.thermal_sanctuary_enabled}")
    print(f"  疲劳积累率: {pop.environment.fatigue_build_rate}")
    
    # 记录历史
    history = {
        'stored': [],
        'eaten': [],
        'winter_in_sanctuary': [],
        'avg_fatigue': [],
        'avg_speed': []
    }
    
    max_gen = 60
    
    for gen in range(max_gen):
        # 渐进式增加冬天难度
        if gen == 20:
            pop.environment.winter_metabolic_multiplier = 1.15
        elif gen == 40:
            pop.environment.winter_metabolic_multiplier = 1.3
        
        winter_in_sanctuary = []
        avg_fatigue = []
        avg_speed = []
        
        for step in range(pop.lifespan):
            pop.environment.step()
            
            # 统计
            if pop.environment.current_season == 'winter':
                # 冬天在庇护所的Agent数 (默认在中心)
                nest_x = getattr(pop.environment, 'nest_x', pop.environment.width / 2)
                nest_y = getattr(pop.environment, 'nest_y', pop.environment.height / 2)
                nest_r = getattr(pop.environment, 'nest_radius', 8.0)
                
                in_sanct = sum(1 for a in pop.agents if a.is_alive and 
                    np.sqrt((a.x - nest_x)**2 + (a.y - nest_y)**2) < nest_r)
                winter_in_sanctuary.append(in_sanct)
            
            # 平均疲劳和速度
            for a in pop.agents:
                if a.is_alive:
                    avg_fatigue.append(a.fatigue)
                    speed = np.sqrt(a.vx**2 + a.vy**2) if hasattr(a, 'vx') else 0
                    avg_speed.append(speed)
        
        stored = sum(a.food_stored for a in pop.agents)
        eaten = sum(a.food_eaten for a in pop.agents)
        
        avg_w_inter = np.mean(winter_in_sanctuary) if winter_in_sanctuary else 0
        avg_fat = np.mean(avg_fatigue) if avg_fatigue else 0
        avg_spd = np.mean(avg_speed) if avg_speed else 0
        
        history['stored'].append(stored)
        history['eaten'].append(eaten)
        history['winter_in_sanctuary'].append(avg_w_inter)
        history['avg_fatigue'].append(avg_fat)
        history['avg_speed'].append(avg_spd)
        
        if gen % 10 == 0:
            print(f"\nGen {gen:2d}:")
            print(f"  贮粮: {sum(history['stored']):3d} | 进食: {sum(history['eaten']):3d}")
            print(f"  冬天庇护所: {avg_w_inter:.1f} agents")
            print(f"  平均疲劳: {avg_fat:.1f}")
            print(f"  平均速度: {avg_spd:.2f}")
        
        if gen < max_gen - 1:
            pop.reproduce(verbose=False)
            pop.generation += 1
    
    # 最终结果
    print("\n" + "=" * 60)
    print("阶段三考核结果")
    print("=" * 60)
    
    total_stored = sum(history['stored'])
    total_eaten = sum(history['eaten'])
    avg_winter_sanct = np.mean(history['winter_in_sanctuary'])
    avg_fatigue = np.mean(history['avg_fatigue'])
    avg_speed = np.mean(history['avg_speed'])
    
    print(f"总贮粮: {total_stored}")
    print(f"总进食: {total_eaten}")
    print(f"冬天平均在庇护所: {avg_winter_sanct:.1f} agents")
    print(f"平均疲劳: {avg_fatigue:.1f}")
    print(f"平均速度: {avg_speed:.2f}")
    
    # 考核
    p1 = avg_winter_sanct > 1.0  # 趋温性
    p2 = avg_fatigue > 10.0  # 有疲劳感知
    p3 = total_stored >= 50  # 贮粮能力保持
    
    print(f"\n考核项:")
    print(f"1. 趋温性 (冬天在庇护所): {avg_winter_sanct:.1f} - {'✅' if p1 else '❌'}")
    print(f"2. 疲劳感知: {avg_fatigue:.1f} - {'✅' if p2 else '❌'}")
    print(f"3. 贮粮保持: {total_stored} - {'✅' if p3 else '❌'}")
    print(f"\n综合: {p1+p2+p3}/3")
    
    if p1 and p2 and p3:
        print("\n🎉 阶段三毕业！")


if __name__ == '__main__':
    main()