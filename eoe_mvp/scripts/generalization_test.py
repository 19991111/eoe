#!/usr/bin/env python3
"""
EOE v0.26 泛化能力测试
测试训练后的代理在不同环境下的表现
"""

import sys
import os
import time
import pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import core


def run_generalization_test():
    """测试网络的泛化能力"""
    
    print("\n" + "="*55)
    print("🧪 泛化能力测试 - 在基准训练，测试多环境")
    print("="*55)
    
    # 1. 在基准环境训练
    print("\n[1/3] 训练中 (基准环境, 60代)...")
    
    pop = core.Population(population_size=15, lifespan=80, use_champion=True)
    pop.environment.synaptic_pruning_enabled = False
    pop.environment.metabolic_beta = 0.0
    
    for agent in pop.agents:
        for edge in agent.genome.edges:
            edge['enabled'] = True
    
    best_agent = None
    for gen in range(60):
        pop.epoch(verbose=False)
        best = max(pop.agents, key=lambda a: a.fitness)
        if best.fitness > (best_agent.fitness if best_agent else 0):
            best_agent = best
    
    print(f"  最佳适应度: {best_agent.fitness:.1f}")
    
    # 保存最佳代理
    with open('/tmp/best_agent.pkl', 'wb') as f:
        pickle.dump(best_agent, f)
    
    # 2. 在各种环境中测试
    test_configs = [
        ("基准环境", {"sensor_noise": 0, "n_boss_beasts": 0, "toxicity": 0}),
        ("轻度噪声", {"sensor_noise": 0.15, "n_boss_beasts": 0, "toxicity": 0}),
        ("中度噪声", {"sensor_noise": 0.25, "n_boss_beasts": 0, "toxicity": 0}),
        ("重度噪声", {"sensor_noise": 0.40, "n_boss_beasts": 0, "toxicity": 0}),
        ("1捕食者", {"sensor_noise": 0, "n_boss_beasts": 1, "toxicity": 0}),
        ("2捕食者", {"sensor_noise": 0, "n_boss_beasts": 2, "toxicity": 0}),
        ("部分有毒", {"sensor_noise": 0, "n_boss_beasts": 0, "toxicity": 0.3}),
        ("全部有毒", {"sensor_noise": 0, "n_boss_beasts": 0, "toxicity": 1.0}),
    ]
    
    print("\n[2/3] 测试不同环境...")
    
    results = []
    
    for env_name, config in test_configs:
        # 加载最佳代理
        with open('/tmp/best_agent.pkl', 'rb') as f:
            test_agent = pickle.load(f)
        
        # 创建测试环境
        test_pop = core.Population(population_size=1, lifespan=80, use_champion=False)
        test_pop.environment.synaptic_pruning_enabled = False
        test_pop.environment.metabolic_beta = 0.0
        
        # 应用配置
        test_pop.environment.sensor_noise = config["sensor_noise"]
        test_pop.environment.n_boss_beasts = config["n_boss_beasts"]
        
        # 设置毒性
        if config["toxicity"] > 0:
            n_food = test_pop.environment.n_food
            test_pop.environment.food_toxicity = {
                i: (i < n_food * config["toxicity"]) 
                for i in range(n_food)
            }
        
        # 替换代理
        test_pop.agents = [test_agent]
        test_agent.x = test_pop.environment.width / 2
        test_agent.y = test_pop.environment.height / 2
        test_agent.fitness = 0
        test_agent.food_eaten = 0
        test_agent.steps_alive = 0
        test_agent.is_alive = True
        
        # 运行
        for _ in range(80):
            if not test_agent.is_alive:
                break
            test_pop.environment.step()
            test_pop.environment.agents = [test_agent]
        
        results.append({
            'name': env_name,
            'fitness': test_agent.fitness,
            'food': test_agent.food_eaten,
            'steps': test_agent.steps_alive
        })
        
        print(f"  {env_name:<12} Fit={test_agent.fitness:>7.1f} Food={test_agent.food_eaten:>2} Steps={test_agent.steps_alive}")
    
    # 3. 分析
    print("\n[3/3] 分析结果")
    print("="*55)
    
    base_fitness = results[0]['fitness']
    
    print(f"\n{'环境':<12} {'Fitness':>10} {'变化率':>12}")
    print("-"*40)
    
    for r in results:
        if base_fitness > 0:
            pct = (r['fitness'] - base_fitness) / base_fitness * 100
        else:
            pct = 0
        print(f"{r['name']:<12} {r['fitness']:>10.1f} {pct:>+11.1f}%")
    
    # 找出最鲁棒和最脆弱的环境
    robust = min(results, key=lambda x: abs(x['fitness'] - base_fitness) / base_fitness if base_fitness > 0 else 999)
    fragile = max(results, key=lambda x: abs(x['fitness'] - base_fitness) / base_fitness if base_fitness > 0 else 0)
    
    print(f"\n🏅 最鲁棒: {robust['name']}")
    print(f"💀 最脆弱: {fragile['name']}")


if __name__ == "__main__":
    run_generalization_test()