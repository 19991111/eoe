"""
v0.97 三大突破机制 - 叠加使用

方案1: 代谢疲劳 + 安全掩体
  - Agent活动时积累疲劳，疲劳满必须睡眠
  - 开阔地睡眠有致命危险(99%死亡率)
  - 墙拐角是安全区
  → 将"贮粮过冬"降维成"带零食回安全屋睡觉"

方案2: 无聊信息素
  - Agent原地不动时分泌"废气"
  - 气味会随时间累积和消散
  - 演化出厌恶自身气味 → 被动后退
  → 用环境物理学强制打破死循环

方案3: 夏日食物热力学
  - 平原食物暴露在阳光下，保质期短
  - 墙后/巢穴阴影区食物保鲜
  - 腐败食物有毒
  → "贮存"在当下就有收益

Flag控制:
  fatigue_system=True/False      # 方案1
  pheromone_system=True/False    # 方案2
  food_thermodynamics=True/False # 方案3
"""
import sys; sys.path.insert(0, 'core')
import numpy as np
import pickle
from core import Population

# L型墙 (与v0.95相同)
WALLS = [(50, 30, 50, 70), (50, 30, 80, 30)]

class BreakingEnv:
    """三大突破机制环境"""
    
    def __init__(self, 
                 use_fatigue=True, 
                 use_pheromone=True, 
                 use_thermodynamics=True):
        self.use_fatigue = use_fatigue
        self.use_pheromone = use_pheromone
        self.use_thermodynamics = use_thermodynamics
        
        # 墙后区域 (危险区 = 平原)
        self.wall_zone = {'x1': 10, 'x2': 45, 'y1': 10, 'y2': 45}
        self.open_zone = {'x1': 55, 'x2': 95, 'y1': 55, 'y2': 95}
        
        # 巢穴位置 (墙内 = 安全)
        self.nest_pos = np.array([25.0, 25.0])
        
    def spawn_food(self):
        """60%墙后(阴影区), 40%开阔地(暴晒区)"""
        if np.random.random() < 0.6:
            x = np.random.uniform(10, 45)
            y = np.random.uniform(10, 45)
        else:
            x = np.random.uniform(60, 90)
            y = np.random.uniform(60, 90)
        return x, y
    
    def is_shadow_zone(self, x, y):
        """墙后或巢穴内 = 阴影区"""
        # 巢穴内
        nx, ny = self.nest_pos
        if np.sqrt((x - nx)**2 + (y - ny)**2) < 10:
            return True
        # 墙后
        if 10 <= x <= 45 and 10 <= y <= 45:
            return True
        return False
    
    def is_safe_zone(self, x, y):
        """安全区: 墙拐角或巢穴"""
        # 巢穴
        nx, ny = self.nest_pos
        if np.sqrt((x - nx)**2 + (y - ny)**2) < 10:
            return True
        # L型墙拐角 (50, 30)
        if abs(x - 50) < 8 and abs(y - 30) < 8:
            return True
        return False


print("=" * 60)
print("v0.97 Stage 2: 起床饥饿 - 强制行为耦合")
print("  目标: 逼迫Agent'带着食物去睡觉'")
print("  - 开启: 起床饥饿 (睡觉消耗大量能量)")
print("  - 关闭: 物理掉落 (让Agent主动放下食物)")
print("  - 期望: 睡眠基因+携带基因强制融合")
print("=" * 60)

# 创建环境 (只开疲劳)
env = BreakingEnv(use_fatigue=True, use_pheromone=False, use_thermodynamics=False)

# 创建种群
pop = Population(
    population_size=20, 
    elite_ratio=0.20, 
    lifespan=200, 
    use_champion=True,
    n_food=6, 
    food_energy=50, 
    seasonal_cycle=True, 
    season_length=40,
    winter_food_multiplier=0.0, 
    winter_metabolic_multiplier=1.5
)

# 设置环境参数
pop.environment.sensor_range = 50
pop.environment.nest_enabled = True
pop.environment.nest_position = env.nest_pos
pop.environment.nest_radius = 10.0

# 添加L型墙
pop.environment.n_walls = 2
pop.environment.walls = WALLS

# ====================== 启用三大机制 ======================

# 方案1: 代谢疲劳 + 安全掩体 (Stage 2: 一次性起床惩罚+物理掉落)
# 关键: 睡觉安全，醒来扣血，食物掉脚下可立刻吃
pop.environment.enable_fatigue_system(
    enabled=True,
    max_fatigue=50.0,           # 最大疲劳值
    fatigue_build_rate=0.5,     # 每步积累0.5 (100步满)
    sleep_danger_prob=0.0,      # 不强制死亡
    enable_wakeup_hunger=True,  # Stage 2: 一次性起床惩罚
    enable_sleep_drop=True      # Stage 2: 启用物理掉落，食物掉脚下
)

# 为所有Agent设置max_fatigue
for a in pop.agents:
    a.max_fatigue = 50.0
print("  ✓ 代谢疲劳系统: enabled (纯净版)")

# Stage 1: 关闭其他两个机制
pop.environment.enable_pheromone_system(enabled=False)
print("  ○ 无聊信息素系统: disabled")

pop.environment.enable_food_thermodynamics(enabled=False)
print("  ○ 夏日食物热力学: disabled")

# 运行演化
best_fitness = 0
best_agent = None
stats_history = []

for gen in range(300):  # Stage 2: 300代
    # 重置食物
    pop.environment.food_positions = [env.spawn_food() for _ in range(6)]
    pop.environment.food_spawn_timer = 0
    
    # 重置Agent状态 (保留sleep_cycles用于统计)
    for a in pop.agents:
        a.food_in_stomach = a.food_carried = a.food_stored = 0
        a.fatigue = 0
        a.is_sleeping = False
        a.pheromone_level = 0
        a.stationary_frames = 0
        a.died_in_sleep = False
        # 注意: 不重置sleep_cycles，让它累积
    
    for step in range(200):
        # 季节
        is_summer = (step % 80) < 40
        pop.environment.winter_metabolic_multiplier = 1.5 if not is_summer else 1.0
        pop.environment.current_season = 'winter' if not is_summer else 'summer'
        
        # 代谢消耗
        for a in pop.agents:
            if not a.is_alive: continue
            if a.food_in_stomach >= 1: a.internal_energy -= 0.3
            if a.food_carried > 0: a.internal_energy -= 0.1
        
        pop.environment.agents = pop.agents
        pop.environment.step()
        
        # 检查食物和巢穴
        for a in pop.agents:
            if not a.is_alive: continue
            
            # 巢穴存储
            if pop.environment.nest_enabled:
                nx, ny = pop.environment.nest_position
                dist_to_nest = np.sqrt((a.x - nx)**2 + (a.y - ny)**2)
                if dist_to_nest < pop.environment.nest_radius and a.food_carried > 0:
                    # 贮粮成功
                    a.food_stored += a.food_carried
                    a.food_carried = 0
            
            # 死亡检查
            if a.internal_energy <= 0:
                a.is_alive = False
    
    # 统计
    fitnesses = []
    total_stored = 0
    total_carried = 0
    total_sleep = 0
    died_in_sleep = 0
    agents_with_both = 0  # 同时有Carry和Sleep的个体
    
    for a in pop.agents:
        # 适应度计算
        food_bonus = a.food_eaten * 100
        stored_bonus = a.food_stored * 200
        carried_bonus = a.food_carried * 150
        energy_bonus = a.internal_energy * 0.5
        survival_bonus = a.steps_alive * 0.5
        
        # Stage 2: 强化睡眠奖励 + 行为耦合奖励
        sleep_bonus = a.sleep_cycles * 20  # 每次睡眠尝试 +20
        
        # 行为耦合奖励: 睡觉时携带食物
        if a.sleep_cycles > 0 and a.food_carried > 0:
            carried_in_sleep = a.food_carried  # 假设睡觉时携带
            coupling_bonus = carried_in_sleep * 100  # 强烈的耦合奖励!
            sleep_bonus += coupling_bonus
        
        # 惩罚
        surprise_penalty = a.surprise_accumulated * 0.1
        
        fitness = food_bonus + stored_bonus + carried_bonus + energy_bonus + survival_bonus + sleep_bonus - surprise_penalty
        
        if a.died_in_sleep:
            fitness -= 50  # 睡眠死亡惩罚
        
        a.fitness = fitness
        fitnesses.append(fitness)
        
        total_stored += a.food_stored
        total_carried += a.food_carried
        total_sleep += a.sleep_cycles
        if a.died_in_sleep:
            died_in_sleep += 1
        # 追踪行为耦合 (使用新增的has_slept_with_food标记)
        if a.sleep_cycles > 0 and (a.food_carried > 0 or getattr(a, 'has_slept_with_food', False)):
            agents_with_both += 1
    
    avg_fit = np.mean(fitnesses) if fitnesses else 0
    max_fit = max(fitnesses) if fitnesses else 0
    
    stats_history.append({
        'gen': gen,
        'avg_fit': avg_fit,
        'max_fit': max_fit,
        'stored': total_stored,
        'carried': total_carried,
        'sleep': total_sleep,
        'died_sleep': died_in_sleep
    })
    
    if gen % 20 == 0:
        print(f"Gen {gen:3d} | Best: {max_fit:7.1f} | Avg: {avg_fit:6.1f} | "
              f"Stored: {total_stored:2d} | Carried: {total_carried:2d} | "
              f"Sleep: {total_sleep:3d} | Both: {agents_with_both:2d}")
    
    # 保存最佳
    if max_fit > best_fitness:
        best_fitness = max_fit
        best_agent = pop.agents[np.argmax(fitnesses)]
    
    # 阶段性保存关键代数的冠军 (Gen 120, 260, 280, 420, 480)
    if gen in [120, 260, 280, 420, 480] and max_fit > 100:
        best_at_gen = pop.agents[np.argmax(fitnesses)]
        best_at_gen.genome.save_json(f'../champions/best_v097_gen{gen}_fit{max_fit:.0f}.json')
        print(f"  💾 已保存 Gen {gen} 冠军 (fit={max_fit:.0f})")
    
    # 演化下一代
    pop.reproduce(verbose=False)

# 最终结果
print("\n" + "=" * 60)
print("最终结果:")
print(f"  最高适应度: {best_fitness:.1f}")
print(f"  存储食物: {best_agent.food_stored}")
print(f"  携带食物: {best_agent.food_carried}")
print(f"  睡眠次数: {best_agent.sleep_cycles}")
print("=" * 60)

# 保存冠军大脑
import os
os.makedirs('../champions', exist_ok=True)
best_agent.genome.save_json('../champions/best_v097_brain.json')

print("\n冠军大脑已保存到: champions/best_v097_brain.json")