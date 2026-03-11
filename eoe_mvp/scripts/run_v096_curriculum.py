"""
v0.96 课程学习版 - 循序渐进四大机制

Phase 1 (Gen 0-99):   基础觅食
Phase 2 (Gen 100-199): 加入L型墙
Phase 3 (Gen 200-299): 加入气味盲区
Phase 4 (Gen 300-399): 混合地形
Phase 5 (Gen 400-499): 加入敌对Agent
Phase 6 (Gen 500-699): 物流大考(巢穴墙外)
Phase 7 (Gen 700+):    全部混合
"""
import sys; sys.path.insert(0, 'core')
import numpy as np
import pickle
from core import Population

class CurriculumEnv:
    """课程学习环境"""
    
    def __init__(self, phase=0):
        self.phase = phase
        
        # L型墙
        self.walls = [(50, 30, 50, 70), (50, 30, 80, 30)]
        
        # 墙后区域
        self.wall_zone = {'x1': 10, 'x2': 45, 'y1': 10, 'y2': 45}
        
        # 开阔地区域
        self.open_zone = {'x1': 60, 'x2': 95, 'y1': 60, 'y2': 95}
        
        # 盲区
        self.blind_spot = {'x1': 45, 'x2': 55, 'y1': 20, 'y2': 30}
        
        # 巢穴位置 (根据phase变化)
        self.nest_pos = np.array([80.0, 80.0])  # 墙外
        
        # 敌对Agent
        self.rivals = []
    
    def is_wall(self, x, y):
        for wx1, wy1, wx2, wy2 in self.walls:
            if wx1 == wx2:
                if abs(x - wx1) < 2 and min(wy1, wy2) <= y <= max(wy1, wy2):
                    return True
            else:
                if abs(y - wy1) < 2 and min(wx1, wx2) <= x <= max(wx1, wx2):
                    return True
        return False
    
    def is_blind_spot(self, x, y):
        if self.phase < 2: return False  # Phase 1-2: 无盲区
        return (self.blind_spot['x1'] <= x <= self.blind_spot['x2'] and
                self.blind_spot['y1'] <= y <= self.blind_spot['y2'])
    
    def spawn_food(self):
        """根据phase决定食物位置"""
        if self.phase < 3:
            # Phase 1-2: 开阔地
            x = np.random.uniform(60, 90)
            y = np.random.uniform(60, 90)
        elif self.phase < 4:
            # Phase 3: 30%墙后
            if np.random.random() < 0.3:
                x = np.random.uniform(10, 45)
                y = np.random.uniform(10, 45)
            else:
                x = np.random.uniform(60, 90)
                y = np.random.uniform(60, 90)
        else:
            # Phase 4+: 60%墙后
            if np.random.random() < 0.6:
                x = np.random.uniform(10, 45)
                y = np.random.uniform(10, 45)
            else:
                x = np.random.uniform(60, 90)
                y = np.random.uniform(60, 90)
        return x, y
    
    def get_scent(self, x, y, food_positions):
        """带盲区检测的气味"""
        if self.is_blind_spot(x, y):
            return 0.0
        
        if not food_positions:
            return 0.0
        
        best_scent = 0.0
        for fx, fy in food_positions:
            dx, dy = abs(x - fx), abs(y - fy)
            dist = dx + dy
            
            blocked = False
            if self.phase >= 1:  # Phase 2+: 有墙
                if dx > dy:
                    for wx in np.arange(min(x, fx), max(x, fx), 2):
                        if self.is_wall(wx, y): blocked = True; break
                else:
                    for wy in np.arange(min(y, fy), max(y, fy), 2):
                        if self.is_wall(x, wy): blocked = True; break
            
            effective_dist = dist + 25 if blocked else dist
            scent = max(0, 1.0 - effective_dist / 60)
            best_scent = max(best_scent, scent)
        
        return best_scent


print("=" * 60)
print("v0.96 课程学习版")
print("  Phase 1: 基础  |  Phase 2: +墙  |  Phase 3: +盲区")
print("  Phase 4: +混合  |  Phase 5: +敌对  |  Phase 6: +物流")
print("=" * 60)

# 巢穴在墙外 (Phase 6+)
nest_pos = np.array([80.0, 80.0])

pop = Population(population_size=20, elite_ratio=0.20, lifespan=200, use_champion=True,
                 n_food=6, food_energy=50, seasonal_cycle=True, season_length=40,
                 winter_food_multiplier=0.0, winter_metabolic_multiplier=1.5)

pop.environment.sensor_range = 50
pop.environment.nest_enabled = True
pop.environment.nest_position = nest_pos
pop.environment.rival_enabled = True
pop.environment.n_rivals = 3

best_fitness = 0
best_agent = None
stats_history = []

for gen in range(700):
    # 课程学习阶段
    if gen < 100: phase = 0
    elif gen < 200: phase = 1
    elif gen < 300: phase = 2
    elif gen < 400: phase = 3
    elif gen < 500: phase = 4
    elif gen < 700: phase = 5
    else: phase = 6
    
    if gen % 100 == 0:
        print(f"\n>>> Phase {phase} 开始 (Gen {gen})")
    
    env = CurriculumEnv(phase=phase)
    
    # 根据phase设置环境
    use_walls = phase >= 1
    use_blind = phase >= 2
    use_nest_outside = phase >= 5
    
    # 食物生成
    pop.environment.food_positions = [env.spawn_food() for _ in range(6)]
    pop.environment.food_spawn_timer = 0
    
    for a in pop.agents:
        a.food_in_stomach = a.food_carried = a.food_stored = 0
        a.scent_delta_reward = 0
        a.explore_reward = 0
        a.visited = set()
        a.in_blind_spot = 0
    
    current_nest = nest_pos if use_nest_outside else np.array([20.0, 80.0])
    
    for step in range(200):
        # 季节
        is_summer = (step % 80) < 40
        pop.environment.winter_metabolic_multiplier = 1.5 if not is_summer else 1.0
        pop.environment.current_season = 'winter' if not is_summer else 'summer'
        
        for a in pop.agents:
            if not a.is_alive: continue
            if a.food_in_stomach >= 1: a.internal_energy -= 0.3
            if a.food_carried > 0: a.internal_energy -= 0.1
        
        pop.environment.agents = pop.agents
        pop.environment.step()
        
        for a in pop.agents:
            if not a.is_alive: continue
            
            # 气味 (带墙和盲区)
            if use_walls:
                best_scent = env.get_scent(a.x, a.y, pop.environment.food_positions)
            else:
                if pop.environment.food_positions:
                    fx, fy = pop.environment.food_positions[0]
                    d = abs(a.x - fx) + abs(a.y - fy)
                    best_scent = max(0, 1.0 - d / 60)
                else:
                    best_scent = 0
            
            prev = getattr(a, 'prev_scent', 0)
            if best_scent > prev + 0.01:
                a.scent_delta_reward += int((best_scent - prev) * 30)
            a.prev_scent = best_scent
            
            # 盲区
            if use_blind and env.is_blind_spot(a.x, a.y):
                a.in_blind_spot += 1
            
            # 撞墙
            if use_walls and env.is_wall(a.x, a.y):
                a.internal_energy -= 3
            
            # 探索
            gx, gy = int(a.x/5), int(a.y/5)
            if (gx, gy) not in a.visited:
                a.visited.add((gx, gy))
                a.explore_reward += 1
            
            # 吃食物
            for i, (fx, fy) in enumerate(list(pop.environment.food_positions)):
                d = np.sqrt((a.x-fx)**2+(a.y-fy)**2)
                if d < 8:
                    if a.food_in_stomach < 1:
                        a.food_in_stomach = 1
                        a.food_eaten += 1
                        a.internal_energy = min(200, a.internal_energy+50)
                    else:
                        a.food_carried = 1
                    pop.environment.food_positions.pop(i); break
            
            # 贮粮
            nx, ny = current_nest
            if np.sqrt((a.x-nx)**2+(a.y-ny)**2) < 15 and a.food_carried > 0:
                a.food_stored += a.food_carried
                a.food_carried = 0
            
            # 冬天用贮粮
            if a.internal_energy < 50 and a.food_stored > 0:
                a.food_stored -= 1
                a.food_in_stomach = 1
                a.internal_energy = min(200, a.internal_energy + 50)
        
        # Hebbian学习
        for a in pop.agents:
            if a.is_alive and hasattr(a, 'genome'):
                try:
                    acts = {n: node.activation for n, node in a.genome.nodes.items()}
                    a.genome.hebbian_update(acts, lr=0.01)
                except: pass
    
    # 适应度
    for a in pop.agents:
        f = a.food_eaten + a.food_carried + a.food_stored * 2
        base = f * 1000 + a.scent_delta_reward + a.explore_reward - a.steps_alive * 0.1
        bonus = a.food_stored * 5000 + a.food_carried * 3000
        if a.in_blind_spot > 5:
            bonus += a.in_blind_spot * 10
        a.fitness = base + bonus
    
    best = max(pop.agents, key=lambda a: a.fitness)
    if best.fitness > best_fitness:
        best_fitness = best.fitness
        best_agent = pickle.loads(pickle.dumps(best))
    
    if gen % 50 == 0:
        print(f"Gen {gen:3d} P{phase} | Fit={int(best.fitness):6d} | C={best.food_carried} S={best.food_stored} B={a.in_blind_spot}")
    
    stats_history.append({'gen': gen, 'phase': phase, 'fitness': best.fitness, 
                         'carry': best.food_carried, 'store': best.food_stored})
    
    pop.reproduce(verbose=False)

print(f"\n🎯 最高适应度: {int(best_fitness)}")

if best_agent:
    with open('champions/best_v096_curriculum.pkl', 'wb') as f:
        pickle.dump(best_agent, f)
    
    import json
    meta = {
        'version': 'v0.96',
        'fitness': best_fitness,
        'total_gen': 700,
        'n_nodes': len(best_agent.genome.nodes),
        'n_edges': len(best_agent.genome.edges),
        'stats': stats_history
    }
    with open('champions/best_v096_curriculum_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    print("✅ v0.96课程学习版大脑已保存!")
