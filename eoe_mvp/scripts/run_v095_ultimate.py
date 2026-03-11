"""
v0.95 终极挑战 - 四大机制

行动2: 环境地貌混合 - 开阔地+墙后食物+敌对Agent
行动3: 气味盲区 - 墙拐角处无信号
行动4: 物流大考 - 巢穴在墙外，食物在墙内
"""
import sys; sys.path.insert(0, 'core')
import numpy as np
import pickle
from core import Population

class UltimateEnv:
    """终极环境 - 混合地形 + 气味盲区"""
    
    def __init__(self):
        # L型墙
        self.walls = [
            (50, 30, 50, 70),  # 竖墙
            (50, 30, 80, 30),  # 横墙
        ]
        
        # 墙后食物区 (高价值)
        self.wall_zone = {'x1': 10, 'x2': 45, 'y1': 10, 'y2': 45}
        
        # 开阔地食物区 (基础)
        self.open_zone = {'x1': 60, 'x2': 95, 'y1': 60, 'y2': 95}
        
        # 盲区: 墙拐角外侧 (50, 25) - 几帧内无气味
        self.blind_spot = {'x1': 45, 'x2': 55, 'y1': 20, 'y2': 30}
    
    def is_wall(self, x, y):
        for wx1, wy1, wx2, wy2 in self.walls:
            if wx1 == wx2:  # 竖墙
                if abs(x - wx1) < 2 and min(wy1, wy2) <= y <= max(wy1, wy2):
                    return True
            else:  # 横墙
                if abs(y - wy1) < 2 and min(wx1, wx2) <= x <= max(wx1, wx2):
                    return True
        return False
    
    def is_blind_spot(self, x, y):
        """气味盲区"""
        return (self.blind_spot['x1'] <= x <= self.blind_spot['x2'] and
                self.blind_spot['y1'] <= y <= self.blind_spot['y2'])
    
    def spawn_food_mixed(self):
        """混合食物生成 - 60%墙后, 40%开阔地"""
        if np.random.random() < 0.6:
            # 墙后 (高难度)
            x = np.random.uniform(self.wall_zone['x1'], self.wall_zone['x2'])
            y = np.random.uniform(self.wall_zone['y1'], self.wall_zone['y2'])
        else:
            # 开阔地 (简单)
            x = np.random.uniform(self.open_zone['x1'], self.open_zone['x2'])
            y = np.random.uniform(self.open_zone['y1'], self.open_zone['y2'])
        return x, y
    
    def get_scent(self, x, y, food_positions):
        """带盲区检测的气味"""
        # 盲区内无气味
        if self.is_blind_spot(x, y):
            return 0.0
        
        if not food_positions:
            return 0.0
        
        best_scent = 0.0
        for fx, fy in food_positions:
            dx, dy = abs(x - fx), abs(y - fy)
            dist = dx + dy
            
            # 检查是否被墙阻挡
            blocked = False
            if dx > dy:
                for wx in np.arange(min(x, fx), max(x, fx), 2):
                    if self.is_wall(wx, y): blocked = True; break
            else:
                for wy in np.arange(min(y, fy), max(y, fy), 2):
                    if self.is_wall(x, wy): blocked = True; break
            
            if blocked:
                effective_dist = dist + 25  # 绕行惩罚
            else:
                effective_dist = dist
            
            scent = max(0, 1.0 - effective_dist / 60)
            best_scent = max(best_scent, scent)
        
        return best_scent


print("=" * 60)
print("v0.95 终极挑战")
print("  混合地形 + 气味盲区 + 物流大考")
print("=" * 60)

# 巢穴在墙外! (关键)
nest_pos = np.array([80.0, 80.0])

pop = Population(population_size=20, elite_ratio=0.20, lifespan=250, use_champion=True,
                 n_food=8, food_energy=50, seasonal_cycle=True, season_length=50,
                 winter_food_multiplier=0.0, winter_metabolic_multiplier=1.5)

pop.environment.sensor_range = 50
pop.environment.nest_enabled = True
pop.environment.nest_position = nest_pos  # 墙外!
pop.environment.rival_enabled = True
pop.environment.n_rivals = 3

env = UltimateEnv()

best_fitness = 0
best_agent = None
stats = {'detour': 0, 'store': 0, 'carry': 0}

for gen in range(1000):
    # 混合食物
    pop.environment.food_positions = [env.spawn_food_mixed() for _ in range(8)]
    pop.environment.food_spawn_timer = 0
    
    for a in pop.agents: 
        a.food_in_stomach = a.food_carried = a.food_stored = 0
        a.scent_delta_reward = 0
        a.explore_reward = 0
        a.visited = set()
        a.in_blind_spot = 0
        a.prev_pos = None
    
    gen_detour = 0
    
    for step in range(250):
        slen = pop.environment.season_length * 2
        is_summer = (step % slen) < pop.environment.season_length
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
            
            # 气味奖励 (带盲区)
            best_scent = env.get_scent(a.x, a.y, pop.environment.food_positions)
            prev = getattr(a, 'prev_scent', 0)
            if best_scent > prev + 0.01:
                a.scent_delta_reward += int((best_scent - prev) * 30)
            a.prev_scent = best_scent
            
            # 盲区检测
            if env.is_blind_spot(a.x, a.y):
                a.in_blind_spot += 1
            
            # 撞墙
            if env.is_wall(a.x, a.y):
                a.internal_energy -= 3
            
            # 探索奖励
            gx, gy = int(a.x/5), int(a.y/5)
            if (gx, gy) not in a.visited:
                a.visited.add((gx, gy))
                a.explore_reward += 1
            
            a.prev_pos = (a.x, a.y)
            
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
                        # 检测是否绕墙拿到
                        if env.is_wall(fx, fy):  # 简化判断
                            gen_detour += 1
                    pop.environment.food_positions.pop(i); break
            
            # 巢穴贮粮 (墙外!)
            nx, ny = nest_pos
            if np.sqrt((a.x-nx)**2+(a.y-ny)**2) < 15 and a.food_carried > 0:
                a.food_stored += a.food_carried
                a.food_carried = 0
            
            # 冬天用贮粮
            if a.internal_energy < 50 and a.food_stored > 0:
                a.food_stored -= 1
                a.food_in_stomach = 1
                a.internal_energy = min(200, a.internal_energy + 50)
        
        for a in pop.agents:
            if a.is_alive and hasattr(a, 'genome'):
                try:
                    acts = {n: node.activation for n, node in a.genome.nodes.items()}
                    a.genome.hebbian_update(acts, lr=0.01)
                except: pass
    
    # 适应度计算
    for a in pop.agents:
        f = a.food_eaten + a.food_carried + a.food_stored * 2
        base = f * 1000 + a.scent_delta_reward + a.explore_reward - a.steps_alive * 0.1
        # 物流大考加分: 贮粮+绕行组合
        bonus = a.food_stored * 5000 + a.food_carried * 3000
        # 盲区存活加分
        if a.in_blind_spot > 5:
            bonus += a.in_blind_spot * 10
        a.fitness = base + bonus
    
    stats['detour'] += gen_detour
    best = max(pop.agents, key=lambda a: a.fitness)
    if best.fitness > best_fitness:
        best_fitness = best.fitness
        best_agent = pickle.loads(pickle.dumps(best))
        stats['carry'] = best.food_carried
        stats['store'] = best.food_stored
    
    if gen % 100 == 0: 
        print(f"Gen {gen:4d} | Fit={int(best.fitness):6d} | C={best.food_carried} S={best.food_stored} D={gen_detour}")
    
    pop.reproduce(verbose=False)

print(f"\n🎯 最高适应度: {int(best_fitness)}")
print(f"📊 统计: 绕行={stats['detour']}, 携带={stats['carry']}, 贮粮={stats['store']}")

if best_agent:
    with open('champions/best_v095_ultimate.pkl', 'wb') as f:
        pickle.dump(best_agent, f)
    
    import json
    meta = {
        'version': 'v0.95',
        'fitness': best_fitness,
        'stats': stats,
        'n_nodes': len(best_agent.genome.nodes),
        'n_edges': len(best_agent.genome.edges)
    }
    with open('champions/best_v095_ultimate_meta.json', 'w') as f:
        json.dump(meta, f)
    
    print("✅ v0.95大脑已保存!")
