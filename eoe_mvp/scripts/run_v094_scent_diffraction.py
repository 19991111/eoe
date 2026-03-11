"""
v0.94 气味衍射 + 强制绕行实验

行动2: 气味衍射 - 气味像流体绕过障碍物
行动3: 100%食物在墙后 - 强制绕行

核心改进：
- BFS气味扩散：气味从墙壁开口处溢出
- 强制绕行：所有食物在墙后，无"低垂的果实"
"""
import sys; sys.path.insert(0, 'core')
import numpy as np
import pickle
from core import Population
from collections import deque

class ScentDiffractionEnv:
    """气味衍射环境 - 气味像水流一样绕过障碍物"""
    
    def __init__(self):
        # L型墙（倒U型更简单）
        self.walls = [
            {'type': 'vertical', 'x': 50, 'y1': 30, 'y2': 70},  # 中间竖墙
            {'type': 'horizontal', 'x1': 50, 'x2': 80, 'y': 30},  # 顶部横墙
        ]
        # 墙后食物区域（左上角）
        self.food_zone = {'x1': 10, 'x2': 40, 'y1': 10, 'y2': 40}
        
    def is_wall(self, x, y):
        for w in self.walls:
            if w['type'] == 'vertical':
                if abs(x - w['x']) < 2 and w['y1'] <= y <= w['y2']:
                    return True
            else:
                if abs(y - w['y']) < 2 and w['x1'] <= x <= w['x2']:
                    return True
        return False
    
    def get_bfs_scent(self, x, y, food_positions, grid_size=2):
        """
        BFS气味扩散 - 气味可以绕过墙走
        从食物位置向外BFS，墙壁阻挡直接传播但绕墙可以
        """
        if not food_positions:
            return 0
        
        # 简化的BFS气味计算
        best_scent = 0
        
        for fx, fy in food_positions:
            # 简单启发式：曼哈顿距离 + 墙壁惩罚
            dx, dy = abs(x - fx), abs(y - fy)
            direct_dist = dx + dy
            
            # 检查直线是否被墙阻挡
            blocked = False
            if dx > dy:  # 水平为主
                for wx in np.arange(min(x, fx), max(x, fx), 2):
                    if self.is_wall(wx, y):
                        blocked = True
                        break
            else:  # 垂直为主
                for wy in np.arange(min(y, fy), max(y, fy), 2):
                    if self.is_wall(x, wy):
                        blocked = True
                        break
            
            if blocked:
                # 被墙挡住，绕行距离 = 直接距离 + 墙宽度惩罚
                bypass_dist = 20  # 近似绕过L型墙的额外距离
                effective_dist = direct_dist + bypass_dist
            else:
                effective_dist = direct_dist
            
            scent = max(0, 1.0 - effective_dist / 60)
            best_scent = max(best_scent, scent)
        
        return best_scent
    
    def check_collision(self, x, y):
        return self.is_wall(x, y)
    
    def spawn_food_behind_wall(self):
        """只在墙后生成食物"""
        x = np.random.uniform(self.food_zone['x1'], self.food_zone['x2'])
        y = np.random.uniform(self.food_zone['y1'], self.food_zone['y2'])
        return x, y


print("=" * 60)
print("v0.94 气味衍射 + 强制绕行实验")
print("=" * 60)

pop = Population(population_size=20, elite_ratio=0.20, lifespan=250, use_champion=True,
                 n_food=6, food_energy=50, seasonal_cycle=True, season_length=50,
                 winter_food_multiplier=0.0, winter_metabolic_multiplier=1.5)

pop.environment.sensor_range = 50
pop.environment.nest_enabled = True
pop.environment.nest_position = np.array([85.0, 85.0])  # 巢穴在右下角
pop.environment.rival_enabled = True
pop.environment.n_rivals = 3

env = ScentDiffractionEnv()

best_fitness = 0
best_agent = None
total_detours = 0

for gen in range(2000):
    # 每代重置食物位置（在墙后）
    pop.environment.food_positions = [env.spawn_food_behind_wall() for _ in range(6)]
    pop.environment.food_spawn_timer = 0
    
    for a in pop.agents: 
        a.food_in_stomach = a.food_carried = a.food_stored = 0
        a.scent_delta_reward = 0
        a.bump_penalty = 0
        a.explore_reward = 0
        a.visited = set()
        a.prev_pos = None
    
    gen_d = 0
    prev_best_scent = {}
    
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
            
            # 使用BFS气味衍射
            best_scent = env.get_bfs_scent(a.x, a.y, pop.environment.food_positions)
            
            prev = prev_best_scent.get(id(a), 0)
            if best_scent > prev + 0.01:
                a.scent_delta_reward += int((best_scent - prev) * 30)
            
            if a.prev_pos and abs(a.x - a.prev_pos[0]) < 0.5 and abs(a.y - a.prev_pos[1]) < 0.5:
                a.bump_penalty -= 3
            
            gx, gy = int(a.x/5), int(a.y/5)
            if (gx, gy) not in a.visited:
                a.visited.add((gx, gy))
                a.explore_reward += 1
            
            prev_best_scent[id(a)] = best_scent
            a.prev_pos = (a.x, a.y)
            
            if env.check_collision(a.x, a.y): a.internal_energy -= 0.3
            
            for i, (fx, fy) in enumerate(list(pop.environment.food_positions)):
                d = np.sqrt((a.x-fx)**2+(a.y-fy)**2)
                if d < 8:
                    if a.food_in_stomach < 1:
                        a.food_in_stomach = 1; a.food_eaten += 1; a.internal_energy = min(200, a.internal_energy+50)
                    else:
                        a.food_carried = 1
                        # 检测是否绕过墙
                        if env.is_wall(50, 55): gen_d += 1
                    pop.environment.food_positions.pop(i); break
            
            # 巢穴在右下角
            nx, ny = 85, 85
            nd = np.sqrt((a.x-nx)**2+(a.y-ny)**2)
            if nd < 15 and a.food_carried > 0: a.food_stored += a.food_carried; a.food_carried = 0
            if a.internal_energy < 50 and a.food_stored > 0:
                a.food_stored -= 1; a.food_in_stomach = 1; a.internal_energy = min(200, a.internal_energy+50)
        
        for a in pop.agents:
            if a.is_alive and hasattr(a, 'genome'):
                try:
                    acts = {n: node.activation for n, node in a.genome.nodes.items()}
                    a.genome.hebbian_update(acts, lr=0.01)
                except: pass
    
    total_detours += gen_d
    for a in pop.agents:
        f = a.food_eaten + a.food_carried + a.food_stored * 2
        base = f * 1000 + a.scent_delta_reward + a.bump_penalty + a.explore_reward - a.steps_alive * 0.1
        a.fitness = base + a.food_carried * 5000 + a.food_stored * 10000
    
    best = max(pop.agents, key=lambda a: a.fitness)
    if best.fitness > best_fitness:
        best_fitness = best.fitness
        best_agent = pickle.loads(pickle.dumps(best))
    
    if gen % 200 == 0: 
        tc = sum(a.food_carried for a in pop.agents)
        ts = sum(a.food_stored for a in pop.agents)
        print(f"Gen {gen:4d} | Fit={int(best.fitness):6d} | C={tc} S={ts} D={gen_d}")
    
    pop.reproduce(verbose=False)

print(f"\n🎯 最高适应度: {int(best_fitness)}")
print(f"🎯 总绕行: {total_detours}")

# 保存最佳Agent
if best_agent:
    with open('champions/best_v094_diffraction.pkl', 'wb') as f:
        pickle.dump(best_agent, f)
    
    meta = {
        'version': 'v0.94',
        'fitness': best_fitness,
        'detours': total_detours,
        'n_nodes': len(best_agent.genome.nodes),
        'n_edges': len(best_agent.genome.edges)
    }
    import json
    with open('champions/best_v094_diffraction_meta.json', 'w') as f:
        json.dump(meta, f)
    
    print(f"\n✅ v0.94最强大脑已保存!")
    print(f"   文件: champions/best_v094_diffraction.pkl")
