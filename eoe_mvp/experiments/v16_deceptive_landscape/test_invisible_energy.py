#!/usr/bin/env python3
"""
v16.1 欺骗性景观 - 能量隐身机制基准测试
=========================================
目标：验证环境设计是否能够筛选出"记忆"能力

测试设计：
1. 2节点反射Agent：只能看见时追踪，隐身时停滞
2. 4节点记忆Agent：使用RNN逻辑，隐身时保持运动向量
3. 基准线：记忆Agent必须获得比反射Agent高出300%以上的能量

陷阱封堵：
- 乌龟策略：斑块隐身移动距离 > 感知半径
- 可塑性死锁：使用pre-activation计算Hebbian，或添加自发噪声
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import torch
import numpy as np
import matplotlib.pyplot as plt

# ==================== 环境设置 ====================
from core.eoe.environment_gpu import EnvironmentGPU

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# 欺骗性景观参数（运动学强制隔离 - 调整相对速度）
SENSOR_RANGE = 12.0        # 感知半径
FLICKER_PERIOD = 25        # 可见25步
INVISIBLE_MOVES = 75       # 隐身75步（100步一周期）
ENERGY_SPEED = 0.5         # 斑块速度
AGENT_SPEED = 0.6          # Agent速度（相对速度差0.1）

# 验证：25步可见期内，Agent能拉近 25 × 0.1 = 2.5 距离
# 初始距离6.0，剩余 6.0 - 2.5 = 3.5 > 0
# 隐身75步继续追，距离缩小 75 × 0.1 = 7.5，总共可缩小10
# 理论追上时间：6.0/0.1 = 60步（在第2个周期内）
MAX_PURSUIT_IN_VISIBLE = (AGENT_SPEED - ENERGY_SPEED) * FLICKER_PERIOD
INITIAL_DISTANCE = 6.0  # 斑块初始位置
print(f"  相对速度差: {AGENT_SPEED - ENERGY_SPEED:.2f}")
print(f"  可见期最大追击距离: {MAX_PURSUIT_IN_VISIBLE:.2f}")
print(f"  初始距离: {INITIAL_DISTANCE}")
print(f"  剩余距离: {INITIAL_DISTANCE - MAX_PURSUIT_IN_VISIBLE:.2f}")
print(f"  理论追上时间: {INITIAL_DISTANCE / (AGENT_SPEED - ENERGY_SPEED):.0f}步")

# 关键验证：可见期内的最大追击距离
# 相对速度 = 0.05，追击20步只能走1.0距离 << 感知范围10
# 这意味着2节点Agent在可见期内绝对碰不到斑块！
max_pursuit_in_visible = (AGENT_SPEED - ENERGY_SPEED) * FLICKER_PERIOD
print(f"  可见期最大追击距离: {max_pursuit_in_visible:.2f} << 感知范围 {SENSOR_RANGE}")
assert max_pursuit_in_visible < SENSOR_RANGE * 0.5, \
    "陷阱：可见期内能追上，测试失效！"

# 创建环境
env = EnvironmentGPU(
    width=100.0,
    height=100.0,
    resolution=1.0,
    device=device,
    energy_field_enabled=True,
    impedance_field_enabled=False,
    stigmergy_field_enabled=False,
    danger_field_enabled=False,
    matter_grid_enabled=False,
    wind_field_enabled=False,
)

# 自定义能量场行为：添加闪烁机制
class FlickeringEnergyField:
    """欺骗性能量场：周期性隐身 + 惯性运动"""
    
    def __init__(self, base_env, flicker_period=50, invisible_moves=150, speed=0.8):
        self.base_env = base_env
        self.flicker_period = flicker_period
        self.invisible_moves = invisible_moves
        self.speed = speed
        self.step_count = 0
        
        # 斑块状态
        self.patches = []
        self.visibility = {}  # patch_id -> bool
        
        # 初始化斑块
        self._init_patches(n_patches=5)
    
    def _init_patches(self, n_patches, agent_positions=None):
        """初始化能量斑块 - 确保初始距离大于感知范围"""
        for i in range(n_patches):
            if agent_positions and len(agent_positions) > 0:
                # 找到离所有Agent最远的位置
                best_x, best_y = 50, 50
                best_dist = 0
                for _ in range(100):  # 尝试100次
                    x = np.random.uniform(10, 90)
                    y = np.random.uniform(10, 90)
                    min_dist = min(np.sqrt((x-ax)**2 + (y-ay)**2) for ax, ay in agent_positions)
                    if min_dist > best_dist:
                        best_dist = min_dist
                        best_x, best_y = x, y
                x, y = best_x, best_y
            else:
                x = np.random.uniform(10, 90)
                y = np.random.uniform(10, 90)
            
            # 速度方向指向地图中心（增加追逐难度）
            vx = (50 - x) * 0.01 + np.random.uniform(-0.1, 0.1)
            vy = (50 - y) * 0.01 + np.random.uniform(-0.1, 0.1)
            # 归一化到固定速度
            v_norm = np.sqrt(vx**2 + vy**2)
            if v_norm > 0:
                vx = vx / v_norm * self.speed
                vy = vy / v_norm * self.speed
            
            self.patches.append({
                'x': x, 'y': y, 
                'vx': vx, 'vy': vy,
                'energy': 50.0,
                'id': i
            })
            self.visibility[i] = True
    
    def step(self):
        """每步更新"""
        self.step_count += 1
        cycle_pos = self.step_count % (self.flicker_period + self.invisible_moves)
        
        for patch in self.patches:
            pid = patch['id']
            
            # 可见性切换
            if cycle_pos < self.flicker_period:
                self.visibility[pid] = True
            else:
                self.visibility[pid] = False
            
            # 斑块移动（惯性运动，非布朗）
            patch['x'] += patch['vx']
            patch['y'] += patch['vy']
            
            # 边界反弹
            if patch['x'] < 0 or patch['x'] > 100:
                patch['vx'] *= -1
                patch['x'] = np.clip(patch['x'], 0, 100)
            if patch['y'] < 0 or patch['y'] > 100:
                patch['vy'] *= -1
                patch['y'] = np.clip(patch['y'], 0, 100)
    
    def get_stats(self):
        """获取统计信息"""
        visible_count = sum(1 for p in self.patches if self.visibility[p['id']])
        return {
            'visible_patches': visible_count,
            'total_patches': len(self.patches),
            'invisible_moves': self.invisible_moves,
            'cycle_pos': self.step_count % (self.flicker_period + self.invisible_moves)
        }
    
    def get_energy_at(self, x, y, sensor_range):
        """获取某位置的能量（只有可见斑块才被感知）"""
        total_energy = 0.0
        
        for patch in self.patches:
            # 只有可见斑块才能被感知！
            if not self.visibility[patch['id']]:
                continue
                
            dx = patch['x'] - x
            dy = patch['y'] - y
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < sensor_range:
                # 高斯衰减
                intensity = np.exp(-dist**2 / (sensor_range**2 / 2))
                total_energy += patch['energy'] * intensity
        
        return total_energy
    
    def get_actual_energy_at(self, x, y):
        """获取实际物理碰撞能量（无论可见与否）"""
        total_energy = 0.0
        
        for patch in self.patches:
            # 只处理有能量的斑块
            if patch['energy'] <= 0:
                continue
                
            dx = patch['x'] - x
            dy = patch['y'] - y
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < 2.0:  # 物理碰撞半径
                total_energy += patch['energy']
                patch['energy'] = 0  # 斑块消失
        
        return total_energy
    
    def respawn_if_needed(self):
        """重生能量为0的斑块"""
        for patch in self.patches:
            if patch['energy'] <= 0:
                # 在地图另一侧重生（在Agent感知范围外）
                patch['x'] = 95.0  # 远处
                patch['y'] = 0.0
                patch['energy'] = 100.0
                # 重置可见性
                self.visibility[patch['id']] = True
                self.step_count = 0  # 重置周期

# 创建闪烁能量场
flicker_env = FlickeringEnergyField(
    env, 
    flicker_period=FLICKER_PERIOD, 
    invisible_moves=INVISIBLE_MOVES,
    speed=ENERGY_SPEED
)


def init_benchmark_scenario():
    """简单的一维直线追击场景"""
    np.random.seed(42)
    
    # 斑块在x=INITIAL_DISTANCE处，向右移动
    flicker_env.patches = [{
        'x': INITIAL_DISTANCE, 'y': 0.0,  # 在x轴上
        'vx': ENERGY_SPEED, 'vy': 0.0,    # 向右匀速
        'energy': 100.0,
        'id': 0
    }]
    flicker_env.visibility = {0: True}
    flicker_env.step_count = 0
    
    # Agent在原点(0,0)
    return 0.0, 0.0


# 初始化场景
INIT_X, INIT_Y = init_benchmark_scenario()
print(f"  场景: 一维直线追击")
print(f"  Agent在(0,0), 斑块在x={INITIAL_DISTANCE}, 向右速度{ENERGY_SPEED}")


# ==================== Agent 定义 ====================

class SimpleReflexAgent:
    """2节点反射Agent：只能看见时追踪，隐身时停滞"""
    
    def __init__(self, x, y, sensor_range, energy_field):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.sensor_range = sensor_range
        self.energy_field = energy_field
        self.energy = 200.0
        self.alive = True
        
        # 代谢参数
        self.base_metabolism = 0.005
        self.move_cost = 0.01
        self.speed = AGENT_SPEED  # 限制为0.6，只比斑块快20%
    
    def compute_gradient(self):
        """计算能量梯度（用于追踪）"""
        dx = 0.1
        e_curr = self.energy_field.get_energy_at(self.x, self.y, self.sensor_range)
        e_right = self.energy_field.get_energy_at(self.x + dx, self.y, self.sensor_range)
        e_up = self.energy_field.get_energy_at(self.x, self.y + dx, self.sensor_range)
        
        grad_x = (e_right - e_curr) / dx if e_right > 0 else 0
        grad_y = (e_up - e_curr) / dx if e_up > 0 else 0
        
        return grad_x, grad_y
    
    def step(self, dt=0.1):
        """单步决策"""
        if not self.alive:
            return
        
        # 1. 感知能量场
        sensed_energy = self.energy_field.get_energy_at(self.x, self.y, self.sensor_range)
        
        # 2. 决策逻辑（纯反射）
        if sensed_energy > 0.1:  # 看得见 → 梯度上升追踪
            grad_x, grad_y = self.compute_gradient()
            if grad_x != 0 or grad_y != 0:
                # 归一化并移动
                norm = np.sqrt(grad_x**2 + grad_y**2)
                self.vx = (grad_x / norm) * self.speed
                self.vy = (grad_y / norm) * self.speed
            else:
                # 无能量梯度时：停下来（这是2节点Agent的特征）
                self.vx = 0
                self.vy = 0
        else:  # 看不见 → 停止（陷阱1：乌龟策略）
            self.vx = 0
            self.vy = 0
        
        # 3. 运动
        self.x += self.vx
        self.y += self.vy
        
        # 边界
        self.x = np.clip(self.x, 0, 100)
        self.y = np.clip(self.y, 0, 100)
        
        # 4. 能量获取（只有物理碰撞才真正获取能量）
        actual_energy = self.energy_field.get_actual_energy_at(self.x, self.y)
        self.energy += actual_energy
        
        # 5. 代谢消耗
        move_cost = self.move_cost if (abs(self.vx) > 0.1 or abs(self.vy) > 0.1) else 0
        self.energy -= (self.base_metabolism + move_cost)
        
        # 6. 存活检查
        if self.energy <= 0:
            self.alive = False


class MemoryAgent:
    """4节点记忆Agent：使用RNN逻辑，隐身时保持运动向量"""
    
    def __init__(self, x, y, sensor_range, energy_field, decay_factor=0.98):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.sensor_range = sensor_range
        self.energy_field = energy_field
        self.decay_factor = decay_factor
        self.energy = 200.0
        self.alive = True
        
        # 内部状态：记忆上一时刻的运动向量
        self.memory_vx = 0.0
        self.memory_vy = 0.0
        
        # 代谢
        self.base_metabolism = 0.005
        self.move_cost = 0.01
        self.speed = AGENT_SPEED  # 限制为0.6
    
    def compute_gradient(self):
        """计算能量梯度"""
        dx = 0.1
        e_curr = self.energy_field.get_energy_at(self.x, self.y, self.sensor_range)
        e_right = self.energy_field.get_energy_at(self.x + dx, self.y, self.sensor_range)
        e_up = self.energy_field.get_energy_at(self.x, self.y + dx, self.sensor_range)
        
        grad_x = (e_right - e_curr) / dx if e_right > 0 else 0
        grad_y = (e_up - e_curr) / dx if e_up > 0 else 0
        
        return grad_x, grad_y
    
    def step(self, dt=0.1):
        """单步决策（带记忆）"""
        if not self.alive:
            return
        
        # 1. 感知
        sensed_energy = self.energy_field.get_energy_at(self.x, self.y, self.sensor_range)
        
        # 2. 决策逻辑（带记忆）
        if sensed_energy > 0.1:  # 看得见 → 梯度追踪 + 更新记忆
            grad_x, grad_y = self.compute_gradient()
            if grad_x != 0 or grad_y != 0:
                norm = np.sqrt(grad_x**2 + grad_y**2)
                new_vx = (grad_x / norm) * self.speed
                new_vy = (grad_y / norm) * self.speed
            else:
                # 无梯度时：保持之前的运动方向
                new_vx = self.memory_vx if self.memory_vx != 0 else self.speed * 0.5
                new_vy = self.memory_vy if self.memory_vy != 0 else self.speed * 0.5
            
            # 更新记忆
            self.memory_vx = new_vx
            self.memory_vy = new_vy
            self.vx = new_vx
            self.vy = new_vy
        else:  # 看不见 → 使用记忆继续运动！
            # 关键：保持之前的运动向量（衰减很小）
            self.vx = self.memory_vx * self.decay_factor
            self.vy = self.memory_vy * self.decay_factor
        
        # 3. 运动
        self.x += self.vx
        self.y += self.vy
        
        # 边界反弹
        if self.x < 0 or self.x > 100:
            self.memory_vx *= -1
        if self.y < 0 or self.y > 100:
            self.memory_vy *= -1
        
        self.x = np.clip(self.x, 0, 100)
        self.y = np.clip(self.y, 0, 100)
        
        # 4. 能量获取
        actual_energy = self.energy_field.get_actual_energy_at(self.x, self.y)
        self.energy += actual_energy
        
        # 5. 代谢
        move_cost = self.move_cost if (abs(self.vx) > 0.1 or abs(self.vy) > 0.1) else 0
        self.energy -= (self.base_metabolism + move_cost)
        
        # 6. 存活
        if self.energy <= 0:
            self.alive = False


# ==================== 运行基准测试 ====================

def run_benchmark(n_steps=300, n_agents=10):
    """运行基准测试（缩短到300步，确保追得上）"""
    
    results = {
        'simple': {'energies': [], 'survived': 0, 'energy_history': []},
        'memory': {'energies': [], 'survived': 0, 'energy_history': []}
    }
    
    # 创建Agent群（在原点）
    simple_agents = [
        SimpleReflexAgent(
            0.0 + np.random.uniform(-0.5, 0.5), 
            0.0 + np.random.uniform(-0.5, 0.5),
            SENSOR_RANGE,
            flicker_env
        ) for i in range(n_agents)
    ]
    
    memory_agents = [
        MemoryAgent(
            0.0 + np.random.uniform(-0.5, 0.5), 
            0.0 + np.random.uniform(-0.5, 0.5),
            SENSOR_RANGE,
            flicker_env,
            decay_factor=0.98
        ) for i in range(n_agents)
    ]
    
    # 运行
    debug_steps = [0, 20, 25, 30, 50, 60, 70, 80, 90, 100, 150, 200]
    
    for step in range(n_steps):
        # 更新能量场
        flicker_env.step()
        
        # 检查并重生被吃掉的斑块
        flicker_env.respawn_if_needed()
        
        # 运行所有Agent
        for agent in simple_agents:
            agent.step()
        
        for agent in memory_agents:
            agent.step()
        
        # Debug信息
        if step in [0, 25, 29, 30, 31, 50, 100, 150]:
            env_state = flicker_env.get_stats()
            patch_x = flicker_env.patches[0]['x'] if flicker_env.patches else 0
            simple_x = simple_agents[0].x
            memory_x = memory_agents[0].x
            dist_to_patch_simple = abs(patch_x - simple_x)
            dist_to_patch_memory = abs(patch_x - memory_x)
            print(f"  Step {step}: visible={env_state['visible_patches']}, "
                  f"patch_x={patch_x:.1f}, simple_x={simple_x:.1f}(d={dist_to_patch_simple:.1f}), "
                  f"memory_x={memory_x:.1f}(d={dist_to_patch_memory:.1f})")
        
        # 记录
        if step % 50 == 0:
            simple_energy = [a.energy for a in simple_agents if a.alive]
            memory_energy = [a.energy for a in memory_agents if a.alive]
            
            results['simple']['energy_history'].append(np.mean(simple_energy) if simple_energy else 0)
            results['memory']['energy_history'].append(np.mean(memory_energy) if memory_energy else 0)
    
    # 统计
    results['simple']['survived'] = sum(1 for a in simple_agents if a.alive)
    results['memory']['survived'] = sum(1 for a in memory_agents if a.alive)
    
    results['simple']['avg_energy'] = np.mean([a.energy for a in simple_agents])
    results['memory']['avg_energy'] = np.mean([a.energy for a in memory_agents])
    
    # 计算优势比
    if results['simple']['avg_energy'] > 0:
        advantage = (results['memory']['avg_energy'] / results['simple']['avg_energy'] - 1) * 100
    else:
        advantage = float('inf')
    
    return results, simple_agents, memory_agents


# ==================== 主测试 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("v16.1 欺骗性景观基准测试")
    print("=" * 60)
    print(f"感知半径: {SENSOR_RANGE}")
    print(f"闪烁周期: {FLICKER_PERIOD}步")
    print(f"隐身移动: {INVISIBLE_MOVES}步 (距离: {INVISIBLE_MOVES * ENERGY_SPEED})")
    print(f"速度: {ENERGY_SPEED}/步")
    print("-" * 60)
    
    # 验证陷阱1
    print("\n[陷阱1验证] 隐身距离 > 感知范围:")
    print(f"  {INVISIBLE_MOVES}步 × {ENERGY_SPEED}速度 = {INVISIBLE_MOVES * ENERGY_SPEED}")
    print(f"  感知范围: {SENSOR_RANGE}")
    print(f"  比率: {(INVISIBLE_MOVES * ENERGY_SPEED) / SENSOR_RANGE:.1f}x ✓")
    
    # 运行测试
    print("\n运行基准测试...")
    results, simple_agents, memory_agents = run_benchmark(n_steps=1000, n_agents=20)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("基准测试结果")
    print("=" * 60)
    print(f"反射Agent (2节点):")
    print(f"  存活: {results['simple']['survived']}/20")
    print(f"  平均能量: {results['simple']['avg_energy']:.1f}")
    print(f"\n记忆Agent (4节点):")
    print(f"  存活: {results['memory']['survived']}/20")
    print(f"  平均能量: {results['memory']['avg_energy']:.1f}")
    
    advantage = (results['memory']['avg_energy'] / max(results['simple']['avg_energy'], 1)) - 1
    print(f"\n记忆优势: {advantage*100:.1f}%")
    
    # 验证300%基准
    print("\n[基准验证]")
    if advantage >= 3.0:
        print(f"  ✓ 记忆Agent获得 {advantage*100:.0f}% 优势 > 300% 基准线！")
        print("  → 环境筛选压力足够，可以开始演化实验")
    else:
        print(f"  ✗ 记忆Agent仅获得 {advantage*100:.0f}% 优势 < 300% 基准线")
        print("  → 需要调整环境参数增强筛选压力")
    
    # 绘图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['simple']['energy_history'], label='Reflex (2-node)', alpha=0.7)
    plt.plot(results['memory']['energy_history'], label='Memory (4-node)', alpha=0.7)
    plt.xlabel('Steps (×50)')
    plt.ylabel('Avg Energy')
    plt.title('Energy History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    categories = ['Survived', 'Avg Energy (÷10)']
    simple_vals = [results['simple']['survived'], results['simple']['avg_energy']/10]
    memory_vals = [results['memory']['survived'], results['memory']['avg_energy']/10]
    x = np.arange(len(categories))
    width = 0.35
    plt.bar(x - width/2, simple_vals, width, label='Reflex')
    plt.bar(x + width/2, memory_vals, width, label='Memory')
    plt.xticks(x, categories)
    plt.legend()
    plt.title('Final Comparison')
    
    plt.tight_layout()
    plt.savefig('experiments/v16_deceptive_landscape/benchmark_results.png', dpi=150)
    print(f"\n图表已保存到: experiments/v16_deceptive_landscape/benchmark_results.png")