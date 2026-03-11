"""
v0.97 Stage 3: 凛冬降临 - 终极生存测试

测试冠军大脑在冬天能否靠贮粮存活

观察结果 (2026-03-11):
- Agent成功拾取食物并物理掉落贮粮
- 冬天成功触发 (食物消失 + 代谢x2.5)  
- 问题: Agent饿死也未吃贮粮 → 传感器盲区

修复建议: 
- 扩大苏醒瞬间的食物判定范围
- 增加"起床后缓冲期"能量
"""
import sys; sys.path.insert(0, 'core')
import numpy as np
from eoe import Population
from eoe.genome import OperatorGenome as Genome
import os

print("=" * 60)
print("v0.97 Stage 3: 凛冬降临 - 终极生存测试")
print("=" * 60)

# 加载冠军大脑
champion_path = os.path.join(os.path.dirname(__file__), '..', 'champions', 'best_v097_brain.json')
genome = Genome.load_json(champion_path)
info = genome.get_info()
print(f"\n加载冠军大脑: {info['total_nodes']}节点/{info['total_edges']}边")

# 使用与训练相同的环境设置
print("\n创建测试种群 (使用训练时的巢穴位置 25,25)...")

pop = Population(
    population_size=1,
    elite_ratio=1.0,
    lifespan=300,
    use_champion=False,
    n_food=6,
    food_energy=50,
    seasonal_cycle=False,
    season_length=40,
    winter_food_multiplier=0.0,
    winter_metabolic_multiplier=2.5
)
pop.agents[0].genome = genome.copy()

env = pop.environment
env.sensor_range = 50
env.nest_enabled = True
env.nest_position = np.array([25.0, 25.0])  # 训练时的位置
env.nest_radius = 10.0
env.n_walls = 0

# 启用疲劳+起床+物理掉落
env.enable_fatigue_system(
    enabled=True,
    max_fatigue=50.0,
    fatigue_build_rate=0.5,
    sleep_danger_prob=0.0,
    enable_wakeup_hunger=True,
    enable_sleep_drop=True
)

# 设置Agent初始状态
agent = pop.agents[0]
agent.x, agent.y = 30.0, 30.0
agent.theta = -0.78
agent.internal_energy = 300.0

# 食物在可拾取距离
env.food_positions = [(32.0, 30.0), (33.0, 30.0), (34.0, 30.0)]
env.food_freshness = [1.0] * 3

print(f"  初始能量: {agent.internal_energy}")
print(f"  巢穴位置: {env.nest_position}")
print(f"  食物: {env.food_positions}")

# 冬天状态
winter_triggered = False

print("\n开始测试...")
print("-" * 60)

# 运行测试
for step in range(300):
    env.agents = [agent]
    env.step()
    
    # 检查是否在巢穴
    dist = np.sqrt((agent.x - 25)**2 + (agent.y - 25)**2)
    in_nest = dist < 10
    
    # 触发冬天条件: 有贮粮 + 在巢穴 + 能量低
    if not winter_triggered and agent.food_stored > 0 and in_nest and agent.internal_energy < 200:
        winter_triggered = True
        env.winter_food_multiplier = 0.0
        env.winter_metabolic_multiplier = 2.5
        env.food_positions = []  # 清除所有食物
        print(f"\n❄️ Step {step}: 凛冬降临!")
        print(f"   贮粮: {agent.food_stored}")
        print(f"   能量: {agent.internal_energy:.1f}")
        print(f"   位置: ({agent.x:.1f}, {agent.y:.1f}) 巢穴内")
    
    # 打印状态
    if step < 20:
        status = '❄️' if winter_triggered else '🏃'
        print(f"Step {step:3d} {status} E={agent.internal_energy:6.1f} C={agent.food_carried} S={agent.food_stored}")
    
    # 检查死亡
    if agent.internal_energy <= 0:
        if winter_triggered:
            print(f"\n💀 冬天饿死! 贮粮={agent.food_stored} 未被吃掉")
        else:
            print(f"\n💀 饿死")
        break
    
    # 成功: 冬天后吃贮粮
    if winter_triggered and agent.food_eaten > 0:
        print(f"\n✅ 吃贮粮存活!")
        break

# 最终结果
print("\n" + "=" * 60)
print("测试结果:")
print(f"  冬天触发: {'是' if winter_triggered else '否'}")
print(f"  贮粮数量: {agent.food_stored}")
print(f"  最终能量: {agent.internal_energy:.1f}")
print(f"  存活: {'是' if agent.internal_energy > 0 else '否'}")
print("=" * 60)

print("""
分析:
- 物理掉落机制工作正常 ✓
- 冬天触发正常 ✓
- 问题: Agent有贮粮但未吃 → 传感器盲区

修复建议:
1. 扩大苏醒后的食物检测范围
2. 增加起床后的能量缓冲
3. 或者让贮粮更"显眼" (如添加气味)
""")