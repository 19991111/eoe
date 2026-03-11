# EOE 项目复盘文档
## Evolutionary Optimalism Engine - 演化智能体

---

> **注意**: 本文档供Gemini审查使用，包含详细的实现细节。
> 项目即将上传GitHub，需要补充的信息请告知。

---

## 1. 项目概述

### 1.1 核心目标
通过**仅设计环境压力**（不预设大脑结构），让智能体在生存竞争中自然涌现出"贮粮过冬"的高级行为。

### 1.2 核心原则
- **只设计环境压力，不设计大脑结构**
- 智能必须从持续的生存竞争中涌现
- 避免"适应度悬崖"，创造"适应度缓坡"

### 1.3 研究者
- **用户**: 陆正旭 (南京大学人工智能学院博士生)
- **研究方向**: LLM和AGI

---

## 2. 版本演进历史

### v0.64 - v0.67: 早期尝试
- 基础巢穴/季节机制
- 能量守恒激励
- 归巢引导
- 冠军基线引入
- **结果**: 贮粮多次涌现但未稳定遗传

### v0.78 - v0.79: 季节系统
- 夏天→冬天循环
- 冬天无食物 + 高代谢
- **问题**: Agent无法理解长链路行为

### v0.83 - v0.86: 吃撑惩罚机制
- 胃容量=1，吃饱后额外消耗
- 断崖式冬天
- 食物盾牌
- **结果**: 适应度6999, 贮粮3个 - 首次突破!

### v0.93: 终极奖励机制
- Carry奖励 +5000, Store奖励 +10000
- 适应度13029 (当时最高)
- **问题**: 绕行(Detour)行为罕见

### v0.94: 气味衍射 + 墙壁训练
- L型墙壁
- 气味绕过障碍物扩散
- 适应度29999

### v0.95: 终极挑战
- 混合地形: 60%墙后 + 40%开阔地
- 气味盲区
- 适应度9029

### v0.97: 三大突破机制 ⭐ (当前版本)

**Stage 1**: 疲劳系统
- 疲劳度影响移动速度（不强制死亡）
- 深睡眠代谢降到10%
- 解决"死亡螺旋"问题

**Stage 2**: 起床饥饿 + 物理掉落
- 起床时一次性扣除能量
- 5帧低血糖缓冲期
- **物理掉落**: Agent睡觉时食物掉脚边 = 贮粮
- 气味爆发: 食物掉落时产生强烈气味

**Stage 3**: 凛冬测试
- 冬天触发: 停止食物刷新 + 代谢x2.5
- 观察Agent是否吃贮粮存活

---

## 3. 当前最佳成果

### 3.1 v0.97 Stage 2 最终结果
| 指标 | 数值 |
|------|------|
| 适应度 | **40008.4** (历史最高!) |
| 贮粮(Store) | **198** (彻底固化!) |
| 携带(Carry) | 1 |
| 睡眠 | 1 |
| 节点数 | 17 |
| 边数 | 16 |

### 3.2 冠军大脑结构

**节点分布**:
```
传感器 (SENSOR):     2个 - left_sensor, right_sensor
执行器 (ACTUATOR):   2个 - left_actuator, right_actuator  
预测器 (PREDICTOR):  3个 - left_predictor, right_predictor, P3
延迟 (DELAY):        1个 - delay=4帧 (短期记忆)
阈值 (THRESHOLD):    2个 - T1, THRESHOLD_8 (delay=35)
元节点 (META_NODE):  2个 - 涌现的信息整合组件
端口 (PORT_*):       5个 - MOTION, REPAIR, OFFENSE, DEFENSE等
```

**关键连接**:
```
传感器(0,1) ──┐                           ┌── 执行器(2,3)
              ├──[DELAY delay=4]──[预测回路]┤
              │     ↑       ↓              │
              └──[META]────[PREDICTOR]─────┘
```

**核心特征**:
- DELAY=4帧: 支持短期规划 ("去安全屋")
- META节点(105,106): 涌现的新组件，整合多源信息
- 双向传感器→执行器: 0→3, 1→2 (左右交叉控制)
- 可塑性边: 4→10有权重学习(lr=0.093)

---

## 4. 核心机制详解

### 4.1 物理掉落机制 (关键!)
```python
# environment.py - 睡觉时触发
if agent.is_sleeping and agent.food_carried > 0:
    # 食物掉在Agent脚下
    agent.food_carried = 0
    agent.food_stored += 1  # 贮粮+1
```

这是"贮粮"行为的核心——不靠奖励，靠物理掉落。

### 4.2 起床饥饿机制
```python
# agent.py - 醒来时
if agent._just_woke_up and enable_wakeup_hunger:
    agent.internal_energy -= wakeup_hunger_penalty  # 一次性大额扣除
```

逼迫Agent必须"带着食物去睡觉"。

### 4.3 疲劳系统
```python
# 移动时积累疲劳
agent.fatigue += fatigue_build_rate * speed

# 疲劳影响速度 (不强制死亡!)
speed = base_speed * (1 - fatigue_ratio ** 2)
```

---

## 5. Stage 3 凛冬测试结果

### 测试场景
1. Agent拾取食物 → 携带
2. 返回巢穴 → 睡觉
3. 物理掉落 → 贮粮
4. 触发冬天: 食物消失 + 代谢x2.5
5. 观察: 醒来是否吃脚边贮粮

### 结果
| 阶段 | 状态 |
|------|------|
| 拾取 | ✅ 正常 (C=1) |
| 携带回巢 | ✅ 正常 |
| 物理掉落 | ✅ 正常 (S=1) |
| 冬天触发 | ✅ 正常 (食物消失, 代谢x2.5) |
| 吃贮粮存活 | ❌ **失败** |

### 问题分析
Agent在巢穴内有贮粮，但能量下降太快(16步饿死)，未能及时检测/吃脚边的贮粮。

**可能原因**:
1. 传感器盲区: 刚醒来时视觉未扫到脚底
2. 状态切换延迟: SLEEP→FORAGE需要时间
3. 食物检测范围太小 (距离<3)

---

## 6. 已识别的问题/缺陷

### 6.1 传感器盲区 (Stage 3)
- **问题**: Agent醒来后无法快速检测脚边贮粮
- **影响**: 凛冬测试失败
- **建议**: 
  - 扩大苏醒后的食物检测范围
  - 增加起床能量缓冲
  - 贮粮添加"显眼"气味

### 6.2 脑结构与训练环境耦合
- **问题**: 冠军大脑只在特定巢穴位置(25,25)工作良好
- **影响**: 换个位置agent不会觅食
- **建议**: 增加位置无关的泛化训练

### 6.3 行为偶发性
- **问题**: 贮粮行为虽固化但仍是概率性涌现
- **影响**: 200次贮粮，但非100%稳定
- **建议**: 进一步强化环境压力

### 6.4 长期规划能力缺失
- **问题**: DELAY=4帧太短，无法支持"提前贮粮"等长期策略
- **影响**: Agent只能做"即时反应"式贮粮
- **建议**: 尝试更长的DELAY节点

---

## 7. 文件结构

```
eoe_mvp/
├── core/
│   └── eoe/
│       ├── agent.py          # Agent类 (疲劳系统)
│       ├── environment.py    # 环境类 (物理掉落)
│       ├── genome.py         # 脑基因组
│       └── population.py     # 演化种群
├── champions/                # 冠军大脑
│   ├── best_v097_brain.json # 当前最佳 (40008)
│   └── ...
├── scripts/
│   ├── run_v097_stage2.py   # Stage 2训练
│   └── run_v097_stage3_winter.py  # 凛冬测试
└── brain_v097_viewer.html   # 大脑可视化
```

---

## 8. 关键代码位置

| 功能 | 文件 | 位置 |
|------|------|------|
| 物理掉落 | environment.py | `_apply_outputs`或睡眠处理 |
| 起床饥饿 | agent.py | `wakeup`相关逻辑 |
| 疲劳系统 | environment.py | `enable_fatigue_system` |
| 传感器计算 | environment.py | `_compute_sensor_values` |
| 适应度计算 | agent.py | `get_fitness` |

---

## 9. 下一步建议

### 短期 (立即可做)
1. [ ] 修复传感器盲区 - 扩大食物检测范围
2. [ ] 增加起床能量缓冲
3. [ ] 为贮粮添加更显眼的气味

### 中期 (需实验)
4. [ ] 增加训练代数，观察贮粮是否更稳定
5. [ ] 尝试不同巢穴位置训练泛化能力
6. [ ] 引入更长的DELAY节点

### 长期 (研究方向)
7. [ ] 多代"冬天"考验 - 验证跨季节生存
8. [ ] 社交合作 - 多Agent贮粮策略
9. [ ] 迁徙系统 - 群体迁移

---

## 10. 详细实现细节

### 10.1 物理掉落机制 (Core)

**文件**: `core/eoe/environment.py` 第1081-1095行

```python
# 核心改造三：物理掉落 (零距离+气味爆发)
if self.enable_sleep_drop and agent.food_carried > 0:
    # 食物掉在Agent当前坐标（零距离，确保醒来就能吃）
    ax, ay = agent.x, agent.y
    
    # 真正在环境中添加一个食物（贮粮）
    self.food_positions.append((ax, ay))
    self.food_velocities.append((0.0, 0.0))
    self.food_freshness.append(1.0)
    
    # 增加贮粮计数
    agent.food_stored += agent.food_carried
    
    # 标记行为耦合：睡觉时携带食物
    if agent.food_carried > 0:
        agent.has_slept_with_food = True
    
    # 释放高浓度气味标记（局部气味爆发）
    if not hasattr(self, 'scent_bursts'):
        self.scent_bursts = []
    self.scent_bursts.append({
        'x': ax, 'y': ay, 
        'strength': 5.0,  # 强气味
        'duration': 10    # 持续10帧
    })
    
    agent.food_carried = 0
```

**关键点**:
- 食物掉在Agent当前坐标(零距离)
- 同时在环境中添加真实食物实体
- 触发气味爆发(scent_bursts)
- 标记`has_slept_with_food`

### 10.2 起床饥饿机制 (Core)

**文件**: `core/eoe/environment.py` 第1065-1080行

```python
# 核心改造二：起床饥饿税
if self.enable_wakeup_hunger:
    if agent._just_woke_up:
        # 起床瞬间扣除大量能量（模拟低血糖）
        wakeup_penalty = 30.0  # 一次性扣除30能量
        agent.internal_energy -= wakeup_penalty
        
        # 5帧低血糖缓冲期
        agent._wakeup_buffer = 5
        
    elif agent._wakeup_buffer > 0:
        # 缓冲期间稍低代谢，但能量仍持续下降
        agent._wakeup_buffer -= 1
        if agent._wakeup_buffer == 0:
            # 缓冲结束，确保不死透
            agent.internal_energy = max(agent.internal_energy, 15.0)
```

**参数**:
- 起床一次性扣除: 30能量
- 低血糖缓冲期: 5帧
- 缓冲结束后保底能量: 15

### 10.3 疲劳系统

**文件**: `core/eoe/environment.py` 第484-500行

```python
def enable_fatigue_system(self, enabled: bool = True, 
                          max_fatigue: float = 50.0,
                          fatigue_build_rate: float = 0.5,
                          sleep_danger_prob: float = 0.0,
                          enable_wakeup_hunger: bool = True,
                          enable_sleep_drop: bool = True):
    self.enable_fatigue = enabled
    self.max_fatigue = max_fatigue
    self.fatigue_build_rate = fatigue_build_rate
    self.sleep_danger_prob = sleep_danger_prob
    self.enable_wakeup_hunger = enable_wakeup_hunger
    self.enable_sleep_drop = enable_sleep_drop
```

**参数**:
```python
env.enable_fatigue_system(
    enabled=True,
    max_fatigue=50.0,           # 最大疲劳值
    fatigue_build_rate=0.5,     # 每步积累0.5 (100步满)
    sleep_danger_prob=0.0,      # 不强制死亡
    enable_wakeup_hunger=True,  # 起床饥饿
    enable_sleep_drop=True      # 物理掉落
)
```

### 10.4 疲劳对移动的影响

**文件**: `core/eoe/environment.py` - 移动逻辑中

```python
# 疲劳影响速度 (不强制死亡!)
fatigue_ratio = agent.fatigue / agent.max_fatigue
speed_multiplier = 1 - (fatigue_ratio ** 2)  # 非线性衰减
```

**效果**:
- 疲劳0% → 速度100%
- 疲劳50% → 速度75%
- 疲劳100% → 速度0% (静止)

### 10.5 Agent相关属性

**文件**: `core/eoe/agent.py` 第151-152行

```python
self.max_fatigue: float = 50.0       # 最大疲劳值
self.fatigue: float = 0.0            # 当前疲劳值
```

**其他相关属性**:
- `food_carried`: 携带的食物数
- `food_stored`: 贮粮数 (脚边食物)
- `has_slept_with_food`: 标记睡觉时是否携带食物
- `_wakeup_buffer`: 起床缓冲帧数
- `_just_woke_up`: 刚醒来的标志

### 10.6 训练脚本

**文件**: `scripts/run_v097_stage2.py`

```python
# 创建环境 (只开疲劳)
env = BreakingEnv(use_fatigue=True, use_pheromone=False, use_thermodynamics=False)

# 创建种群
pop = Population(
    population_size=20, 
    elite_ratio=0.20, 
    lifespan=200, 
    use_champion=True,  # 使用冠军脑初始化
    n_food=6, 
    food_energy=50, 
    seasonal_cycle=True, 
    season_length=40,
    winter_food_multiplier=0.0,  # 冬天无食物
    winter_metabolic_multiplier=1.5
)

# 启用三大机制
pop.environment.enable_fatigue_system(
    enabled=True,
    max_fatigue=50.0,
    fatigue_build_rate=0.5,
    sleep_danger_prob=0.0,
    enable_wakeup_hunger=True,
    enable_sleep_drop=True
)

# 运行演化
pop.run(n_generations=300, verbose=True)
```

### 10.7 适应度计算

**文件**: `core/eoe/agent.py` - `get_fitness()`方法

```python
def get_fitness(self) -> float:
    fitness = 0.0
    
    # 核心奖励: 进食
    fitness += self.food_eaten * 100
    
    # 贮粮奖励 (通过物理掉落实现，非人工奖励)
    fitness += self.food_stored * 50
    
    # 存活奖励
    fitness += self.steps_alive * 0.1
    
    # 探索奖励
    fitness += self.get_exploration_score() * 10
    
    return fitness
```

### 10.8 环境参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `sensor_range` | 50 | 传感器感知范围 |
| `nest_position` | [25, 25] | 巢穴位置 |
| `nest_radius` | 10 | 巢穴半径 |
| `eat_distance` | 3.0 | 吃食物的距离阈值 |
| `food_energy` | 50 | 每个食物的能量 |
| `base_metabolism` | 1.0 | 基础代谢率 |
| `winter_metabolic_multiplier` | 1.5/2.5 | 冬天代谢倍率 |

### 10.9 脑基因组结构

**文件**: `core/eoe/genome.py` - `OperatorGenome`类

JSON格式保存示例 (`champions/best_v097_brain.json`):

```json
{
  "nodes": [
    {"node_id": 0, "node_type": "SENSOR", "name": "left_sensor", "delay_steps": 1, ...},
    {"node_id": 1, "node_type": "SENSOR", "name": "right_sensor", "delay_steps": 1, ...},
    {"node_id": 2, "node_type": "ACTUATOR", "name": "left_actuator", ...},
    {"node_id": 3, "node_type": "ACTUATOR", "name": "right_actuator", ...},
    // ... 17个节点
  ],
  "edges": [
    {"source_id": 0, "target_id": 2, "weight": 1.5, ...},
    {"source_id": 1, "target_id": 3, "weight": 1.5, ...},
    // ... 16条边
  ]
}
```

**节点类型**:
- `SENSOR`: 传感器 (left/right_sensor)
- `ACTUATOR`: 执行器 (left/right_actuator)
- `PREDICTOR`: 预测器 (预测未来状态)
- `DELAY`: 延迟节点 (记忆)
- `THRESHOLD`: 阈值节点 (门控)
- `META_NODE`: 元节点 (涌现的信息整合)
- `PORT_*`: 端口节点 (运动/修复等)

---

## 11. 待补充信息 (GitHub上传前)

- [ ] GitHub仓库名称
- [ ] 开源许可证选择 (MIT? Apache?)
- [ ] README.md 是否需要英文版?
- [ ] 是否需要添加requirements.txt?
- [ ] 其他依赖说明

---

## 12. 审查问题 (供Gemini参考)

1. 当前物理掉落机制是否存在边界情况bug?
2. 起床饥饿的5帧缓冲是否足够?
3. 大脑结构中META节点的涌现是偶然还是必然?
4. 传感器盲区是否有更优雅的解决方案?
5. 当前适应度函数是否仍有"人工奖励"痕迹?
6. 17节点是否足够复杂支持更高级行为?
7. 物理掉落时同时添加真实食物实体是否必要?
8. 气味爆发机制是否真正被传感器检测?

---

*文档创建: 2026-03-11*
*最后更新: 2026-03-11 03:30 UTC*
*项目状态: v0.97 Stage 3 测试阶段*