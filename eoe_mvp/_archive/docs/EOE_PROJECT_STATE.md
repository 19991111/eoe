# EOE 项目状态文档

> 跨会话同步技术断点 | 最后更新: 2026-03-12 UTC

---

## 1. 项目愿景与哲学 (Project DNA)

### 核心定义
- **项目名称**: EOE (Evolving Organisms Explorer) - 演化智能体探索
- **项目类型**: 演化计算 / 人工生命 / 生成式AI

### 第一准则
> **只设计环境压力，不设计大脑结构**
> 
> 智能必须从持续的生存竞争中涌现，而非人为预设。预设节点类型(SENSOR, DELAY等)是"先天结构"，加速自然选择，但不应预设特定神经回路。

### 演化阶段

| 阶段 | 名称 | 核心机制 | 状态 |
|------|------|----------|------|
| 阶段一 | 爬行脑 | 感知→移动→进食 | ✅ |
| 阶段二 | 哺乳动物脑 | 季节系统、巢穴贮粮 | ✅ |
| 阶段三 | 哺乳动物进阶 | 疲劳系统、热力学庇护所 | ✅ |
| 阶段四 | 红皇后竞争 | 生态位分化、多智能体博弈 | 🔄 |

### 当前版本: v10.4
- 主题: 红皇后竞争与文明觉醒
- 关键突破: 非线性代谢缩放 (sqrt节点数 × alpha)
- 里程碑: 贮粮行为首次大规模涌现 (680次/40代)

---

## 2. 物理常数与环境配置 (Physics Manifest)

### 基准代谢常数
```python
metabolic_alpha = 0.003  # 从0.05降至，破解"智力税"黑洞
metabolic_beta = 0.003
```

### 能耗公式 (非线性缩放)
```python
# v10.4: 异速生长定律 - 大脑功耗增长率远低于神经元增加速度
node_cost = sqrt(node_count) * metabolic_alpha
edge_cost = sqrt(edge_count) * metabolic_beta
total_metabolic = node_cost + edge_cost + node_specific_cost
```

### 季节系统
```python
season_length = 25  # 帧/季节
winter_food_multiplier = 0.0  # 冬天无新食物
winter_metabolic_multiplier = 1.3 - 1.8  # 冬天代谢倍率
```

### 巢穴机制 (Nest)
```python
nest_enabled = seasonal_cycle  # 有季节才启用
nest_position = (width * 0.15, height * 0.15)  # 左上角
nest_radius = 10.0
nest_metabolic_reduction = 0.3  # 巢穴内代谢降至30%
nest_defense_immunity = True   # 巢穴内免疫抢劫
```

### 战斗系统 (v10.1 - v10.4)
```python
PORT_OFFENSE_cost = 3.0  # 攻击能耗 ×3 (暴力代价)
PORT_DEFENSE_cost = 0.65  # 防御能耗 ×0.65 (防御优势)
defense_effectiveness = 0.8  # 防御减伤80%
corpse_energy = food_energy × 5  # 击杀奖励 (高风险高回报)
carry_buff = 1.3  # 携带食物时防御+30%
```

### 携带决策 (v10.4)
```python
# 废除硬编码阈值
# 只在能量极低时(<20%)直接吃，否则携带回巢
eat_threshold = 0.20
```

---

## 3. 种群进化现状 (Evolutionary Status)

### 大脑规格演进
| 阶段 | 节点数 | 边数 | 密度 | META节点 |
|------|--------|------|------|----------|
| 阶段一 | 11 | ~6 | 0.5 | 0 |
| 阶段四(v10.1b) | 27 | 54 | 2.0 | 8 |
| 阶段四(v10.4) | 31 | 60 | 1.9 | 12 |

### 神经模块
- **SENSOR**: 基础感知 (2个)
- **ACTUATOR**: 运动输出 (2个)
- **META_NODE**: 内部状态精算 (12个) - 涌现自我监控
- **PREDICTOR**: 预测回路 (4个) - 时间序列预测
- **DELAY**: 时间记忆 (2个) - 短期规划
- **PORT_***: 端口节点 - 行为分化

### 生态位分化 (阶段四目标)
```
状态: ✅ 已达成三足鼎立
- 捕食者 (predator): ~20-30%
- 采集者 (forager): ~60-70%  
- 贮粮者 (storer): ~5-10%
```

### 最高记录
```
贮粮次数: 680次/40代 (v10.4)
最高适应度: 13029 (v0.93历史记录)
```

---

## 4. 技术栈与性能 (Hardware & Perf)

### 硬件环境
```yaml
CPU: AMD EPYC 128核
GPU: NVIDIA A100 × 4 (80GB each)
Python: 3.11+
内存: 512GB+
```

### 性能分析
- **当前运行**: 100% CPU
- **瓶颈**: `environment.step()` 占96%运行时
  - 传感器计算: 15%
  - 神经网络: 7%
  - 食物逃逸: 19%

### 优化方案
```python
# 决策: 使用Numba JIT优化，而非GPU加速
# 原因: 小模型 + GPU PCIe延迟 > CPU计算

# 待优化函数:
# 1. fast_distance() - 距离计算
# 2. neural_network_forward() - 前向传播
# 3. scent_diffusion() - 气味扩散
```

---

## 5. 已解决的"死亡陷阱" (Bug Log)

### ✅ 已修复

| Bug | 描述 | 修复方案 |
|-----|------|----------|
| 智力税黑洞 | metabolic=0.05导致27节点大脑快速死亡 | 降至0.003 + sqrt非线性缩放 |
| 硬编码阈值 | 50%能量阈值在0.05代谢下达不到 | 废除阈值，改为极简20% |
| 夏季误判 | 夏季死亡被误判为"冬季耗竭" | 新增`death_by_winter_exhaustion`分类 |
| 内存不同步 | JSON配置修改后物理引擎未读取 | 添加配置热更新检查 |
| 通信断连 | LLM API请求Broken pipe | 增加重试机制 |
| Chunk Hash | Python Hash随机化导致地图不一致 | 使用确定性hash算法 |

### ⚠️ 仍需关注
- 生态位在极低代谢下容易坍缩为单一类型
- 压力测试仍存在90%+适应度跌幅

---

## 6. 下一步待办事项 (The Next Step)

### 短期目标 (阶段四完成)
- [ ] 实现稳定的三生态位分化 (当前: 条件依赖)
- [ ] 压力测试跌幅 < 50% (当前: 90%+)
- [ ] 泛化测试贮粮 > 200 (当前: 680 ✅)

### 中期目标 (阶段五)
- [ ] LLM Demiurge 自动化调参
- [ ] 超大规模种群测试 (50+ Agent)
- [ ] 多环境并行演化

### 长期目标 (通用智能)
- [ ] 涌现出真正的规划能力
- [ ] 跨环境知识迁移
- [ ] 自我模型构建

---

## 7. 机制设计细节 (Mechanism Design)

### 7.1 能量经济学

```python
# 食物能量流
food_energy = 30.0  # 基础能量
eta_plant = 0.8     # 植物吸收率 (80%)
eta_meat = 0.7      # 肉食吸收率 (70%)

# 获取能量
absorbed = food_energy * eta_plant
wasted = food_energy * (1 - eta_plant)
agent.internal_energy += absorbed

# 野外buff (v10.3)
if dist_to_nest > 15.0:
    food_energy *= 1.5  # 远距离食物高能奖励
```

### 7.2 战斗系统

```python
def _resolve_battle(attacker, defender):
    # 攻击修正
    if attacker.port_offense > 0.7:
        attack *= 1.5  # 高频攻击增强
    
    # 防御修正 (v10.1)
    if defender.food_carried > 0:
        defense *= 1.3  # 携带buff
    
    effective_defense = defense * 0.8  # 80%减伤
    
    # 胜者判定
    if attack > (1.0 - effective_defense):
        # 攻击方获胜
        steal = defender.energy * 0.7
        attacker.energy += steal * attacker.eta_meat
        
        # 暴力代价
        attacker.fatigue += 0.2  # 攻击疲劳
        attacker.energy *= 0.9   # 10%反作用力
    
    # 防御成功惩罚 (v10.1)
    if defense > 0.3 and attack_failed:
        attacker.fatigue += 0.4  # 双倍疲劳
```

### 7.3 生态避难所 (Refugia)

```python
# v10.3: 地图边缘5%区域
refugia_margin = width * 0.05

in_refugia = (
    (x < refugia_margin and y < refugia_margin) or  # 左下角
    (x > width - refugia_margin and y > height - refugia_margin)  # 右上角
)

if in_refugia:
    metabolic_cost *= 0.5  # 避难所代谢减半
```

---

## 8. 环境与进化逻辑 (Environment & Evolution)

### 8.1 环境类层级

```
Environment
├── 季节系统 (SeasonSystem)
│   ├── 温度计算 (_update_temperature)
│   └── 食物重生 (_respawn_food)
├── 热力学庇护所 (ThermalSanctuary)
│   └── 温度效应 (_apply_temperature_effects)
├── 疲劳系统 (FatigueSystem)
│   └── 疲劳累积
├── 形态计算 (MorphologicalComputation)
│   ├── 吸附检测 (_check_adhesion_collision)
│   └── 能量卸载 (_attempt_discharge)
└── 战斗系统 (BattleSystem)
    └── 能量夺取 (_attempt_energy_theft)
```

### 8.2 进化循环

```
for generation in range(max_generations):
    # 1. 环境交互
    for step in range(lifespan):
        environment.step()
        
        # 每步执行:
        # - 传感器计算
        # - 神经网络前向
        # - 行为执行
        # - 能量代谢
        # - 战斗判定
        # - 赫布学习
    
    # 2. 适应度计算
    fitness = calculate_fitness(agents)
    
    # 3. 精英选择
    elite = select_top(fitness, elite_ratio)
    
    # 4. 繁殖变异
    offspring = mutate(elite, population_size)
    
    # 5. 环境重置
    environment.reset()
```

### 8.3 关键演化驱动力

| 压力源 | 机制 | 演化响应 |
|--------|------|----------|
| 能量危机 | 代谢消耗 > 能量获取 | 提升进食效率 |
| 季节变化 | 冬天无食物 | 贮粮行为涌现 |
| 捕食竞争 | 能量抢夺 | 防御/逃跑分化 |
| 极端压力 | LLM干预 | 动态重塑能力 |

---

## 附录: 文件结构

```
eoe_mvp/
├── core/
│   └── eoe/
│       ├── agent.py        # Agent类
│       ├── environment.py  # 物理环境
│       ├── genome.py       # 基因组
│       ├── population.py   # 种群管理
│       └── node.py         # 节点类型
├── scripts/
│   ├── llm_demiurge_loop.py  # LLM调参
│   └── run_stage*.py         # 各阶段测试
├── champions/              # 冠军大脑
│   ├── stage1_best_brain.json
│   └── ...
├── physics_config.json    # 物理参数
└── EOE_PROJECT_STATE.md   # 本文档
```

---

*End of Document*