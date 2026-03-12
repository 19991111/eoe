# EOE 项目追踪器

## 项目概述
演化智能体 (Evolving Organism Ensemble) - 通过自然选择涌现智能

## 阶段规划

### 阶段一：爬行脑 (Reptile Brain) ✅ 完成
- **目标**: 感知→移动→进食 基础闭环
- **核心机制**: 即时进食、代谢消耗
- **冠军大脑**: champions/stage1_best_brain.json
- **节点数**: 11

### 阶段二：哺乳动物脑 (Mammal Brain) ✅ 完成
- **目标**: 跨季节贮粮、时间规划
- **核心机制**: 季节系统、巢穴贮粮、食物逃逸
- **学习方法**: 课程学习 (15代无逃逸 + 50代有逃逸)
- **考核结果**: 3/3 通过 (贮粮104次)
- **节点数**: 22 | **边数**: 42

### 阶段三：高级生理与微观管理 (Advanced Physiology) ✅ 完成
- **目标**: 自我调节 (Homeostasis)
- **核心机制**: 
  - 疲劳系统 (fatigue_build_rate: 0.2)
  - 热力学庇护所 (80%疲劳+30%代谢惩罚)
  - 深度睡眠代谢补偿
- **考核结果**: 2.5/3 (趋温性✅ 贮粮✅ 疲劳感知2.8接近阈值)
- **大脑复杂度**: 31节点 / 60边 (+41%)
- **关键发现**: Agent从"蛮干"变为"能量管理大师"
- **冠军大脑**: champions/stage3_champion.json

### 阶段四：红皇后假说 (Red Queen) 🔄 待开启
- **目标**: 竞争演化 + 泛化能力
- **核心机制**: 
  - 多Agent竞争 (22个Agent)
  - Agent间物理碰撞和伤害
  - 资源掠夺、巢穴争夺
  - PORT_OFFENSE/PORT_DEFENSE激活
- **考核指标**:
  1. 跨环境泛化 (5个随机地图测试)
  2. 对抗性生存率 (生态位分化)
  3. 造物主压力测试 (10代极端压力)

### 阶段五：LLM Demiurge 📋 待实现
- **目标**: AI辅助进化
- **核心机制**: LLM分析和建议

---

## 物理配置 (physics_config.json)

```json
{
  "metabolic_alpha": 0.003,
  "metabolic_beta": 0.003,
  "sensor_range": 40,
  "season_length": 35,
  "winter_metabolic_multiplier": 1.2,
  "fatigue_build_rate": 0.15,
  "food_energy": 80.0,
  "enable_fatigue_system": true,
  "enable_thermal_sanctuary": true
}
```

## 关键文件

| 文件 | 用途 |
|------|------|
| core/eoe/agent.py | Agent智能体 |
| core/eoe/environment.py | 生存环境 |
| core/eoe/population.py | 种群演化 |
| core/eoe/genome.py | 基因组 |
| champions/stage1_best_brain.json | 阶段一冠军 |
| champions/stage2_champion.json | 阶段二冠军 |

## 下次实验建议

阶段三测试脚本: `scripts/run_stage3.py`