# EOE 机制架构梳理

## 一、环境机制 (Environment Mechanisms)

### 1.1 物理与代谢系统
| 机制 | 开关 | 核心参数 | 作用 |
|------|------|----------|------|
| **基础代谢** | 默认开启 | `metabolic_alpha`, `metabolic_beta` | 能量消耗 |
| **代谢熵增 (v11.0)** | `energy_decay_k` | `k=0.0001` | 能量随时间挥发 |
| **端口干涉成本 (v11.0)** | `port_interference_gamma` | `γ=2.0` | 多端口激活额外消耗 |
| **季节波动 (v11.0)** | `season_jitter` | `±10%` | 参数随机扰动 |

### 1.2 季节与气候系统
| 机制 | 开关 | 核心参数 | 作用 |
|------|------|----------|------|
| **季节循环** | `seasonal_cycle` | `season_length=35` | 夏天→冬天循环 |
| **冬天代谢惩罚** | `winter_metabolic_multiplier` | `1.2x` | 冬天能量消耗增加 |
| **冬天食物减少** | `winter_food_multiplier` | `0.0` | 冬天无食物重生 |
| **热力学庇护所** | `enable_thermal_sanctuary()` | `temp=-10~28°C` | 温度调节生存 |

### 1.3 资源与食物系统
| 机制 | 开关 | 核心参数 | 作用 |
|------|------|----------|------|
| **食物能量** | `food_energy` | `80.0` | 食物提供能量 |
| **食物逃逸** | `food_escape_enabled` | `speed=0.6` | 食物躲避Agent |
| **即时进食** | `immediate_eating` | `bool` | 拾取立即恢复能量 |
| **食物热力学** | `enable_food_thermodynamics()` | `heat_radius=15` | 食物散发热量 |
| **入库税 (v11.0)** | `nest_tax` | `10%` | 贮粮时能量损失 |

### 1.4 空间与感知系统
| 机制 | 开关 | 核心参数 | 作用 |
|------|------|----------|------|
| **传感器范围** | `sensor_range` | `40.0` | 感知距离 |
| **无限世界** | `enable_infinite_mode()` | `chunk_size` | 分块加载 |
| **障碍物系统** | `n_walls` | 数量 | 物理阻挡 |
| **不透明障碍物** | `enable_opaque_obstacles()` | 视距 | 遮挡感知 |

### 1.5 环境记忆系统
| 机制 | 开关 | 核心参数 | 作用 |
|------|------|----------|------|
| **压痕系统** | `enable_stigmergic_friction()` | `friction_coef` | 地面印记 |
| **信息素系统** | `enable_pheromone_system()` | `decay_rate` | 气味标记 |

---

## 二、Agent内部机制 (Agent Internal Mechanisms)

### 2.1 疲劳与睡眠系统
| 机制 | 开关 | 核心参数 | 作用 |
|------|------|----------|------|
| **疲劳系统** | `enable_fatigue_system()` | `max_fatigue=50` | 移动速度递减 |
| **起床饥饿** | `enable_wakeup_hunger` | `tax` | 醒来消耗额外能量 |
| **物理掉落** | `enable_sleep_drop` | `bool` | 睡觉时物品掉落 |

### 2.2 行为端口系统
| 端口 | 描述 | 用途 |
|------|------|------|
| **PORT_MOTION** | 运动输出 | 速度+转向 |
| **PORT_OFFENSE** | 攻击 | 捕食/竞争 |
| **PORT_DEFENSE** | 防御 | 抵御攻击 |
| **PORT_REPAIR** | 修复 | 自我修复 |
| **PORT_SIGNAL** | 信号 | 社会通信 |

### 2.3 形态与发育系统
| 机制 | 开关 | 核心参数 | 作用 |
|------|------|----------|------|
| **形态计算** | `enable_morphological_computation()` | `adhesion_range` | 物理吸附 |
| **发育相变** | `enable_ontogenetic_phase()` | `juvenile_duration` | 幼体保护期 |

---

## 三、演化与选择机制 (Evolution & Selection Mechanisms)

### 3.1 基础选择机制
| 机制 | 参数 | 作用 |
|------|------|------|
| **精英选择** | `elite_ratio=0.2` | 保留top 20% |
| **末位淘汰** | `50%` | 淘汰后50% |
| **能量加权** | `survivors_with_food` | 优先选择吃饱的 |

### 3.2 红皇后竞争 (v0.80)
| 机制 | 参数 | 作用 |
|------|------|------|
| **敌对Agent** | `red_queen=True` | 启用敌对竞争 |
| **敌对刷新** | `rival_refresh_interval` | 定期刷新敌对 |
| **敌对强度** | `n_rivals` | 敌对数量 |

### 3.3 压力梯度熔炉 (v11.1)
| 机制 | 参数 | 作用 |
|------|------|------|
| **文明溢价** | `complexity_premium=1.5` | 复杂度加成 |
| **蟑螂惩罚** | `usage_ratio<0.3` | 惩罚低复杂度 |
| **逆境加成** | `pressure^2` | 高压下复杂度加成 |
| **英雄冢** | `hall_of_fame` | 历史最强保存 |

### 3.4 课程学习 (v4.1)
| 阶段 | 条件 | 目标 |
|------|------|------|
| Phase 1 | Gen 0-49 | 归巢+觅食 |
| Phase 2 | Gen 50-149 | 贮粮激励 |
| Phase 3 | Gen 150+ | 冬天生存 |

---

## 四、机制层级关系

```
┌─────────────────────────────────────────────────────────────────┐
│                     全局配置层                                   │
│  (physics_config.json, Population __init__参数)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     环境层 (Environment)                         │
│  ├─ 物理系统: 代谢/能量/碰撞                                     │
│  ├─ 季节系统: 温度/食物/周期                                     │
│  ├─ 资源系统: 食物/巢穴/信息素                                  │
│  └─ 空间系统: 传感器/障碍/分块                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Agent层 (Agent)                              │
│  ├─ 感知: 传感器输入                                             │
│  ├─ 认知: 神经网络 (SENSOR→META→ACTUATOR)                       │
│  ├─ 行为: PORT_* 输出                                            │
│  └─ 状态: 能量/疲劳/食物/位置                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     演化层 (Population)                          │
│  ├─ 选择: 精英/末位/熔炉                                         │
│  ├─ 突变: 拓扑/权重                                              │
│  └─ 竞争: 红皇后/敌对                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、重构建议

### 5.1 问题诊断

1. **机制耦合严重**: 环境和Agent的机制混在一起
2. **开关泛滥**: 20+个 `enable_*` 方法
3. **参数散落**: config.json, __init__, enable方法三处配置
4. **版本混乱**: v0.74-v11.1混用，无统一版本管理

### 5.2 重构方案

#### 方案A: 模块化重构
```
core/eoe/mechanisms/
├── __init__.py
├── metabolic/          # 代谢机制
│   ├── entropy.py      # 代谢熵增
│   ├── interference.py # 端口干涉
│   └── thermal.py      # 热力计算
├── seasonal/           # 季节机制
│   ├── cycle.py        # 季节循环
│   ├── sanctuary.py    # 庇护所
│   └── jitter.py       # 参数波动
├── agent/              # Agent机制
│   ├── fatigue.py      # 疲劳系统
│   ├── ports.py        # 行为端口
│   └── morphology.py   # 形态计算
└── evolution/          # 演化机制
    ├── selection.py    # 选择策略
    ├── crucible.py     # 压力熔炉
    └── red_queen.py    # 红皇后竞争
```

#### 方案B: 配置驱动
```python
# 统一的机制配置
class MechanismConfig:
    def __init__(self, stage: int = 4):
        self.enabled = {
            'metabolic.entropy': True,
            'metabolic.interference': True,
            'seasonal.cycle': True,
            'seasonal.sanctuary': True,
            'agent.fatigue': True,
            'agent.ports': True,
            'evolution.crucible': True,
            'evolution.red_queen': False,
        }
        
        self.params = {
            'metabolic.alpha': 0.003,
            'metabolic.decay_k': 0.00005,
            'seasonal.length': 35,
            'seasonal.jitter': 0.05,
            'crucible.premium': 1.5,
        }
```

#### 方案C: 渐进式简化
1. **v12**: 合并相似机制 (thermal + seasonal → climate)
2. **v13**: 移除未使用的机制
3. **v14**: 统一配置入口
4. **v15**: 实现方案A或B

### 5.3 优先级

| 优先级 | 问题 | 解决方案 |
|--------|------|----------|
| P0 | 参数不一致 | 统一physics_config.json |
| P1 | enable_*过多 | 合并到 MechanismConfig |
| P2 | 版本混乱 | 添加版本号常量 |
| P3 | 代码耦合 | 模块化拆分 |

---

## 六、当前机制清单

### 总计: 25+ 机制

| 类别 | 数量 | 占比 |
|------|------|------|
| 环境物理 | 4 | 16% |
| 季节气候 | 4 | 16% |
| 资源食物 | 5 | 20% |
| Agent内部 | 4 | 16% |
| 演化选择 | 4 | 16% |
| 辅助系统 | 4 | 16% |