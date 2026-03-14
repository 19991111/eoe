# 代码优化方案
## 审查日期: 2026-03-13

---

## 1. 当前代码状态

### 1.1 项目规模
```
核心代码: ~16,900 行 (core/eoe/ + scripts/)
主要文件:
├── core/eoe/integrated_simulation.py   (325行) - 主循环
├── core/eoe/batched_agents.py          (334行) - Agent管理
├── core/eoe/environment_gpu.py         (528行) - 环境场
├── config/mechanisms.yaml              (125行) - 机制配置
└── scripts/test_neural_evolution_v13.py (460行) - 演化脚本
```

### 1.2 性能基准
- 500 Agent, 1500步 → ~4秒/代
- 1000 Agent, 1000步 → ~2.7秒/代
- GPU利用率: ~32%

---

## 2. 问题分析

### 2.1 高优先级 (P0) - 影响功能

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| P0-1 | 机制配置未实际使用 | all files | config/mechanisms.yaml 定义了开关,但代码未读取 |
| P0-2 | brain_masks 未应用 | batched_agents.py | 30%连接率掩码从未生效 |
| P0-3 | 能量守恒检查每步执行 | integrated_simulation.py | 不必要的GPU同步开销 |

### 2.2 中优先级 (P1) - 影响性能

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| P1-1 | get_sensors 多次GPU调用 | batched_agents.py | 每步9次场采样,可合并 |
| P1-2 | 神经网络forward未使用batch | batched_agents.py | 单次bmm,可用torch.compile加速 |
| P1-3 | ISF diffusion 每步计算 | environment_gpu.py | 可用CUDA kernel融合 |
| P1-4 | 无早停优化 | integrated_simulation.py | 全部死亡后仍继续循环 |

### 2.3 低优先级 (P2) - 代码质量

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| P2-1 | 硬编码参数散落各处 | multiple | 能量/代谢参数应从config加载 |
| P2-2 | 重复的gradient计算 | environment_gpu.py | EPF/KIF/ISF分别计算,可复用 |
| P2-3 | 文档注释不完整 | most files | 缺少API文档 |

---

## 3. 优化方案

### 3.1 P0-1: 集成机制配置到代码 (预计 1h)

```python
# 目标: 在关键位置读取 Mechanisms / EnvMechanisms

# core/eoe/integrated_simulation.py
from config.agent_mechanisms import Mechanisms, EnvMechanisms, load_config

class IntegratedSimulation:
    def __init__(self, ...):
        load_config("full")  # 或从命令行指定
        
    def step(self):
        # 传感器
        if Mechanisms.SENSOR_EPF:
            sensors = self.agents.get_sensors(self.env)
        
        # 神经网络
        if Mechanisms.ACTUATOR_THRUST:
            brain_outputs = self.agents.forward_brains(sensors)
        
        # 热力学
        if Mechanisms.ENERGY_EXTRACTION:
            thermo_stats = self.thermo.apply(...)
        
        # 场更新
        if EnvMechanisms.DIFFUSION:
            self.env.step()  # 扩散
        
        # 能量守恒检查 (优化: 改为每100步)
        if Mechanisms.ENERGY_EXTRACTION and self.step_count % 100 == 0:
            is_conserved = self.thermo.check_energy_conservation(...)
```

### 3.2 P0-2: 修复 brain_masks 应用 (预计 0.5h)

```python
# core/eoe/batched_agents.py
def forward_brains(self, sensors, brain_weights, brain_masks):
    # 现有代码: 权重未乘掩码
    # 修复:
    W1 = brain_weights[:, :input_dim, :hidden_dim]
    M1 = brain_masks[:, :input_dim, :hidden_dim]
    hidden = torch.bmm(sensors.unsqueeze(1), W1 * M1).squeeze(1)
    hidden = F.relu(hidden)
```

### 3.3 P1-1: 合并传感器读取 (预计 1h)

```python
# 现状: 9次单独GPU调用
# 优化: 批量采样

def get_sensors_batch(self, env):
    # 一次性获取所有场值
    positions = self.state.positions  # [N, 2]
    
    # 批量双线性插值 (单次kernel)
    epf_values = F.grid_sample(
        env.energy_field.field,
        positions.unsqueeze(0).unsqueeze(0),
        mode='bilinear', align_corners=True
    )  # [1, 1, 1, N]
    
    # 类似获取 KIF, ISF
    # 总计: 3次调用 vs 9次
```

### 3.4 P1-4: 早停优化 (预计 0.5h)

```python
# integrated_simulation.py
def run(self, max_steps, verbose):
    for step in range(max_steps):
        self.step(...)
        
        # 现有代码
        if torch.sum(self.alive_mask) == 0:
            break
            
        # 新增: 早停阈值
        alive_ratio = torch.sum(self.alive_mask) / self.n_agents
        if alive_ratio < 0.01 and step > 100:  # 存活<1%时提前结束
            break
```

### 3.5 P2-1: 统一配置管理 (预计 2h)

```yaml
# config/simulation.yaml (新建)
simulation:
  default_preset: "full"
  energy:
    max: 200.0
    min: 0.0
    initial: 100.0
    extraction_rate: 0.5
  metabolic:
    base: 0.001
    motion_factor: 0.01
    signal_factor: 0.01
  evolution:
    elite_ratio: 0.1
    mutation_rate: 0.1
    isf_decay: 0.5
```

---

## 4. 实施计划

### Phase 1: 核心修复 (优先级 P0)
| 任务 | 预计时间 | 依赖 |
|------|----------|------|
| 集成机制配置 | 1h | - |
| 修复 brain_masks | 0.5h | - |
| 能量守恒优化 | 0.5h | - |

### Phase 2: 性能优化 (优先级 P1)
| 任务 | 预计时间 | 依赖 |
|------|----------|------|
| 合并传感器读取 | 1h | Phase 1 |
| 早停优化 | 0.5h | - |
| torch.compile 加速 | 1h | - |

### Phase 3: 代码质量 (优先级 P2)
| 任务 | 预计时间 | 依赖 |
|------|----------|------|
| 统一配置管理 | 2h | Phase 1 |
| 补充文档 | 2h | - |

---

## 5. 预期效果

| 指标 | 当前 | 优化后 | 提升 |
|------|------|--------|------|
| 500 Agent/代 | 4.0s | 2.5s | +37% |
| 1000 Agent/代 | 2.7s | 1.8s | +33% |
| GPU利用率 | 32% | 50% | +18% |
| 代码覆盖率 | 60% | 85% | +25% |

---

## 6. 审批

请确认以下内容:

- [ ] 同意 Phase 1 实施 (核心修复)
- [ ] 同意 Phase 2 实施 (性能优化)
- [ ] 同意 Phase 3 实施 (代码质量)
- [ ] 有其他优先级调整需求?

**开始实施: _______**