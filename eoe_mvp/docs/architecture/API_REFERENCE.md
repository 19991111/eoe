# API 参考文档

## 核心类

### Simulation

统一仿真入口类。

```python
from core.eoe import Simulation

sim = Simulation(
    n_agents: int = 100,       # Agent数量
    env_width: float = 100.0,  # 环境宽度
    env_height: float = 100.0, # 环境高度
    lifespan: int = 1500,      # 生命周期
    device: str = 'cuda:0',    # 计算设备
    energy_field: bool = True, # 启用能量场
    impedance_field: bool = True, # 启用阻抗场
    stigmergy_field: bool = True  # 启用压痕场
)
```

**方法:**

- `step() -> SimState` - 执行单步
- `run(max_steps, verbose) -> List[SimState]` - 运行完整生命周期
- `get_state() -> dict` - 获取当前状态 (用于可视化)

**属性:**

- `history` - 仿真历史记录
- `env` - 环境实例
- `agents` - Agent实例

---

### AgentState

批量Agent状态容器。

```python
from core.eoe import AgentState
from core.eoe.batch import to_cpu

state = agent.state

# 张量属性
state.positions     # [N, 2] 位置
state.velocities    # [N, 2] 速度
state.energies      # [N] 能量
state.thetas        # [N] 朝向
state.permeabilities # [N] 渗透率
state.defenses      # [N] 防御
state.signals       # [N] 信号
state.is_alive      # [N] 存活标志

# 转换为CPU (可视化)
cpu_state = to_cpu(state)
```

---

### Field Classes

物理场基类和实现。

```python
from core.eoe import Field, EnergyField, ImpedanceField, StigmergyField

# 创建场
epf = EnergyField(
    width=100.0,
    height=100.0,
    resolution=1.0,
    device='cuda:0',
    n_sources=3,
    source_strength=50.0,
    decay_rate=0.99
)

# 采样
value = epf.sample(x, y)

# 梯度
grad_x, grad_y = epf.compute_gradient()

# 批量采样
positions = torch.randn(100, 2, device='cuda:0')
values = epf.get_value_tensor(positions)
```

---

### ThermodynamicLaw

热力学定律 - Agent与场的能量交互。

```python
from core.eoe import ThermodynamicLaw

thermo = ThermodynamicLaw(
    device='cuda:0',
    extraction_rate=0.5,    # 能量提取率
    signal_cost_coef=0.01,  # 信号代价
    metabolic_base=0.001,   # 基础代谢
    max_energy=200.0,       # 最大能量
    min_energy=0.0          # 死亡阈值
)

# 初始化
thermo.initialize(env, agents)

# 应用
stats, new_alive_mask = thermo.apply(env, agents, alive_mask)

# 检查稳定性
is_stable, total_energy = thermo.check_stability(env, agents)
```

---

## 配置

### DEFAULT_FIELD_CONFIG

```python
from core.eoe import DEFAULT_FIELD_CONFIG

# 默认场配置
{
    'energy': {
        'n_sources': 3,
        'source_strength': 50.0,
        'decay_rate': 0.99
    },
    'impedance': {
        'noise_scale': 1.0,
        'obstacle_density': 0.15
    },
    'stigmergy': {
        'diffusion_rate': 0.1,
        'decay_rate': 0.98
    }
}
```

---

## 数据类

### SimState

```python
@dataclass
class SimState:
    generation: int      # 代数
    step: int           # 步数
    total_energy: float # 总能量
    alive_count: int    # 存活数
    mean_energy: float  # 平均能量
    max_energy: float   # 最大能量
    min_energy: float   # 最小能量
```

### ThermoStats

```python
@dataclass
class ThermoStats:
    extracted: float        # 提取的能量
    metabolic: float        # 代谢消耗
    signal_deposited: float # 信号注入
    mean_energy: float      # 平均能量
    max_energy: float       # 最大能量
    min_energy: float       # 最小能量
    alive_count: int        # 存活数
```