# EOE 三大风险修复方案

> 生成日期: 2026-03-15 | 优先级: 极高

---

## 🔴 风险1: 能量守恒验证缺失

### 当前状态
- `integrated_simulation.py` 有 `check_energy_conservation()` 方法
- 但检查阈值宽松: 允许能量在 `initial * 0.1` 到 `initial * 2.0` 之间波动
- **未追踪**: 尸体能量、猎物逃跑耗能、源注入总量

### 修复方案

```python
# core/eoe/energy_audit.py (新建)

class EnergyAuditHook:
    """能量审计钩子 - 每N步强制执行能量守恒检查"""
    
    def __init__(self, tolerance: float = 1e-5, audit_interval: int = 10000):
        self.tolerance = tolerance
        self.audit_interval = audit_interval
        self.initial_total = None
        self.history = []
    
    def initialize(self, env, agents):
        """初始化时记录总能量"""
        self.initial_total = self._compute_total(env, agents)
    
    def _compute_total(self, env, agents):
        """计算当前系统总能量"""
        # 1. EPF场能量
        epf = torch.sum(env.energy_field.field).item()
        
        # 2. ISF场能量
        isf = torch.sum(env.stigmergy_field.field).item()
        
        # 3. 存活Agent能量
        alive_energy = torch.sum(agents.state.energies * agents.alive_mask.float()).item()
        
        # 4. 待生成能量源 (预分配)
        pending_sources = 0
        if hasattr(env, 'food_spawner'):
            pending_sources = env.food_spawner.pending_energy
        
        # 5. 猎物能量 (Intelligent Prey)
        prey_energy = 0
        if hasattr(env, 'prey_energy_sources'):
            prey_energy = env.prey_energy_sources.sum().item()
        
        return epf + isf + alive_energy + pending_sources + prey_energy
    
    def audit(self, env, agents, step: int) -> dict:
        """执行能量审计"""
        if step % self.audit_interval != 0:
            return None
        
        current = self._compute_total(env, agents)
        error = abs(current - self.initial_total) / self.initial_total
        
        result = {
            'step': step,
            'initial': self.initial_total,
            'current': current,
            'relative_error': error,
            'is_conserved': error < self.tolerance
        }
        
        self.history.append(result)
        
        if not result['is_conserved']:
            raise EnergyConservationError(
                f"Step {step}: 能量守恒破缺! "
                f"误差: {error:.2e} (阈值: {self.tolerance:.2e})"
            )
        
        return result
```

### 配置建议
```python
# batched_agents.py PoolConfig
ENERGY_AUDIT_ENABLED = True          # 默认启用
ENERGY_AUDIT_INTERVAL = 10000        # 每10000步检查
ENERGY_AUDIT_TOLERANCE = 1e-5        # 相对误差阈值
```

---

## 🔴 风险2: 可微演化哲学冲突

### 当前状态
- `differentiable_brain.py` 已实现
- `DIFFERENTIABLE_BRAIN = False` (默认关闭)
- 但代码存在且可能被意外启用

### 修复方案

**Step 1: 添加明确的风险标识**

```python
# core/eoe/differentiable_brain.py 开头

"""
⚠️ 风险警告: 可微演化模块
====================================
本模块实现基于梯度下降的权重优化，属于"神谕(Oracle)"基准线，
**不应**纳入开放式演化的主实验路径。

启用此模块将违反 EOE 核心原则:
  "无需任何顶层人工设计的驱动"

仅用于:
1. 理论上限测试 (Upper Bound Benchmark)
2. 证明纯演化可逼近该上限的百分比
3. 学术对比研究
"""
```

**Step 2: 强制确认机制**

```python
# core/eoe/batched_agents.py

class PoolConfig:
    DIFFERENTIABLE_BRAIN = False
    
    # 新增: 强制确认开关
    DIFFERENTIABLE_BRAIN_CONFIRM_REQUIRED = True  # 启用前必须确认
    
    @property
    def DIFFERENTIABLE_BRAIN_safe(self) -> bool:
        """安全访问: 包含强制确认逻辑"""
        if self.DIFFERENTIABLE_BRAIN and self.DIFFERENTIABLE_BRAIN_CONFIRM_REQUIRED:
            raise RuntimeError(
                "⚠️ 危险: 可微演化未授权! "
                "如确需启用，需设置 DIFFERENTIABLE_BRAIN_CONFIRM_REQUIRED = False "
                "并明确理解其违反开放式演化原则"
            )
        return self.DIFFERENTIABLE_BRAIN
```

**Step 3: 文档标注**

在 `README.md` 和实验文档中明确标注:
```
> ⚠️ 实验类别: Oracle基准线 (非主实验)
> 目的: 证明纯演化与梯度下降的差距
```

---

## 🔴 风险3: 欺骗性景观 v16.7 参数失衡

### 当前状态
| 参数 | 当前值 | 问题 |
|------|--------|------|
| 可见时间 | 10% (或5%) | ✅ 合理 |
| 可见能量倍率 | 0.1x | ⚠️ 过低 |
| 隐身能量倍率 | 20x | ✅ 合理 |
| ACTIVE_SENSING_THRESHOLD | 0.3 | ⚠️ 静止Agent无法感知 |

### 根因分析

根据充要条件: $\frac{\partial I_{gain}}{\partial C} > \frac{\partial M}{\partial C}$

- **问题1**: 可见能量0.1x → 即使移动也几乎无收益
- **问题2**: 隐身需要高速(0.3阈值)才能感知 → 静止Agent彻底瞎眼
- **问题3**: 代谢成本恒定 → 任何脑结构都是净亏损

### 修复方案 (三选一)

#### 方案A: 缩小隐身窗口 (推荐)
```python
# 将可见时间从 5-10% 提升到 40%

config.RESOURCE_CYCLE_LENGTH = 500
config.RESOURCE_FADE_STEPS = 300  # 300步可见 (60%), 200步隐身 (40%)
# 或
config.flickering_period = 200     # 200步可见
config.flickering_invisible_moves = 300  # 300步隐身
```

#### 方案B: 认知溢价增强
```python
# 保持95%隐身，但大幅提高隐身捕食收益

config.INVISIBLE_REWARD_MULTIPLIER = 100.0  # 100x (原20x)
config.VISIBLE_REWARD_MULTIPLIER = 0.5       # 0.5x (原0.1x)

# 并降低 ACTIVE_SENSING 门槛
config.ACTIVE_SENSING_THRESHOLD = 0.1        # 0.1 (原0.3)
config.ACTIVE_SENSING_MIN_EFFICIENCY = 0.2   # 静止也有20%感知
```

#### 方案C: 限制能量源运动范围 (釜底抽薪)
```python
# 核心思路: 既然静止能"捡漏"，那就限制能量源移动范围

# 在 environment_gpu.py 或 flickering_energy_field 中
class FlickeringEnergyField:
    def __init__(self, ...):
        self.max_displacement_per_cycle = 30.0  # 每周期最多移动30单位
        # 这样静止Agent无法靠"守株待兔"活命
```

### 推荐: 方案A + 方案C 组合

```python
# v16.8 建议参数
config.RESOURCE_CYCLE_LENGTH = 500
config.RESOURCE_FADE_STEPS = 250        # 50%可见，50%隐身 (平衡点)
config.VISIBLE_REWARD_MULTIPLIER = 1.0  # 恢复正常
config.INVISIBLE_REWARD_MULTIPLIER = 5.0 # 隐身5x (适度溢价)

# 限制能量源游走范围
config.ENERGY_SOURCE_MAX_RANGE = 20.0   # 能量源不会离初始位置太远

# 降低感知门槛
config.ACTIVE_SENSING_THRESHOLD = 0.15
config.ACTIVE_SENSING_MIN_EFFICIENCY = 0.15
```

---

## 实施优先级

| # | 任务 | 优先级 | 状态 |
|---|------|--------|------|
| 1 | 能量审计钩子 | 🔴 极高 | ✅ 已完成 (2026-03-15) |
| 2 | 修复欺骗性景观参数 | 🔴 极高 | ⏳ 待处理 |
| 3 | 可微演化风险标注 | 🟡 中 | ⏳ 待处理 |

---

## 验证方法

修复后运行以下测试:

```bash
# 1. 能量守恒测试
python -c "
from core.eoe.energy_audit import EnergyAuditHook
# 运行10000步，确认 error < 1e-5
"

# 2. 欺骗性景观测试
python experiments/v16_deceptive_landscape/deceptive_exp.py
# 确认: 平均速度 > 0.1, 复杂度不降反升

# 3. 可微演化隔离测试
# 确认: DIFFERENTIABLE_BRAIN = True 时触发警告
```