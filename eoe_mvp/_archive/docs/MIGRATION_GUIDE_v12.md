# EOE v12.x 迁移指南

## 分层策略

根据执行频率和重要性,将机制分为三类:

### 1️⃣ 高频核心法则 (保留在Environment内部)

| 法则 | 原因 | 状态 |
|------|------|------|
| 基础代谢 | 每步执行,计算密集 | ✅ 保留内联 |
| 物理碰撞 | 每步执行 | ✅ 保留内联 |
| 传感器计算 | 每步执行 | ✅ 保留内联 |
| 神经网络前向 | 每步执行 | ✅ 保留内联 |

### 2️⃣ 可选/周期性法则 (迁移到PhysicalLaw)

| 法则 | 类别 | 状态 |
|------|------|------|
| 季节循环 | 周期性 | ✅ 已迁移 |
| 疲劳系统 | 每步但简单 | ✅ 已迁移 |
| 热力庇护所 | 周期性 | ✅ 已迁移 |
| 端口干涉 | 每步但可选 | ✅ 已迁移 |

### 3️⃣ 实验性机制 (通过Registry)

| 法则 | 状态 |
|------|------|
| 突发灾难 | 📋 待实现 |
| 动态地形 | 📋 待实现 |
| 多物种博弈 | 📋 待实现 |

---

## 迁移步骤

### Step 1: 创建PhysicalLaw子类

```python
class MyNewLaw(PhysicalLaw):
    name = "my_new_mechanism"
    
    def __init__(self, manifest):
        super().__init__(manifest)
        # 初始化状态
    
    def reset(self) -> None:
        # 每代重置
        pass
    
    def apply(self, agents: List, world: Dict) -> None:
        # 法则逻辑
        pass
```

### Step 2: 注册到MechanismRegistry

```python
registry.register("my_new", MyNewLaw)
```

### Step 3: 保持向后兼容

```python
class Environment:
    def enable_my_new_system(self, **kwargs):
        # 新方式: 通过manifest
        self.manifest.my_new_enabled = kwargs.get('enabled', True)
        
        # 旧API保持兼容
        self._my_new_params = kwargs
```

---

## 性能优化技巧

### 1. 缓存参数减少属性访问

```python
def apply(self, agents, world):
    # ❌ 每次访问
    if self.manifest.season_length > 30:
        ...
    
    # ✅ 缓存
    season_len = self.manifest.season_length
    if season_len > 30:
        ...
```

### 2. 周期性计算标记

```python
def apply(self, agents, world):
    # 只在必要时更新
    if self._needs_update():
        self._do_expensive_update()
```

### 3. 使用静态绑定

```python
class Environment:
    def __init__(self, manifest):
        # 初始化时绑定活跃法则
        self._seasonal_law = None
        if manifest.seasonal_cycle:
            from .manifest import SeasonalCycleLaw
            self._seasonal_law = SeasonalCycleLaw(manifest)
    
    def step(self):
        # 直接调用,无查找开销
        if self._seasonal_law:
            self._seasonal_law.apply(self.agents, self.world)
```

---

## 迁移状态追踪

### 已完成 ✅ 全部迁移完成！

- [x] SeasonalCycleLaw - 季节循环
- [x] FatigueSystemLaw - 疲劳系统
- [x] ThermalSanctuaryLaw - 热力庇护所
- [x] PortInterferenceLaw - 端口干涉
- [x] RedQueenLaw - 红皇后竞争 (P3)
- [x] MorphologicalComputationLaw - 形态计算 (P2)
- [x] StigmergicFrictionLaw - 压痕系统 (P1)
- [x] OntogeneticPhaseLaw - 发育相变 (P0) ✅ 完成

---

## 版本历史

- **v12.5**: PhysicsManifest + MechanismRegistry 基础架构
- **v12.6**: Environment集成 + 周期性法则迁移
- **v12.7**: 预计完成所有可选法则迁移