# EOE 项目管理规范

> 项目管理核心规范 | 最后更新: 2026-03-13

---

## 一、项目结构

```
eoe_mvp/
├── core/                       # 核心代码
│   ├── core.py                 # 核心入口
│   ├── core_compat.py          # 兼容层
│   └── eoe/                    # EOE 模块
│       ├── __init__.py         # 统一API导出
│       ├── manifest.py         # 物理法则配置 (唯一真值)
│       ├── integrated_simulation.py  # 集成仿真引擎
│       ├── batched_agents.py  # GPU批量Agent系统
│       ├── environment_gpu.py # GPU环境
│       ├── thermodynamic_law.py  # 热力学定律
│       ├── brain_manager.py   # 大脑管理系统
│       ├── agent.py           # 单Agent (CPU兼容)
│       ├── genome.py          # 基因组与演化
│       ├── node.py            # 神经网络节点
│       ├── population.py      # 种群管理
│       ├── environment.py     # CPU环境
│       ├── fields/            # 物理场模块
│       ├── batch/             # 批量GPU系统
│       └── env/               # 环境模块
│
├── scripts/                    # 活跃脚本
├── docs/                       # 文档
├── config/                     # 配置文件
├── memory/                     # 每日记忆
├── champions/                  # 大脑库
├── _archive/                   # 归档 (不推荐使用)
├── _experimental/              # 实验功能
└── main_v13_gpu.py            # 统一入口
```

---

## 二、代码规范

### 2.1 模块分层

| 层级 | 模块 | 职责 |
|------|------|------|
| **入口** | main_v13_gpu.py | 命令行入口 |
| **仿真** | integrated_simulation.py | 统一调度 |
| **环境** | environment_gpu.py | 物理场管理 |
| **Agent** | batched_agents.py | 批量Agent |
| **配置** | manifest.py | 参数唯一真值 |

### 2.2 命名规范

```
文件: snake_case (e.g., brain_manager.py)
类: PascalCase (e.g., IntegratedSimulation)
函数: snake_case (e.g., get_sensors)
常量: UPPER_SNAKE_CASE
```

### 2.3 配置文件

**唯一真值**: `core/eoe/manifest.py`

所有物理参数必须通过 `PhysicsManifest` 获取:
```python
from core.eoe.manifest import PhysicsManifest

manifest = PhysicsManifest.from_yaml("full")
```

### 2.4 机制预设

| 预设 | 用途 |
|------|------|
| full | 完整功能 |
| simple | 基础调试 |
| no_signal | 禁用信号场 |
| infinite_energy | 无限能量模式 |
| no_evolution | 禁用进化 |
| wrap_world | 边界环绕 |

---

## 三、Git 工作流

### 3.1 提交规范

```
<类型>: <简短描述>

[可选: 详细描述]

[可选: 关闭Issue]
```

**类型前缀:**

| 类型 | 说明 |
|------|------|
| feat | 新功能 |
| fix | Bug修复 |
| docs | 文档更新 |
| chore | 杂项/清理 |
| refactor | 重构 |
| perf | 性能优化 |

**示例:**
```bash
git commit -m "fix: 修复神经网络前向传播bug
- 使用 torch.bmm 替代 torch.matmul
- 正确应用 brain_masks 掩码"
```

### 3.2 分支策略

```
main (生产分支)
  │
  ├── develop (开发分支)
  │     │
  │     ├── feature/xxx (功能分支)
  │     ├── fix/xxx (修复分支)
  │     └── docs/xxx (文档分支)
  │
  └── _archive/ (归档)
```

### 3.3 版本管理

**版本号**: MAJOR.MINOR.PATCH

- MAJOR: 架构重大变更
- MINOR: 新机制/新功能
- PATCH: Bug修复

**当前版本**: v0.0 (等待发布命令)

---

## 四、文档规范

### 4.1 文档结构

| 文档 | 用途 | 更新频率 |
|------|------|----------|
| PROJECT_MANAGEMENT.md | 本文件 - 项目管理 | 每次规范变更 |
| VERSION_MANAGEMENT.md | 版本规范 | 每次版本更新 |
| ARCHITECTURE.md | 架构设计 | 架构变更时 |
| MEMORY.md | 项目记忆 | 里程碑变更 |
| docs/ | 详细文档 | 需要时更新 |

### 4.2 文档内容要求

- 不包含具体版本号 (使用 v0.0 或"当前版本")
- 使用相对路径引用
- 保持简洁,避免冗余

### 4.3 记忆管理

```
memory/
├── MEMORY.md              # 长期记忆 (可选)
└── YYYY-MM-DD.md          # 每日工作日志
```

**每日记忆内容:**
- 实验结果
- 代码变更
- 决策记录
- 待办事项

---

## 五、测试规范

### 5.1 测试类型

| 类型 | 位置 | 说明 |
|------|------|------|
| 单元测试 | 各模块 | 函数级测试 |
| 集成测试 | scripts/test_*.py | 模块交互 |
| 性能测试 | scripts/profile_*.py | 性能基准 |

### 5.2 验证清单

新代码提交前:
- [ ] 代码可运行
- [ ] 无语法错误
- [ ] 无旧版本注释 (v0.x - v13.x)
- [ ] 配置系统正常工作

---

## 六、发布规范

### 6.1 发布检查清单

- [ ] 更新版本号 (VERSION_MANAGEMENT.md + __init__.py)
- [ ] 更新核心代码版本常量
- [ ] 运行测试脚本验证
- [ ] 更新 memory/YYYY-MM-DD.md
- [ ] Git push 到远程

### 6.2 版本发布流程

1. 确定版本号 (MAJOR.MINOR.PATCH)
2. 更新 VERSION_MANAGEMENT.md
3. 更新 core/eoe/__init__.py 中的 __version__
4. 提交并标记 Tag
5. 推送到远程

---

## 七、待办事项

- [ ] 定期清理 _archive 目录
- [ ] 保持文档与代码同步
- [ ] 定期更新 memory/

---

## 八、附录

### A. 关键文件路径

| 用途 | 路径 |
|------|------|
| 统一入口 | main_v13_gpu.py |
| 仿真引擎 | core/eoe/integrated_simulation.py |
| 配置文件 | core/eoe/manifest.py |
| 文档目录 | docs/ |
| 记忆目录 | memory/ |

### B. 快速命令

```bash
# 运行仿真
cd eoe_mvp
python main_v13_gpu.py --agents 500 --steps 1500

# 运行测试
python scripts/test_first_light_v13.py

# 演化实验
python scripts/test_neural_evolution_v13.py --generations 10
```

---

*本文件是项目的核心管理规范,任何变更都需要更新本文件。*