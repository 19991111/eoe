# 旧版本代码清理清单
## 审查日期: 2026-03-13

---

## 1. 归档目录 (已移至 _archive)

| 目录 | 内容 | 建议 |
|------|------|------|
| `_archive/` | config, docs, dryrun | 可删除 |
| `scripts/_archive/` | 旧实验脚本 (v11-v12) | 保留参考,可删除 |
| `core/eoe/_archive/` | environment_v11.py (142KB) | **可删除** |
| `champions/_archive/` | v086-v097 冠军大脑JSON | 保留历史,可删除 |

---

## 2. 旧版本文件 (可能被新代码替代)

| 文件 | 大小 | 状态 | 说明 |
|------|------|------|------|
| `core/eoe/environment.py` | 172KB | ⚠️ 需审查 | CPU版本,新代码用 environment_gpu.py |
| `core/eoe/agent.py` | 23KB | ⚠️ 需审查 | 单Agent版本,新代码用 batched_agents.py |
| `core/eoe/node.py` | 28KB | ⚠️ 需审查 | 旧节点系统 |
| `core/eoe/genome.py` | 46KB | ⚠️ 需审查 | 旧基因组系统 |
| `core/eoe/population.py` | 92KB | ⚠️ 需审查 | 旧种群系统 |
| `core/eoe/thermodynamic_law.py` | 14KB | ⚠️ 需审查 | 旧热力学,现集成到 integrated_simulation.py |
| `core/eoe/stigmergy_field.py` | 10KB | ⚠️ 需审查 | 旧场实现 |
| `core/eoe/stress_field.py` | 9KB | ⚠️ 需审查 | 旧压力场 |

---

## 3. 重复实现 (fields/ vs 直接实现)

| 新位置 | 旧位置 | 状态 |
|--------|--------|------|
| `environment_gpu.py` (含 EPF/KIF/ISF) | `fields/energy.py` | fields/ 未被使用? |
| `batched_agents.py` | `batch/state.py`, `batch/simulation.py` | batch/ 未被使用? |

---

## 4. 旧版本注释 (v0.70-v12.6)

统计: ~50+ 处旧版本注释

```
core/eoe/genome.py:
  - v0.70: DELAY节点
  - v0.74: ReLU激活
  - v0.78-0.99: 各种特性
  - v0.98: 静默突变

core/eoe/population.py:
  - v0.74-v0.99: 旧特性
  - v10.4-v12.6: 版本演进
  - v11.0: 三大突破机制
  - v11.1: 压力梯度熔炉

core/eoe/kinetic_impedance.py:
  - 需检查
```

---

## 5. 建议清理方案

### 方案 A: 保守清理 (仅删除明确废弃的)

```
删除:
- core/eoe/_archive/environment_v11.py
- config/agent_mechanisms.py (已整合到 manifest.py)
```

### 方案 B: 中等清理

```
删除:
- core/eoe/_archive/
- config/agent_mechanisms.py
- fields/ (未使用)
- batch/ (未使用)
- 旧注释清理 (v0.70-v12.6)
```

### 方案 C: 激进清理 (彻底重构)

```
删除:
- environment.py (CPU版本)
- agent.py (单Agent)
- node.py, genome.py, population.py
- 所有 _archive 目录
- 保留核心: batched_agents.py, environment_gpu.py, integrated_simulation.py, manifest.py
```

---

## 6. 审批

请选择清理方案:

- [ ] **方案 A**: 仅删除明确废弃的
- [ ] **方案 B**: 删除未使用的代码
- [ ] **方案 C**: 彻底重构 (需更多测试)

或指定其他清理项:

_________________________________