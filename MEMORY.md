# MEMORY - 长期记忆

> EOE项目快速索引 | 最后更新: 2026-03-14

---

## 2026-03-13 重大更新：v14 鲍德温效应 + 演化棘轮

### 实验成功！🎉

| 机制 | 状态 | 效果 |
|------|------|------|
| 鲍德温效应 (Hebbian学习) | ✅ | 45%种群活跃学习 (+3082%) |
| 演化棘轮 (SuperNode) | ✅ | 10个共识模式冻结，30%代谢折扣 |
| 寒武纪初始化 | ✅ | 初始节点3-7随机 |
| 动态环境 | ✅ | 季节+干旱促进适应 |

### 关键发现

- **简单脑在恶劣环境下胜出** - 3节点SENSOR→PROCESS→ACTUATOR最节能
- **学习能力比结构复杂度更重要** - 符合鲍德温效应
- 详见: `memory/2026-03-13.md`

---

## 2026-03-13 早前更新

- **Bug修复**: 神经网络前向传播 (bmm vs matmul, brain_masks 应用)
- **配置统一**: mechanisms.yaml → PhysicsManifest.from_yaml() → IntegratedSimulation
- **清理清单**: docs/LEGACY_CLEANUP.md 列出待清理的旧代码

---

## 项目索引

- **项目目录**: `eoe_mvp/`
- **核心文档**: `eoe_mvp/docs/`
- **运行脚本**: `eoe_mvp/scripts/`
- **大脑库**: `eoe_mvp/champions/`

---

## 快速命令

```bash
cd eoe_mvp

# 运行实验
python scripts/run_stage4_v111_crucible_test.py

# 管理大脑
python -c "from core.eoe.brain_manager import BrainManager; mgr = BrainManager()"
```

---

## 当前版本

- **核心引擎**: v14.1 (修复版)
- **物理法则**: 9个已注册 (4底层物理 + 5演化机制)
- **GPU加速**: 210x (PyTorch + VRAM常驻)
- **实验脚本**: v14 (test_evo_mechanisms.py)

---

## 2026-03-14 修复: 环境能量系统

### Bug修复
| 问题 | 修复 |
|------|------|
| 能量源容量变负数 | 添加容量检查，限制最大注入量 |
| 能量源重生失效 | 修复条件 `<= min_capacity` |
| 季节计算重复 | 统一使用 `get_seasonal_multiplier()` |

### 验证
- 2000步稳定运行
- 种群达500上限
- 季节倍率正常循环 (0.43-1.39)

---

## 规范文档 (docs/)

- **MEMORY.md** - 项目核心记忆
- **VERSION_MANAGEMENT.md** - 版本规范
- **BRAIN_MANAGEMENT.md** - 大脑管理
- **MEMORY_GUIDE.md** - 记忆管理细则
