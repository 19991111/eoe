# MEMORY - 长期记忆

> EOE项目快速索引 | 最后更新: 2026-03-13

---

## 2026-03-13 更新

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

- **核心引擎**: v13.0 (GPU重构版)
- **物理法则**: 8个已注册
- **GPU加速**: 210x (PyTorch + VRAM常驻)
- **实验脚本**: v13.0 (main_v13_gpu.py)

---

## 规范文档 (docs/)

- **MEMORY.md** - 项目核心记忆
- **VERSION_MANAGEMENT.md** - 版本规范
- **BRAIN_MANAGEMENT.md** - 大脑管理
- **MEMORY_GUIDE.md** - 记忆管理细则
