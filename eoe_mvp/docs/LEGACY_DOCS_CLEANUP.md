# 旧版本文档清理方案

## 问题: 文档中存在大量旧版本内容

**统计:** ~115处旧版本引用 (v10-v13)

---

## 需清理的文档

### 1. VERSION_MANAGEMENT.md (需重写)

| 问题 | 建议 |
|------|------|
| 详细的v0.9x-v13版本历史 | 归档为 VERSION_HISTORY.md |
| 当前版本显示v13.0 | 更新为 v0.0 |
| 脚本版本对照表过时 | 删除或简化 |

### 2. MEMORY.md (需重写)

| 问题 | 建议 |
|------|------|
| 详细版本历史表格 | 归档 |
| 架构图标注v13.0 | 更新为当前版本 |
| 硬件/性能数据可能过时 | 审查更新 |
| 待办事项含v13.0 | 更新为当前 |

### 3. BRAIN_MANAGEMENT.md

| 问题 | 建议 |
|------|------|
| v110/v111 大脑引用 | 更新路径 |
| 实验名称含版本号 | 统一命名规范 |

### 4. ARCHITECTURE.md

| 问题 | 建议 |
|------|------|
| v13.0 引用 | 替换为 v0.0 |
| 架构图可能过时 | 审查更新 |

### 5. 归档类文档 (可删除)

| 文件 | 说明 |
|------|------|
| `_archive/docs/EOE_PROJECT_STATE.md` | 旧项目状态 |
| `_archive/docs/MECHANISM_ARCHITECTURE.md` | 旧架构 |
| `_archive/docs/MIGRATION_GUIDE_v12.md` | v12迁移指南 |
| `_archive/docs/PROJECT_TRACKER.md` | 旧追踪器 |
| `energy_field_v13_proposal.md` | v13提案,已过时 |

---

## 方案

### 方案 A: 保守清理 (推荐)

```
1. VERSION_MANAGEMENT.md:
   - 保留格式,版本历史仅保留最近2个
   - 当前版本改为 v0.0

2. MEMORY.md:
   - 删除详细版本表格
   - 更新 v13.0 → v0.0
   - 保留核心架构说明

3. BRAIN_MANAGEMENT.md:
   - 更新大脑路径引用

4. 删除:
   - energy_field_v13_proposal.md (已过时的提案)
   - _archive/docs/ (4个旧文档)
```

### 方案 B: 全面重构

```
- 完全重写 VERSION_MANAGEMENT.md
- 完全重写 MEMORY.md
- 删除所有 _archive/docs/
- 合并重复文档
```

---

## 审批

请选择:

- [ ] **方案 A**: 保守清理
- [ ] **方案 B**: 全面重构

或指定其他清理项:

_________________________________