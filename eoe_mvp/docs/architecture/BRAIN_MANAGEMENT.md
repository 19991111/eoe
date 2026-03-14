# EOE 大脑管理方案

## 一、当前问题诊断

### 1.1 文件命名混乱
```
现有命名:
- stage1_best_brain.json
- stage3_champion.json
- stage4_v110_r1.json
- stage4_v110_r10.json
- stage4_v110_round1_champ.json  ← 重复冗余
- best_v086_genius.json
- best_v093_brain.json
- best_v093_brain_meta.json  ← 仅有元数据,无大脑
```

### 1.2 缺少实验元数据
当前只保存神经网络的拓扑结构,缺少:
- 适应度得分
- 训练代数
- 使用的机制版本
- 实验参数
- 演化历史

### 1.3 版本混杂
- best_v0xx: 早期版本实验
- stage1/2/3: 阶段性成果
- stage4_v110/v111: 当前版本

### 1.4 存储冗余
- 每轮保存完整大脑
- 未实现增量更新
- 未清理中间结果

---

## 二、推荐的大脑格式

### 2.1 标准大脑文件格式 (BrainArchive)

```json
{
  "version": "1.0",
  "created_at": "2026-03-12T12:00:00Z",
  
  "meta": {
    "stage": 4,
    "experiment": "v11.1_crucible",
    "generation": 20,
    "fitness": 694.7,
    "round": 3,
    
    "mechanisms": {
      "energy_decay_k": 0.00005,
      "port_interference_gamma": 1.5,
      "season_jitter": 0.05,
      "crucible_enabled": true
    },
    
    "brain_stats": {
      "nodes": 124,
      "edges": 288,
      "meta_nodes": 68,
      "predictors": 12,
      "density": 1.96
    },
    
    "origin": {
      "parent_brain": "stage4_v110_r10.json",
      "evolution_type": "crucible_selection"
    }
  },
  
  "brain": {
    "nodes": [...],
    "edges": [...]
  }
}
```

### 2.2 索引文件 (BrainIndex)

```json
{
  "version": "1.0",
  "updated_at": "2026-03-12T12:00:00Z",
  
  "brains": [
    {
      "id": "stage4_v111_final",
      "path": "stage4/v111_final.json",
      "stage": 4,
      "fitness": 694.7,
      "nodes": 124,
      "description": "压力梯度熔炉20代最优"
    },
    {
      "id": "stage3_champion", 
      "path": "stage3/champion.json",
      "stage": 3,
      "fitness": 40008.4,
      "nodes": 17,
      "description": "阶段三冠军"
    }
  ]
}
```

---

## 三、推荐目录结构

```
champions/
├── _index.json              # 全局索引
├── _archive/                # 归档(旧版本)
│   ├── best_v0xx/
│   └── stage1/
├── stage1/
│   ├── champion.json        # 正式冠军
│   └── runs/
│       ├── run_001.json
│       └── run_002.json
├── stage2/
│   ├── champion.json
│   └── runs/
├── stage3/
│   ├── champion.json
│   └── runs/
├── stage4/
│   ├── v110/
│   │   ├── champion.json    # v110最强
│   │   └── runs/
│   │       ├── r01.json
│   │       ├── r02.json
│   │       └── ...
│   └── v111/
│       ├── champion.json    # v111最强 (当前最佳)
│       └── runs/
│           ├── r01.json
│           ├── r02.json
│           └── r03.json
└── hall_of_fame/            # 英雄冢 (历史最佳)
    ├── all_time_best.json
    ├── by_stage/
    │   ├── stage1_best.json
    │   └── stage4_best.json
    └── by_fitness/
        ├── top10.json
        └── top100.json
```

---

## 四、管理API设计

### 4.1 BrainManager 类

```python
class BrainManager:
    """大脑文件管理器"""
    
    def __init__(self, base_dir: str = "champions"):
        self.base_dir = base_dir
        self.index = self._load_index()
    
    # ---- 保存API ----
    
    def save_champion(
        self,
        brain: dict,
        stage: int,
        experiment: str,
        generation: int,
        fitness: float,
        metadata: dict = None
    ) -> str:
        """
        保存冠军大脑
        
        返回: 保存的路径
        """
        # 1. 构建标准格式
        archive = self._build_archive(
            brain, stage, experiment, 
            generation, fitness, metadata
        )
        
        # 2. 确定保存路径
        path = self._get_save_path(stage, experiment, generation)
        
        # 3. 写入文件
        with open(path, 'w') as f:
            json.dump(archive, f, indent=2)
        
        # 4. 更新索引
        self._update_index(archive, path)
        
        return path
    
    def save_hall_of_fame(self, brains: List[dict]) -> None:
        """保存英雄冢"""
        pass
    
    # ---- 加载API ----
    
    def load_champion(self, stage: int, version: str = None) -> dict:
        """加载冠军大脑"""
        pass
    
    def load_latest(self, stage: int = None) -> dict:
        """加载最新的冠军"""
        pass
    
    def get_top_brains(self, n: int = 10, stage: int = None) -> List[dict]:
        """获取Top N大脑"""
        pass
    
    # ---- 管理API ----
    
    def prune_old_runs(self, keep_per_stage: int = 3) -> int:
        """清理旧实验,只保留最近N个"""
        pass
    
    def export_archive(self, output_dir: str) -> None:
        """导出完整归档"""
        pass
    
    def validate_brains(self) -> List[str]:
        """验证大脑文件完整性"""
        pass
```

### 4.2 使用示例

```python
# 保存冠军
manager = BrainManager()

# 方式1: 简洁保存
path = manager.save_champion(
    brain=agent.genome.to_dict(),
    stage=4,
    experiment="v111_crucible",
    generation=20,
    fitness=694.7,
    metadata={
        "stored": 168,
        "survived": 15,
        "complexity": 4.0
    }
)
print(f"已保存到: {path}")

# 加载冠军
champion = manager.load_latest(stage=4)

# 获取Top10
top10 = manager.get_top_brains(n=10)
```

---

## 五、实施计划

### Phase 1: 规范化 (1小时)
- [ ] 创建 BrainManager 类
- [ ] 定义标准格式 BrainArchive
- [ ] 迁移现有文件到新结构

### Phase 2: 索引化 (30分钟)
- [ ] 实现自动索引更新
- [ ] 添加 get_top_brains() 方法
- [ ] 实现 hall_of_fame 管理

### Phase 3: 自动化 (30分钟)
- [ ] 集成到训练脚本
- [ ] 自动清理旧文件
- [ ] 完整性验证

---

## 六、版本演进

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0 | 2026-03-12 | 初始版本 |
| 1.1 | - | 增加增量保存 |
| 1.2 | - | 增加压缩存储 |