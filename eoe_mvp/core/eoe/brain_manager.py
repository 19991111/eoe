#!/usr/bin/env python3
"""
EOE Brain Manager - 大脑文件管理系统

功能:
- 标准化大脑保存格式
- 自动索引更新
- 版本管理
- 英雄冢管理
"""
from __future__ import annotations
import json
import os
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict


# 常量定义
CURRENT_VERSION = "1.0"
BASE_DIR = "champions"


@dataclass
class BrainMeta:
    """大脑元数据"""
    stage: int
    experiment: str
    generation: int
    fitness: float
    round: Optional[int] = None
    
    # 机制配置
    energy_decay_k: Optional[float] = None
    port_interference_gamma: Optional[float] = None
    season_jitter: Optional[float] = None
    nest_tax: Optional[float] = None
    crucible_enabled: bool = False
    
    # 大脑统计
    nodes: int = 0
    edges: int = 0
    meta_nodes: int = 0
    predictors: int = 0
    density: float = 0.0
    
    # 演化信息
    parent_brain: Optional[str] = None
    evolution_type: str = "standard"
    
    # 运行统计
    stored: int = 0
    survived: int = 0
    complexity_score: float = 0.0
    
    # 时间戳
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"


class BrainManager:
    """
    大脑文件管理器
    
    提供统一的大脑保存/加载接口,支持:
    - 标准化格式
    - 自动索引
    - 版本管理
    - 英雄冢
    """
    
    def __init__(self, base_dir: str = BASE_DIR):
        self.base_dir = base_dir
        self._ensure_directories()
        self.index = self._load_index()
    
    def _ensure_directories(self) -> None:
        """确保必要目录存在"""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "stage1"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "stage2"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "stage3"), exist_ok=True)
        
        # stage4子目录
        for v in ["v110", "v111"]:
            os.makedirs(os.path.join(self.base_dir, "stage4", v, "runs"), exist_ok=True)
        
        # hall_of_fame
        os.makedirs(os.path.join(self.base_dir, "hall_of_fame", "by_stage"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "hall_of_fame", "by_fitness"), exist_ok=True)
    
    # ============================================================
    # 保存API
    # ============================================================
    
    def save_champion(
        self,
        brain: dict,
        stage: int,
        experiment: str,
        generation: int,
        fitness: float,
        metadata: Dict[str, Any] = None,
        round_num: int = None,
        is_final: bool = False,
        parent_brain: str = None,
    ) -> str:
        """
        保存冠军大脑
        
        参数:
            brain: 神经网络拓扑 (nodes + edges)
            stage: 阶段 (1-4)
            experiment: 实验名 (如 "v111_crucible")
            generation: 演化代数
            fitness: 适应度得分
            metadata: 额外元数据
            round_num: 轮次 (用于迭代实验)
            is_final: 是否为最终冠军
            parent_brain: 父大脑名称
        
        返回: 保存的相对路径
        """
        # 1. 统计大脑信息
        brain_stats = self._analyze_brain(brain)
        
        # 2. 构建元数据
        meta = BrainMeta(
            stage=stage,
            experiment=experiment,
            generation=generation,
            fitness=fitness,
            round=round_num,
            nodes=brain_stats["nodes"],
            edges=brain_stats["edges"],
            meta_nodes=brain_stats["meta_nodes"],
            predictors=brain_stats["predictors"],
            density=brain_stats["density"],
            parent_brain=parent_brain,
            **(metadata or {})
        )
        
        # 3. 构建完整归档
        archive = {
            "version": CURRENT_VERSION,
            "meta": asdict(meta),
            "brain": brain
        }
        
        # 4. 确定保存路径
        if is_final:
            # 最终冠军
            filename = "champion.json"
        elif round_num is not None:
            # 迭代实验
            filename = f"r{round_num:02d}.json"
        else:
            # 普通保存
            filename = f"gen{generation:04d}.json"
        
        # 构建路径
        subdir = f"stage{stage}/{experiment}"
        if round_num is not None:
            subdir += "/runs"
        
        full_subdir = os.path.join(self.base_dir, subdir)
        os.makedirs(full_subdir, exist_ok=True)
        
        rel_path = os.path.join(subdir, filename)
        full_path = os.path.join(self.base_dir, filename)
        
        # 避免覆盖,添加后缀
        if os.path.exists(full_path) and not is_final:
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(os.path.join(full_subdir, f"{base}_{counter}{ext}")):
                counter += 1
            filename = f"{base}_{counter}{ext}"
            rel_path = os.path.join(subdir, filename)
        
        # 5. 写入文件
        with open(os.path.join(self.base_dir, rel_path), 'w') as f:
            json.dump(archive, f, indent=2)
        
        # 6. 更新索引
        self._update_index(archive, rel_path, meta)
        
        return rel_path
    
    def save_hall_of_fame(self, brains: List[dict]) -> None:
        """保存英雄冢"""
        # 按适应度排序
        sorted_brains = sorted(brains, key=lambda x: x.get("meta", {}).get("fitness", 0), reverse=True)
        
        # 保存全部
        hall_path = os.path.join(self.base_dir, "hall_of_fame", "all_time_best.json")
        with open(hall_path, 'w') as f:
            json.dump({
                "version": CURRENT_VERSION,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "count": len(sorted_brains),
                "brains": sorted_brains[:100]  # 最多100个
            }, f, indent=2)
        
        # 按阶段保存最佳
        by_stage = {}
        for b in sorted_brains:
            stage = b.get("meta", {}).get("stage", 0)
            if stage not in by_stage or b.get("meta", {}).get("fitness", 0) > by_stage[stage].get("meta", {}).get("fitness", 0):
                by_stage[stage] = b
        
        for stage, b in by_stage.items():
            stage_path = os.path.join(self.base_dir, "hall_of_fame", "by_stage", f"stage{stage}_best.json")
            with open(stage_path, 'w') as f:
                json.dump(b, f, indent=2)
        
        # Top10
        top10_path = os.path.join(self.base_dir, "hall_of_fame", "by_fitness", "top10.json")
        with open(top10_path, 'w') as f:
            json.dump({
                "version": CURRENT_VERSION,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "brains": sorted_brains[:10]
            }, f, indent=2)
    
    # ============================================================
    # 加载API
    # ============================================================
    
    def load_champion(self, stage: int, experiment: str = None) -> Optional[dict]:
        """加载冠军大脑"""
        if experiment is None:
            # 找最新的experiment
            stage_dir = os.path.join(self.base_dir, f"stage{stage}")
            if not os.path.exists(stage_dir):
                return None
            
            subdirs = [d for d in os.listdir(stage_dir) if os.path.isdir(os.path.join(stage_dir, d))]
            if not subdirs:
                return None
            experiment = sorted(subdirs, reverse=True)[0]
        
        champion_path = os.path.join(self.base_dir, f"stage{stage}", experiment, "champion.json")
        
        if os.path.exists(champion_path):
            with open(champion_path) as f:
                return json.load(f)
        
        return None
    
    def load_latest(self, stage: int = None) -> Optional[dict]:
        """加载最新的冠军"""
        if stage is not None:
            return self.load_champion(stage)
        
        # 找所有阶段最新的
        for st in range(4, 0, -1):
            champ = self.load_champion(st)
            if champ:
                return champ
        
        return None
    
    def load(self, path: str) -> Optional[dict]:
        """加载指定路径的大脑"""
        full_path = os.path.join(self.base_dir, path)
        if os.path.exists(full_path):
            with open(full_path) as f:
                return json.load(f)
        return None
    
    def get_top_brains(self, n: int = 10, stage: int = None) -> List[dict]:
        """获取Top N大脑"""
        brains = []
        
        # 遍历所有大脑文件
        for root, dirs, files in os.walk(self.base_dir):
            if "hall_of_fame" in root:
                continue
            
            for f in files:
                if f.endswith('.json') and f != "_index.json":
                    try:
                        with open(os.path.join(root, f)) as fp:
                            data = json.load(fp)
                            if "brain" in data:
                                # 计算相对路径
                                rel_path = os.path.relpath(os.path.join(root, f), self.base_dir)
                                data["_path"] = rel_path
                                brains.append(data)
                    except:
                        pass
        
        # 过滤
        if stage is not None:
            brains = [b for b in brains if b.get("meta", {}).get("stage") == stage]
        
        # 按适应度排序
        brains.sort(key=lambda x: x.get("meta", {}).get("fitness", 0), reverse=True)
        
        return brains[:n]
    
    # ============================================================
    # 管理API
    # ============================================================
    
    def prune_old_runs(self, keep_per_stage: int = 3) -> int:
        """清理旧实验,只保留最近N个"""
        removed = 0
        
        for stage in range(1, 5):
            stage_dir = os.path.join(self.base_dir, f"stage{stage}")
            if not os.path.exists(stage_dir):
                continue
            
            # 获取所有experiment目录
            exps = [d for d in os.listdir(stage_dir) if os.path.isdir(os.path.join(stage_dir, d))]
            
            if len(exps) <= keep_per_stage:
                continue
            
            # 保留最新的
            exps.sort(reverse=True)
            to_remove = exps[keep_per_stage:]
            
            for exp in to_remove:
                exp_dir = os.path.join(stage_dir, exp)
                shutil.rmtree(exp_dir)
                removed += 1
        
        return removed
    
    def validate_brains(self) -> List[str]:
        """验证大脑文件完整性"""
        errors = []
        
        for root, dirs, files in os.walk(self.base_dir):
            for f in files:
                if not f.endswith('.json'):
                    continue
                
                path = os.path.join(root, f)
                try:
                    with open(path) as fp:
                        data = json.load(fp)
                    
                    # 检查必要字段
                    if "version" not in data:
                        errors.append(f"{path}: 缺少version字段")
                    if "brain" not in data:
                        errors.append(f"{path}: 缺少brain字段")
                    if "meta" not in data:
                        errors.append(f"{path}: 缺少meta字段")
                        
                except json.JSONDecodeError as e:
                    errors.append(f"{path}: JSON解析错误 - {e}")
                except Exception as e:
                    errors.append(f"{path}: {e}")
        
        return errors
    
    def get_stats(self) -> dict:
        """获取大脑库统计信息"""
        brains = self.get_top_brains(n=1000)
        
        stats = {
            "total_brains": len(brains),
            "by_stage": {},
            "by_experiment": {},
            "top_fitness": 0,
            "largest_brain": 0
        }
        
        for b in brains:
            meta = b.get("meta", {})
            
            # 按阶段统计
            stage = meta.get("stage", 0)
            stats["by_stage"][stage] = stats["by_stage"].get(stage, 0) + 1
            
            # 按实验统计
            exp = meta.get("experiment", "unknown")
            stats["by_experiment"][exp] = stats["by_experiment"].get(exp, 0) + 1
            
            # 最高适应度
            fit = meta.get("fitness", 0)
            if fit > stats["top_fitness"]:
                stats["top_fitness"] = fit
            
            # 最大大脑
            nodes = meta.get("nodes", 0)
            if nodes > stats["largest_brain"]:
                stats["largest_brain"] = nodes
        
        return stats
    
    # ============================================================
    # 内部方法
    # ============================================================
    
    def _analyze_brain(self, brain: dict) -> dict:
        """分析大脑结构"""
        nodes = brain.get("nodes", [])
        edges = brain.get("edges", [])
        
        # 节点类型统计
        node_types = {}
        for n in nodes:
            t = n.get("node_type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1
        
        return {
            "nodes": len(nodes),
            "edges": len(edges),
            "meta_nodes": node_types.get("META_NODE", 0),
            "predictors": node_types.get("PREDICTOR", 0),
            "density": len(edges) / max(len(nodes), 1)
        }
    
    def _load_index(self) -> dict:
        """加载索引"""
        idx_path = os.path.join(self.base_dir, "_index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                return json.load(f)
        return {"version": CURRENT_VERSION, "brains": []}
    
    def _update_index(self, archive: dict, rel_path: str, meta: BrainMeta) -> None:
        """更新索引"""
        # 添加到索引
        entry = {
            "path": rel_path,
            "stage": meta.stage,
            "experiment": meta.experiment,
            "fitness": meta.fitness,
            "nodes": meta.nodes,
            "updated_at": meta.created_at
        }
        
        # 查找并更新或添加
        found = False
        for i, e in enumerate(self.index.get("brains", [])):
            if e["path"] == rel_path:
                self.index["brains"][i] = entry
                found = True
                break
        
        if not found:
            self.index["brains"].append(entry)
        
        self.index["updated_at"] = datetime.utcnow().isoformat() + "Z"
        
        # 写回索引
        idx_path = os.path.join(self.base_dir, "_index.json")
        with open(idx_path, 'w') as f:
            json.dump(self.index, f, indent=2)


# ============================================================
# 便捷函数
# ============================================================

def get_manager(base_dir: str = BASE_DIR) -> BrainManager:
    """获取BrainManager实例"""
    return BrainManager(base_dir)


if __name__ == "__main__":
    # 测试
    mgr = BrainManager()
    
    print("="*60)
    print("Brain Manager 测试")
    print("="*60)
    
    # 统计
    stats = mgr.get_stats()
    print(f"\\n统计信息:")
    print(f"  总大脑数: {stats['total_brains']}")
    print(f"  最高适应度: {stats['top_fitness']:.1f}")
    print(f"  最大节点数: {stats['largest_brain']}")
    print(f"  按阶段: {stats['by_stage']}")
    
    # 验证
    errors = mgr.validate_brains()
    print(f"\\n验证结果: {'全部通过' if not errors else f'{len(errors)}个错误'}")
    
    # Top10
    top10 = mgr.get_top_brains(n=5)
    print(f"\\nTop 5:")
    for i, b in enumerate(top10, 1):
        meta = b.get("meta", {})
        print(f"  {i}. [{meta.get('stage')}] {meta.get('experiment')}: fit={meta.get('fitness', 0):.1f}")