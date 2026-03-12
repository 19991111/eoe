#!/usr/bin/env python3
"""
Champions 目录智能清理脚本
基于多目标优化: 复杂度 + 贮粮 + 存活能力

保存策略:
1. 保留复杂度最高的前N个
2. 保留贮粮最多的前N个  
3. 保留元数据文件
4. 删除所有checkpoint
"""

import os
import json
from pathlib import Path

CHAMPIONS_DIR = Path("/home/node/.openclaw/workspace/eoe_mvp/champions")
MAX_KEEP_COMPLEX = 3   # 保留复杂度最高的3个
MAX_KEEP_STORAGE = 2   # 保留贮粮最多的2个
MAX_KEEP_SURVIVAL = 2  # 保留冬天存活最长的2个
MIN_BRAIN_SIZE = 500   # 最小大脑文件大小

def parse_brain_file(filepath):
    """解析大脑文件，提取关键指标"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # 基础指标
        nodes = len(data.get('nodes', []))
        edges = len(data.get('edges', []))
        complexity = nodes + edges
        
        # 从文件名提取信息
        name = filepath.stem
        
        # 尝试提取适应度
        fitness = 0
        if 'fit' in name:
            parts = name.split('fit')
            if len(parts) > 1:
                fitness = float(parts[1].split('_')[0])
        
        # 提取贮粮数量
        stored = 0
        if 'stored' in name:
            parts = name.split('stored')
            if len(parts) > 1:
                stored = int(parts[1].split('_')[0])
        
        # 如果数据中有stats
        stats = data.get('stats', {})
        stored = stored or stats.get('stored', 0)
        
        return {
            'filepath': filepath,
            'name': name,
            'nodes': nodes,
            'edges': edges,
            'complexity': complexity,
            'fitness': fitness,
            'stored': stored,
            'size': filepath.stat().st_size
        }
    except Exception as e:
        print(f"  ⚠️ 解析失败 {filepath.name}: {e}")
        return None

def parse_meta_file(filepath):
    """解析元数据文件"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return {
            'filepath': filepath,
            'name': filepath.stem,
            'fitness': data.get('fitness', 0),
            'n_nodes': data.get('n_nodes', 0),
            'n_edges': data.get('n_edges', 0),
            'description': data.get('description', '')
        }
    except:
        return None

def cleanup():
    """执行智能清理"""
    print("🧹 Champions 智能清理")
    print("=" * 60)
    
    # 收集所有大脑
    brains = []
    for f in CHAMPIONS_DIR.glob("*.json"):
        # 跳过checkpoint和meta
        if f.name.startswith("checkpoint_"):
            f.unlink()
            print(f"  🗑️ 删除checkpoint: {f.name}")
            continue
        if f.name.endswith("_meta.json"):
            continue
            
        info = parse_brain_file(f)
        if info:
            brains.append(info)
    
    print(f"\n📊 分析了 {len(brains)} 个大脑文件")
    
    # 1. 按复杂度排序
    by_complexity = sorted(brains, key=lambda x: x['complexity'], reverse=True)
    print(f"\n🧠 复杂度排名 (Top 5):")
    for i, b in enumerate(by_complexity[:5]):
        print(f"  {i+1}. {b['name']}: 节点={b['nodes']}, 边={b['edges']}, 复杂度={b['complexity']}")
    
    # 2. 按贮粮排序
    by_stored = sorted(brains, key=lambda x: x['stored'], reverse=True)
    print(f"\n🏠 贮粮排名 (Top 5):")
    for i, b in enumerate(by_stored[:5]):
        print(f"  {i+1}. {b['name']}: 贮粮={b['stored']}")
    
    # 确定保留集合
    keep = set()
    
    # 保留复杂度最高的
    for b in by_complexity[:MAX_KEEP_COMPLEX]:
        keep.add(b['name'])
        print(f"  ✅ 保留(复杂度): {b['name']}")
    
    # 保留贮粮最多的
    for b in by_stored[:MAX_KEEP_STORAGE]:
        keep.add(b['name'])
        print(f"  ✅ 保留(贮粮): {b['name']}")
    
    # 删除其他的
    deleted = 0
    for b in brains:
        if b['name'] not in keep:
            b['filepath'].unlink()
            deleted += 1
            print(f"  🗑️ 删除: {b['name']} (complexity={b['complexity']}, stored={b['stored']})")
    
    # 统计元数据
    metas = list(CHAMPIONS_DIR.glob("*_meta.json"))
    
    print(f"\n✅ 清理完成!")
    print(f"  删除: {deleted} 个大脑")
    print(f"  保留: {len(keep)} 个大脑 + {len(metas)} 个元数据")
    
    # 显示最终保留
    print(f"\n📦 最终保留:")
    for f in sorted(CHAMPIONS_DIR.glob("*.json")):
        if f.name.endswith("_meta.json"):
            print(f"  📄 {f.name}")
        else:
            info = parse_brain_file(f)
            if info:
                print(f"  🧠 {f.name} (节点={info['nodes']}, 边={info['edges']}, 贮粮={info['stored']})")

if __name__ == "__main__":
    cleanup()