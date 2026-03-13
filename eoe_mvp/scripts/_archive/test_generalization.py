#!/usr/bin/env python3
"""
Stage 4 脑结构分析
==================
分析大脑结构,评估泛化潜力
"""
import json


def analyze_brain(brain_data):
    """分析大脑结构"""
    nodes = brain_data.get('nodes', [])
    edges = brain_data.get('edges', [])
    
    # 节点类型统计
    node_types = {}
    for n in nodes:
        t = n.get('node_type', 'unknown')
        node_types[t] = node_types.get(t, 0) + 1
    
    return {
        'total_nodes': len(nodes),
        'total_edges': len(edges),
        'density': len(edges) / max(len(nodes), 1),
        'node_types': node_types,
        'meta_ratio': node_types.get('META_NODE', 0) / max(len(nodes), 1),
        'predictor_count': node_types.get('PREDICTOR', 0),
        'sensor_count': node_types.get('SENSOR', 0),
    }


def main():
    print("="*60)
    print("Stage 4 脑结构分析")
    print("="*60)
    
    # 分析多个大脑
    brains = [
        ('champions/stage4_v111_r3.json', 'v111_r3 (125节点)'),
        ('champions/stage4_v111_20gen.json', 'v111_20gen (124节点)'),
        ('champions/stage4_v110_r10.json', 'v110_r10 (??节点)'),
    ]
    
    for path, name in brains:
        try:
            with open(path) as f:
                brain_data = json.load(f)
            
            analysis = analyze_brain(brain_data)
            
            print(f"\n【{name}】")
            print(f"  节点: {analysis['total_nodes']}")
            print(f"  边: {analysis['total_edges']}")
            print(f"  密度: {analysis['density']:.2f}")
            print(f"  META节点: {analysis['node_types'].get('META_NODE', 0)} ({analysis['meta_ratio']:.0%})")
            print(f"  预测节点: {analysis['predictor_count']}")
            print(f"  传感器: {analysis['sensor_count']}")
            
            # 评估
            if analysis['meta_ratio'] > 0.5:
                print(f"  ⚠️ 过拟合风险: META节点过多")
            if analysis['sensor_count'] < 5:
                print(f"  ⚠️ 泛化受限: 传感器不足")
                
        except Exception as e:
            print(f"\n【{name}】")
            print(f"  错误: {e}")
    
    print("\n" + "="*60)
    print("结论")
    print("="*60)
    print("""
这些大脑在训练环境中表现出色(适应度694.7),
但存在以下泛化风险:
    
1. META节点过高 (56%): 过度压缩环境特定概念
2. 传感器较少: 对新环境适应性受限
3. 密集连接: 难以适应拓扑变化

建议: 
- 在更极端的环境下继续演化
- 减少META节点压缩频率
- 增加传感器多样性
""")


if __name__ == "__main__":
    main()