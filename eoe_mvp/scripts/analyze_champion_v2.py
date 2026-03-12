#!/usr/bin/env python3
"""
冠军大脑结构分析
================
分析严苛环境下冠军大脑的策略特征
"""
import json


def analyze_brain(brain_path):
    """深度分析大脑结构"""
    
    with open(brain_path) as f:
        brain = json.load(f)
    
    print("="*70)
    print("冠军大脑深度分析")
    print("="*70)
    
    nodes = brain['nodes']
    edges = brain['edges']
    n_nodes = len(nodes)
    n_edges = len(edges)
    
    # 节点类型统计
    types = {}
    for n in nodes:
        t = n.get('node_type', 'unknown')
        types[t] = types.get(t, 0) + 1
    
    print(f"\n【基础信息】")
    print(f"  总节点: {n_nodes}")
    print(f"  总边: {n_edges}")
    print(f"  连接密度: {n_edges/n_nodes:.2f}")
    
    # 节点类型
    print(f"\n【节点类型分布】")
    type_order = ['SENSOR', 'ACTUATOR', 'PREDICTOR', 'META_NODE', 'DELAY', 'ADD', 'MULTIPLY', 
                  'PORT_MOTION', 'PORT_OFFENSE', 'PORT_DEFENSE', 'PORT_REPAIR']
    for t in type_order:
        if t in types:
            c = types[t]
            pct = c/n_nodes*100
            bar = "█" * int(pct/3)
            print(f"  {t:15s}: {c:2d} ({pct:5.1f}%) {bar}")
    
    # 连接分析
    print(f"\n【连接分析】")
    
    # 计算入度和出度
    in_degree = {}
    out_degree = {}
    for e in edges:
        src = e.get('source')
        dst = e.get('target')
        if src is not None:
            out_degree[src] = out_degree.get(src, 0) + 1
        if dst is not None:
            in_degree[dst] = in_degree.get(dst, 0) + 1
    
    # 找出高度节点
    all_degrees = {}
    for n in nodes:
        nid = n.get('node_id')
        deg = in_degree.get(nid, 0) + out_degree.get(nid, 0)
        all_degrees[nid] = {'in': in_degree.get(nid, 0), 'out': out_degree.get(nid, 0), 'type': n.get('node_type')}
    
    # 排序
    sorted_nodes = sorted(all_degrees.items(), key=lambda x: -(x[1]['in'] + x[1]['out']))
    
    print(f"\n【核心节点 (Top 10 Hub)】")
    for nid, deg in sorted_nodes[:10]:
        total = deg['in'] + deg['out']
        t = deg['type']
        print(f"  Node {nid:2d}: in={deg['in']:1d}, out={deg['out']:1d}, total={total:2d} ← {t}")
    
    # 脑区分析
    print(f"\n【脑区架构分析】")
    
    # 传感器
    sensors = [n for n in nodes if n.get('node_type') == 'SENSOR']
    print(f"  感知层: {len(sensors)} 节点")
    if sensors:
        for s in sensors:
            sid = s.get('node_id')
            outs = out_degree.get(sid, 0)
            print(f"    - Sensor {sid}: → {outs} 个下游节点")
    
    # 执行器
    actuators = [n for n in nodes if n.get('node_type') == 'ACTUATOR']
    print(f"  执行层: {len(actuators)} 节点")
    if actuators:
        for a in actuators:
            aid = a.get('node_id')
            ins = in_degree.get(aid, 0)
            print(f"    - Actuator {aid}: ← {ins} 个上游节点")
    
    # 预测节点
    predictors = [n for n in nodes if n.get('node_type') == 'PREDICTOR']
    print(f"  预测层: {len(predictors)} 节点")
    
    # 元认知
    metas = [n for n in nodes if n.get('node_type') == 'META_NODE']
    print(f"  元认知层: {len(metas)} 节点 (极简!)")
    
    # 计算节点
    computes = [n for n in nodes if n.get('node_type') in ['ADD', 'MULTIPLY', 'DELAY']]
    print(f"  计算层: {len(computes)} 节点")
    
    # 端口节点
    ports = [n for n in nodes if 'PORT' in n.get('node_type', '')]
    print(f"  行为端口: {len(ports)} 节点")
    
    # 策略推断
    print(f"\n{'='*70}")
    print("【策略推断】")
    print("="*70)
    
    strategies = []
    
    # 1. 元认知水平
    if len(metas) <= 2:
        strategies.append("⚡ 极简元认知: 不依赖自我监控,直接基于环境反应")
    else:
        strategies.append("📊 复杂元认知: 具备自我监控和反思能力")
    
    # 2. 记忆能力
    delays = types.get('DELAY', 0)
    if delays >= 8:
        strategies.append(f"🧠 强时序记忆: {delays}个延迟节点,能记住过去")
    elif delays >= 3:
        strategies.append(f"📝 基础记忆: {delays}个延迟节点")
    
    # 3. 预测能力
    preds = types.get('PREDICTOR', 0)
    if preds >= 4:
        strategies.append(f"🔮 预测驱动: {preds}个预测节点,能预测未来")
    elif preds >= 2:
        strategies.append(f"👁️ 被动感知: {preds}个预测节点")
    
    # 4. 计算复杂度
    compute_ratio = (types.get('ADD', 0) + types.get('MULTIPLY', 0)) / n_nodes
    if compute_ratio > 0.4:
        strategies.append(f"🧮 高计算密集: {compute_ratio*100:.0f}%节点用于计算")
    
    # 5. 行为复杂度
    port_count = len(ports)
    if port_count >= 4:
        strategies.append(f"🎯 多行为模式: {port_count}个行为端口(运动/攻击/防御/修复)")
    
    # 6. 感知丰富度
    sensor_count = len(sensors)
    if sensor_count <= 2:
        strategies.append(f"👓 简约感知: 仅{sensor_count}个传感器,高效利用")
    
    for s in strategies:
        print(f"  {s}")
    
    # 对比
    print(f"\n【对比Stage 3】")
    print(f"  Stage 3: 37节点, 72边, META=17(46%)")
    print(f"  冠军:    {n_nodes}节点, {n_edges}边, META={types.get('META_NODE',0)}({types.get('META_NODE',0)/n_nodes*100:.0f}%)")
    
    meta_change = types.get('META_NODE', 0) - 17
    print(f"  变化: META {meta_change:+d}个")
    
    return types


if __name__ == "__main__":
    brain_path = "champions/stage4_v126_harsh.json"
    types = analyze_brain(brain_path)