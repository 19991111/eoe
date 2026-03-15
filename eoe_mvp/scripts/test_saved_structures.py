#!/usr/bin/env python3
"""
复杂结构测试器
==============
加载保存的脑结构并测试其能力

测试场景:
1. T迷宫记忆测试
2. 捕猎测试 (追逐移动目标)
3. 觅食测试 (寻找能量源)
4. 导航测试 (空间探索)
5. 避障测试 (躲避危险)

运行: python scripts/test_saved_structures.py [--structures FILE] [--iterations N]
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import torch
import numpy as np
import argparse
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.genome import OperatorGenome
from core.eoe.node import NodeType


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    score: float
    details: str


def load_structures(filepath: str) -> List[Dict]:
    """加载保存的脑结构"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    structures = data.get('structures', [])
    
    # 支持两种格式: list 或 dict
    if isinstance(structures, dict):
        structures = list(structures.values())
    
    return structures


def create_brain_from_structure(struct: Dict, device='cpu') -> torch.Tensor:
    """从结构创建脑矩阵"""
    nodes = struct['nodes']
    edges = struct['edges']
    
    max_nodes = max(len(nodes), 10)
    brain_matrix = torch.zeros(max_nodes, max_nodes, device=device)
    brain_masks = torch.zeros(max_nodes, max_nodes, device=device, dtype=torch.bool)
    
    # 填充边
    for src, tgt, weight in edges:
        if src < max_nodes and tgt < max_nodes:
            brain_matrix[src, tgt] = weight
            brain_masks[src, tgt] = True
    
    return brain_matrix, brain_masks


def run_t_maze_test(brain_matrix, brain_masks, n_trials=50, device='cpu') -> TestResult:
    """T迷宫记忆测试 - 测试短期记忆能力"""
    correct = 0
    
    for trial in range(n_trials):
        # 随机信号方向 (左=1, 右=2)
        signal_dir = np.random.choice([1, 2])
        
        # 信号阶段 (5步)
        sensors = torch.zeros(10, device=device)
        sensors[0] = signal_dir / 2.0  # 归一化信号
        
        # 简单前向传播
        hidden = torch.relu(torch.mm(sensors.unsqueeze(0), brain_matrix[:10, :10]).squeeze(0))
        output = torch.relu(torch.mm(hidden.unsqueeze(0), brain_matrix[:10, :5]).squeeze(0))
        
        # 决策 (输出0=左, 1=右)
        decision = 0 if output[0] < output[1] else 1
        predicted_dir = decision + 1
        
        if predicted_dir == signal_dir:
            correct += 1
    
    accuracy = correct / n_trials
    return TestResult(
        test_name="T迷宫记忆",
        score=accuracy,
        details=f"{correct}/{n_trials} 次正确"
    )


def run_hunting_test(brain_matrix, brain_masks, n_trials=30, device='cpu') -> TestResult:
    """捕猎测试 - 追逐移动目标"""
    total_closeness = 0
    
    for trial in range(n_trials):
        # 初始化位置
        agent_pos = torch.tensor([50.0, 50.0], device=device)
        prey_pos = torch.tensor([30.0, 30.0], device=device)
        
        steps = 0
        max_steps = 50
        
        while steps < max_steps:
            # 传感器: 猎物的相对位置
            rel_pos = prey_pos - agent_pos
            distance = torch.norm(rel_pos)
            
            sensors = torch.zeros(10, device=device)
            sensors[0] = rel_pos[0] / 50.0  # 归一化
            sensors[1] = rel_pos[1] / 50.0
            sensors[2] = distance / 50.0
            
            # 前向传播
            hidden = torch.relu(torch.mm(sensors.unsqueeze(0), brain_matrix[:10, :10]).squeeze(0))
            output = torch.relu(torch.mm(hidden.unsqueeze(0), brain_matrix[:10, :5]).squeeze(0))
            
            # 移动
            move_dir = output[:2]
            if torch.norm(move_dir) > 0:
                move_dir = move_dir / torch.norm(move_dir)
            agent_pos = agent_pos + move_dir * 2.0
            
            # 猎物随机移动
            prey_pos = prey_pos + torch.randn(2, device=device) * 3.0
            
            steps += 1
        
        # 计算最终距离
        final_dist = torch.norm(agent_pos - prey_pos).item()
        total_closeness += (50 - final_dist) / 50  # 归一化
    
    score = total_closeness / n_trials
    return TestResult(
        test_name="捕猎能力",
        score=score,
        details=f"平均接近度: {score*100:.1f}%"
    )


def run_foraging_test(brain_matrix, brain_masks, n_trials=30, device='cpu') -> TestResult:
    """觅食测试 - 寻找能量源"""
    total_found = 0
    
    for trial in range(n_trials):
        # 随机初始化
        agent_pos = torch.tensor([50.0, 50.0], device=device)
        food_pos = torch.tensor([
            np.random.uniform(10, 90),
            np.random.uniform(10, 90)
        ], device=device)
        
        steps = 0
        found = False
        
        while steps < 100 and not found:
            # 传感器: 食物方向
            rel_pos = food_pos - agent_pos
            distance = torch.norm(rel_pos)
            
            sensors = torch.zeros(10, device=device)
            sensors[0] = torch.tanh(rel_pos[0] / 30.0)
            sensors[1] = torch.tanh(rel_pos[1] / 30.0)
            sensors[2] = torch.sigmoid(distance / 50.0)
            
            # 前向传播
            hidden = torch.relu(torch.mm(sensors.unsqueeze(0), brain_matrix[:10, :10]).squeeze(0))
            output = torch.relu(torch.mm(hidden.unsqueeze(0), brain_matrix[:10, :5]).squeeze(0))
            
            # 移动
            move_dir = output[:2]
            if torch.norm(move_dir) > 0:
                move_dir = move_dir / torch.norm(move_dir)
            agent_pos = agent_pos + move_dir * 3.0
            
            # 检测是否找到
            if distance < 3.0:
                found = True
                total_found += 1
            
            steps += 1
    
    score = total_found / n_trials
    return TestResult(
        test_name="觅食能力",
        score=score,
        details=f"{total_found}/{n_trials} 次找到食物"
    )


def run_navigation_test(brain_matrix, brain_masks, n_trials=20, device='cpu') -> TestResult:
    """导航测试 - 探索并覆盖区域"""
    total_coverage = 0
    
    for trial in range(n_trials):
        # 100x100空间, 随机起点
        agent_pos = torch.tensor([50.0, 50.0], device=device)
        
        visited = set()
        grid_size = 10  # 10x10网格
        
        for step in range(50):
            sensors = torch.zeros(10, device=device)
            # 简化的边界传感器
            sensors[0] = agent_pos[0] / 100.0
            sensors[1] = agent_pos[1] / 100.0
            
            # 前向传播
            hidden = torch.relu(torch.mm(sensors.unsqueeze(0), brain_matrix[:10, :10]).squeeze(0))
            output = torch.relu(torch.mm(hidden.unsqueeze(0), brain_matrix[:10, :5]).squeeze(0))
            
            # 移动
            move_dir = output[:2]
            if torch.norm(move_dir) > 0:
                move_dir = move_dir / torch.norm(move_dir)
            agent_pos = agent_pos + move_dir * 5.0
            
            # 边界限制
            agent_pos = torch.clamp(agent_pos, 0, 100)
            
            # 记录访问
            grid_x = int(agent_pos[0].item() // grid_size)
            grid_y = int(agent_pos[1].item() // grid_size)
            visited.add((grid_x, grid_y))
        
        total_coverage += len(visited) / 100.0  # 归一化
    
    score = total_coverage / n_trials
    return TestResult(
        test_name="空间探索",
        score=score,
        details=f"平均覆盖率: {score*100:.1f}%"
    )


def run_oscillation_test(brain_matrix, brain_masks, device='cpu') -> TestResult:
    """振荡测试 - 测试动态稳定性"""
    sensors = torch.zeros(10, device=device)
    
    # 注入稳定输入
    sensors[0] = 0.5
    
    outputs = []
    for _ in range(20):
        hidden = torch.relu(torch.mm(sensors.unsqueeze(0), brain_matrix[:10, :10]).squeeze(0))
        output = torch.relu(torch.mm(hidden.unsqueeze(0), brain_matrix[:10, :5]).squeeze(0))
        outputs.append(output[:2].cpu().numpy())
        sensors[:2] = output[:2]  # 反馈
    
    # 计算输出变化
    outputs = np.array(outputs)
    variance = np.std(outputs, axis=0).mean()
    
    # 适度变化是好的 (表示动态响应)
    score = min(variance * 10, 1.0)
    
    return TestResult(
        test_name="动态响应",
        score=score,
        details=f"输出变化: {variance:.3f}"
    )


def test_structure(struct: Dict, device='cpu') -> List[TestResult]:
    """测试单个结构的各项能力"""
    print(f"\n🧠 测试结构: {struct['structure_id']}")
    print(f"   节点: {struct['nodes']}, 边: {len(struct['edges'])}")
    print(f"   复杂度: {struct['complexity_score']:.2f}")
    
    brain_matrix, brain_masks = create_brain_from_structure(struct, device)
    
    results = []
    
    # T迷宫测试
    result = run_t_maze_test(brain_matrix, brain_masks, device=device)
    results.append(result)
    print(f"   T迷宫: {result.score:.1%} - {result.details}")
    
    # 捕猎测试
    result = run_hunting_test(brain_matrix, brain_masks, device=device)
    results.append(result)
    print(f"   捕猎: {result.score:.1%} - {result.details}")
    
    # 觅食测试
    result = run_foraging_test(brain_matrix, brain_masks, device=device)
    results.append(result)
    print(f"   觅食: {result.score:.1%} - {result.details}")
    
    # 导航测试
    result = run_navigation_test(brain_matrix, brain_masks, device=device)
    results.append(result)
    print(f"   探索: {result.score:.1%} - {result.details}")
    
    # 振荡测试
    result = run_oscillation_test(brain_matrix, brain_masks, device=device)
    results.append(result)
    print(f"   动态: {result.score:.1%} - {result.details}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='测试保存的脑结构')
    parser.add_argument('--structures', default='experiments/v15_cognitive_premium/saved_structures/top_k_structures.json',
                       help='结构文件路径')
    parser.add_argument('--iterations', type=int, default=10, help='测试的结构数量')
    parser.add_argument('--device', default='cpu', help='设备 (cpu/cuda)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("复杂结构能力测试")
    print("=" * 60)
    
    # 加载结构
    structures = load_structures(args.structures)
    print(f"\n📂 加载 {len(structures)} 个结构")
    
    # 按复杂度排序
    structures = sorted(structures, key=lambda s: s['complexity_score'], reverse=True)
    
    # 测试Top N
    test_structures = structures[:args.iterations]
    
    all_results = []
    for struct in test_structures:
        results = test_structure(struct, device=args.device)
        all_results.append((struct, results))
    
    # 汇总报告
    print("\n" + "=" * 60)
    print("📊 能力汇总")
    print("=" * 60)
    
    test_names = ["T迷宫记忆", "捕猎能力", "觅食能力", "空间探索", "动态响应"]
    
    for i, name in enumerate(test_names):
        scores = [rs[i].score for _, rs in all_results]
        avg = np.mean(scores) if scores else 0
        print(f"  {name}: {avg:.1%}")
    
    # 找出每个结构的最强能力
    print("\n🏆 各结构最强能力:")
    for struct, results in all_results:
        best_idx = np.argmax([r.score for r in results])
        best = results[best_idx]
        print(f"  {struct['structure_id']}: {best.test_name} ({best.score:.1%})")
    
    # 保存结果
    output = {
        'structures_tested': args.iterations,
        'summary': {
            name: float(np.mean([rs[i].score for _, rs in all_results]))
            for i, name in enumerate(test_names)
        },
        'details': [
            {
                'structure_id': s['structure_id'],
                'complexity': s['complexity_score'],
                'results': {r.test_name: r.score for r in rs}
            }
            for s, rs in all_results
        ]
    }
    
    output_path = args.structures.replace('.json', '_test_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 结果已保存: {output_path}")


if __name__ == '__main__':
    main()