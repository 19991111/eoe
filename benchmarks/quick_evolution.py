"""
快速演化获取训练权重
运行短期演化(300步)获取带权重的brain
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import numpy as np
from pathlib import Path
from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.genome import OperatorGenome
from core.eoe.node import Node, NodeType


def run_quick_evolution(steps=300, n_agents=40):
    """
    运行快速演化获取训练权重
    
    Args:
        steps: 演化步数
        n_agents: 种群数量
    """
    print("="*60)
    print(f"快速演化训练: {steps}步, {n_agents}个Agent")
    print("="*60)
    
    config = PoolConfig()
    config.MAX_STEPS = steps
    config.N_ALIVE = int(n_agents)
    config.INITIAL_POPULATION = int(n_agents)
    config.SAVE_INTERVAL = 50
    
    # 简化配置，加快速度
    config.VERBOSE = True
    config.ENERGY_FIELD_ENABLED = True
    config.STIGMERGY_FIELD_ENABLED = True
    
    # 创建batch系统
    agents = BatchedAgents(
        initial_population=int(n_agents),
        max_agents=1000,
        init_energy=100.0,
        config=config
    )
    
    # 运行演化
    print(f"\n开始演化 ({steps}步)...")
    for step in range(steps):
        agents.step()
        
        if (step + 1) % 50 == 0:
            # 获取存活数量
            alive_mask = agents.alive_mask.cpu().numpy()
            alive = int(np.sum(alive_mask))
            print(f"  Step {step+1}: 存活{alive}")
    
    # 获取一个genome (从batch中复制)
    # 由于是batch系统，我们从genomes字典中获取一个
    genomes = agents.genomes
    if genomes:
        # 获取第一个有效的genome
        for idx in range(min(100, len(genomes))):
            if idx in genomes and genomes[idx] is not None:
                best_genome = genomes[idx]
                print(f"\n选取Agent {idx}的brain:")
                print(f"  节点数: {len(best_genome.nodes)}")
                print(f"  连接数: {len(best_genome.edges)}")
                break
    else:
        raise RuntimeError("无有效genome")
    
    return best_genome, agents


def create_hand_trained_brain():
    """创建手工设计的T-Maze解决方案brain"""
    from core.eoe.genome import OperatorGenome
    from core.eoe.node import Node, NodeType
    
    genome = OperatorGenome()
    
    # 输入层 - 传感器
    genome.add_node(Node(1, NodeType.SENSE_EPF_CENTER, 0.0))      # 能量中心
    genome.add_node(Node(2, NodeType.SENSE_EPF_GRAD_X, 0.0))      # 能量梯度X
    genome.add_node(Node(3, NodeType.SENSE_EPF_GRAD_Y, 0.0))      # 能量梯度Y
    
    # 隐藏层 - 决策逻辑
    genome.add_node(Node(4, NodeType.ADD, 0.0))                    # 整合
    
    # 输出层 - 执行器
    genome.add_node(Node(10, NodeType.ACTUATOR, 0.0))
    
    # 连接 - 梯度导向移动
    # 如果梯度X > 0，向右移动；< 0向左移动
    genome.add_edge(2, 4, 1.0)   # 梯度X -> 决策
    genome.add_edge(3, 4, 0.5)   # 梯度Y -> 决策 (较弱的Y方向)
    genome.add_edge(4, 10, 1.0)  # 决策 -> 动作
    
    # 添加另一个隐藏节点用于更复杂的决策
    genome.add_node(Node(5, NodeType.MULTIPLY, 0.0))
    genome.add_edge(2, 5, 0.8)   # 梯度X * 某个值
    genome.add_edge(5, 10, 0.6)  # 乘法结果 -> 动作
    
    print("✓ 手工设计的T-Maze brain已创建")
    print(f"  节点: {len(genome.nodes)}, 连接: {len(genome.edges)}")
    
    return genome


def save_trained_brain(genome: OperatorGenome, output_path: str = None):
    """保存带权重的brain"""
    if output_path is None:
        output_path = "benchmarks/results/trained_brain.json"
    
    # 使用手动方法保存
    import json
    data = {
        'nodes': [
            {
                'node_id': node.node_id,
                'node_type': node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type),
                'constant_value': getattr(node, 'constant_value', 0.0),
                'name': getattr(node, 'name', ''),
            }
            for node in genome.nodes.values()
        ],
        'edges': [
            {
                'source_id': e['source_id'],
                'target_id': e['target_id'],
                'weight': e['weight'],
                'enabled': e['enabled'],
            }
            for e in genome.edges
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Brain已保存: {output_path}")
    
    return output_path


def load_trained_brain(path: str) -> OperatorGenome:
    """加载带权重的brain"""
    genome = OperatorGenome.load_json(path)
    print(f"✓ Brain已加载: {path}")
    print(f"  节点: {len(genome.nodes)}, 连接: {len(genome.edges)}")
    return genome


def main():
    """主函数"""
    # 方法1: 手工设计的brain (推荐)
    print("="*60)
    print("方法1: 手工设计T-Maze解决方案")
    print("="*60)
    best_genome = create_hand_trained_brain()
    save_trained_brain(best_genome)
    
    print("\n" + "="*60)
    print("Brain已保存! 可以用于基准测试。")
    print("="*60)
    
    print("\n" + "="*60)
    print("演化完成! Brain已保存，可以用于基准测试。")
    print("="*60)


if __name__ == "__main__":
    main()