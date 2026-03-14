#!/usr/bin/env python3
"""
v14.0 演化实验: 鲍德温效应 + 演化棘轮
=====================================
运行长时间演化，观察:
1. 鲍德温效应: Agent是否学会"记住"死胡同/食物源
2. 演化棘轮: SuperNode是否冻结，复杂度是否提升
"""

import torch
import time
import numpy as np
from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.genome import OperatorGenome
from core.eoe.node import Node, NodeType
from core.eoe.environment_gpu import EnvironmentGPU


def create_initial_genomes(n_agents: int, config=None):
    """
    寒武纪大爆发: 创建多样化的初始基因组
    
    节点数在3-7之间随机分布
    随机混入DELAY和MULTIPLY节点
    随机连线
    """
    if config is None:
        config = type('Config', (), {
            'CAMBRIAN_INIT': True,
            'CAMBRIAN_MIN_NODES': 3,
            'CAMBRIAN_MAX_NODES': 7,
            'CAMBRIAN_DELAY_PROB': 0.3,
            'CAMBRIAN_MULTIPLY_PROB': 0.3,
            'SILENT_MUTATION': True,
            'SILENT_WEIGHT': 0.001,
        })()
    
    # 节点类型池
    node_type_pool = [NodeType.ADD, NodeType.MULTIPLY, NodeType.THRESHOLD, NodeType.DELAY]
    
    genomes = {}
    for i in range(n_agents):
        g = OperatorGenome()
        
        if config.CAMBRIAN_INIT:
            # 随机节点数
            n_nodes = np.random.randint(config.CAMBRIAN_MIN_NODES, config.CAMBRIAN_MAX_NODES + 1)
            
            # 节点类型: 始终一个SENSOR和一个ACTUATOR，中间随机
            node_types = [NodeType.SENSOR]  # 输入
            for _ in range(n_nodes - 2):
                rt = np.random.random()
                if rt < config.CAMBRIAN_DELAY_PROB:
                    node_types.append(NodeType.DELAY)
                elif rt < config.CAMBRIAN_DELAY_PROB + config.CAMBRIAN_MULTIPLY_PROB:
                    node_types.append(NodeType.MULTIPLY)
                else:
                    node_types.append(NodeType.THRESHOLD)
            node_types.append(NodeType.ACTUATOR)  # 输出
            
            # 添加节点
            for j, nt in enumerate(node_types):
                g.add_node(Node(node_id=j, node_type=nt))
            
            # 随机连线 (确保SENSOR连接到某物，ACTUATOR被某物连接)
            # 每个节点尝试连接到一个后继
            for src in range(len(node_types) - 1):
                if np.random.random() < 0.7:  # 70%概率连接
                    # 连接到随机后继
                    tgt = np.random.randint(src + 1, len(node_types))
                    weight = np.random.uniform(-0.5, 0.5)  # 较小初始权重
                    if config.SILENT_MUTATION:
                        weight = config.SILENT_WEIGHT  # 静默突变
                    g.add_edge(src, tgt, weight=weight)
            
            # 确保SENSOR有输出，ACTUATOR有输入
            # (如果没连上，随机补连)
            if not any(e['source_id'] == 0 for e in g.edges):
                tgt = np.random.randint(1, len(node_types))
                g.add_edge(0, tgt, weight=config.SILENT_WEIGHT)
            
            if not any(e['target_id'] == len(node_types)-1 for e in g.edges):
                src = np.random.randint(0, len(node_types)-1)
                g.add_edge(src, len(node_types)-1, weight=config.SILENT_WEIGHT)
        else:
            # 原始简单大脑
            g.add_node(Node(node_id=0, node_type=NodeType.SENSOR))
            g.add_node(Node(node_id=1, node_type=NodeType.ADD))
            g.add_node(Node(node_id=2, node_type=NodeType.THRESHOLD))
            g.add_node(Node(node_id=3, node_type=NodeType.ACTUATOR))
            g.add_edge(0, 1, weight=np.random.uniform(-1, 1))
            g.add_edge(1, 2, weight=np.random.uniform(-1, 1))
            g.add_edge(2, 3, weight=np.random.uniform(-1, 1))
        
        genomes[i] = g
    
    return genomes


def run_evolution_experiment(
    n_steps: int = 50000,
    initial_population: int = 500,
    device: str = 'cuda:0'
):
    """运行演化实验"""
    
    print("="*70)
    print("🧬 v14.0 演化实验: 鲍德温效应 + 演化棘轮")
    print("="*70)
    print(f"设备: {device}")
    print(f"初始人口: {initial_population}")
    print(f"运行步数: {n_steps}")
    print(f"SuperNode上限: 无限制")
    print()
    
    # 配置
    config = PoolConfig()
    config.HEBBIAN_ENABLED = True
    config.HEBBIAN_REWARD_MODULATION = True
    config.SUPERNODE_ENABLED = True
    config.SUPERNODE_DETECTION_FREQUENCY = 500
    config.PREDATION_ENABLED = True
    config.AGE_ENABLED = True
    
    # 季节/干旱参数
    SEASON_LENGTH = 3000  # 季节周期 (步)
    WINTER_MULT = 0.15    # 冬季能量倍率
    SUMMER_MULT = 1.8     # 夏季能量倍率
    DROUGHT_INTENSITY = 0.08  # 干旱期倍率
    
    # 创建环境 (带季节变化)
    env = EnvironmentGPU(
        width=100.0,
        height=100.0,
        resolution=1.0,
        device=device,
        energy_field_enabled=True,
        impedance_field_enabled=False,
        stigmergy_field_enabled=False,
        danger_field_enabled=False,
        seasons_enabled=True,
        season_length=SEASON_LENGTH,
        winter_multiplier=WINTER_MULT,
        summer_multiplier=SUMMER_MULT,
        drought_intensity=DROUGHT_INTENSITY
    )
    
    # 手动注入能量到环境 (模拟食物源)
    # 直接修改field张量
    env.energy_field.field[0, 0, 50, 50] = 200.0  # 中心
    env.energy_field.field[0, 0, 25, 25] = 100.0  # 左上
    env.energy_field.field[0, 0, 75, 25] = 100.0  # 右上
    env.energy_field.field[0, 0, 25, 75] = 100.0  # 左下
    env.energy_field.field[0, 0, 75, 75] = 100.0  # 右下
    
    # 定期补充能量的回调
    energy_refill_step = 100  # 每100步补充一次
    last_refill = 0
    
    print(f"  🍎 注入能量源到环境")
    
    # 创建Agent池
    agents = BatchedAgents(
        initial_population=initial_population,
        max_agents=5000,
        env_width=100.0,
        env_height=100.0,
        device=device,
        init_energy=100.0,
        config=config,
        env=env
    )
    
    # 添加初始基因组
    genomes = create_initial_genomes(initial_population, config)
    for i, g in genomes.items():
        g.energy = 100.0
        agents.genomes[i] = g
        # 同步node_counts到状态张量
        agents.state.node_counts[i] = len(g.nodes)
    
    # 统计历史
    history = {
        'steps': [],
        'n_alive': [],
        'births': [],
        'deaths': [],
        'avg_energy': [],
        'avg_nodes': [],
        'hebbian_active': [],
        'supernodes': [],
        'savings': []
    }
    
    start_time = time.time()
    last_report = 0
    
    # 步进循环
    for step in range(n_steps):
        # 定期补充能量 (模拟食物源重生)
        if step - last_refill >= energy_refill_step:
            env.energy_field.field[0, 0, 50, 50] = 200.0
            env.energy_field.field[0, 0, 25, 25] = 100.0
            env.energy_field.field[0, 0, 75, 25] = 100.0
            env.energy_field.field[0, 0, 25, 75] = 100.0
            env.energy_field.field[0, 0, 75, 75] = 100.0
            last_refill = step
        
        result = agents.step(env=env, dt=0.1)
        
        # 模拟部分Agent找到食物 (随机给予能量奖励)
        # 这会触发Hebbian学习
        if step % 10 == 0:
            batch = agents.get_active_batch()
            if batch.n > 0:
                # 随机选择10%的Agent给予能量奖励
                reward_mask = torch.rand(batch.n, device=device) < 0.1
                reward_indices = batch.indices[reward_mask]
                agents.state.energies[reward_indices] += 15.0  # 能量奖励
        
        # 每500步报告
        if step - last_report >= 500:
            elapsed = time.time() - start_time
            
            batch = agents.get_active_batch()
            n_alive = batch.n
            avg_energy = batch.energies.mean().item() if n_alive > 0 else 0
            
            # 统计节点数
            node_counts = agents.state.node_counts[agents.alive_mask]
            avg_nodes = node_counts.float().mean().item() if n_alive > 0 else 0
            
            # Hebbian状态
            hebbian_active = 0
            if hasattr(agents, '_hebbian_progress') and agents._hebbian_progress is not None:
                hebbian_active = (agents._hebbian_progress.abs() > 0.001).sum().item()
            
            # SuperNode状态
            n_supernodes = 0
            savings = 0
            if hasattr(agents, 'supernode_registry'):
                stats = agents.supernode_registry.get_stats()
                n_supernodes = stats['n_supernodes']
                savings = stats['total_savings']
            
            # 记录
            history['steps'].append(step)
            history['n_alive'].append(n_alive)
            history['births'].append(result['births'])
            history['deaths'].append(result['deaths'])
            history['avg_energy'].append(avg_energy)
            history['avg_nodes'].append(avg_nodes)
            history['hebbian_active'].append(hebbian_active)
            history['supernodes'].append(n_supernodes)
            history['savings'].append(savings)
            
            print(f"Step {step:5d} | 存活: {n_alive:3d} | "
                  f"平均能量: {avg_energy:6.2f} | "
                  f"平均节点: {avg_nodes:.1f} | "
                  f"Hebbian: {hebbian_active:3d} | "
                  f"SuperNode: {n_supernodes:2d} | "
                  f"节省: {savings:.4f} | "
                  f"耗时: {elapsed:.1f}s")
            
            last_report = step
    
    total_time = time.time() - start_time
    
    # 最终报告
    print("\n" + "="*70)
    print("📊 最终统计")
    print("="*70)
    print(f"总运行时间: {total_time:.1f}秒")
    print(f"每秒步数: {n_steps/total_time:.1f}")
    print()
    print(f"存活人数: {history['n_alive'][-1]}")
    print(f"总出生: {sum(history['births'])}")
    print(f"总死亡: {sum(history['deaths'])}")
    print()
    print(f"最终平均能量: {history['avg_energy'][-1]:.2f}")
    print(f"最终平均节点: {history['avg_nodes'][-1]:.1f}")
    print(f"Hebbian活跃: {history['hebbian_active'][-1]}")
    print(f"SuperNode数量: {history['supernodes'][-1]}")
    print(f"代谢节省: {history['savings'][-1]:.4f}")
    
    # 趋势分析
    print("\n" + "="*70)
    print("📈 趋势分析")
    print("="*70)
    
    # 节点数趋势
    if len(history['avg_nodes']) > 10:
        early_avg = np.mean(history['avg_nodes'][:10])
        late_avg = np.mean(history['avg_nodes'][-10:])
        print(f"平均节点: {early_avg:.1f} → {late_avg:.1f} (变化: {late_avg-early_avg:+.1f})")
    
    # Hebbian学习趋势
    if len(history['hebbian_active']) > 10:
        early_heb = np.mean(history['hebbian_active'][:10])
        late_heb = np.mean(history['hebbian_active'][-10:])
        print(f"Hebbian活跃: {early_heb:.0f} → {late_heb:.0f}")
    
    # SuperNode趋势
    if len(history['supernodes']) > 1:
        first_sn = history['supernodes'][0]
        last_sn = history['supernodes'][-1]
        print(f"SuperNode: {first_sn} → {last_sn}")
    
    print("\n" + "="*70)
    print("✅ 演化实验完成")
    print("="*70)
    
    return history, agents


if __name__ == "__main__":
    import sys
    
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    run_evolution_experiment(n_steps=n_steps, device=device)