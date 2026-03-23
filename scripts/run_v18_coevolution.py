#!/usr/bin/env python3
"""
v18 复杂性梯度驱动的环境协同演化 (MVP版)
==========================================
Complexity-Driven Co-Evolution (CDC)

核心思想：将"环境配置"本身当成一个需要被演化的物种
- 外层循环：演化环境参数
- 内层循环：演化智能体大脑  
- 火炬传递机制：传递最优大脑的"结构模板"给下一代

MVP版本：先验证环境演化逻辑，大脑传递后续完善
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import time
import json
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

from configs import PoolConfig
from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.batched_agents import BatchedAgents
from core.eoe.thermodynamic_law import ThermodynamicLaw
from core.eoe.genome import OperatorGenome
from core.eoe.node import Node, NodeType


# ============================================================================
# 环境基因组 (Environment Genome)
# ============================================================================

@dataclass
class EnvironmentGenome:
    """环境基因组 - 可被演化的环境参数"""
    
    # ===== 代谢压力 =====
    base_metabolism: float = 0.008
    neural_cost: float = 0.0008
    activation_cost: float = 0.001
    
    # ===== 能量系统 =====
    food_energy: float = 80.0
    food_count: int = 30
    energy_recirculation_ratio: float = 0.6
    
    # ===== 生态压力 =====
    soft_carrying_cap: bool = True
    global_energy_budget: float = 3000.0
    crowding_penalty_enabled: bool = True
    crowding_radius: float = 8.0
    
    # ===== 捕食者 =====
    predation_enabled: bool = True
    predation_rate: float = 0.8
    predation_range: float = 4.0
    predation_cost: float = 0.05
    
    # ===== 动态环境 =====
    seasons_enabled: bool = True
    season_length: int = 2000
    winter_multiplier: float = 0.1
    summer_multiplier: float = 1.5
    
    # ===== 动态能量源 (v17.3) =====
    use_dynamic_energy: bool = False
    energy_move_interval: int = 10
    energy_jump_prob: float = 0.2
    
    # ===== 变异参数 =====
    mutation_rate: float = 0.5
    
    # ===== 元数据 =====
    generation: int = 0
    universe_id: int = 0
    
    @staticmethod
    def random(high_pressure: bool = False) -> 'EnvironmentGenome':
        """随机初始化一个环境基因组
        
        Args:
            high_pressure: 如果为True，使用高压参数范围
        """
        genome = EnvironmentGenome()
        
        if high_pressure:
            # 回退到验证成功的参数 (350步版本)
            genome.base_metabolism = np.random.uniform(0.006, 0.025)  
            genome.neural_cost = np.random.uniform(0.0003, 0.0025)  
            genome.food_energy = np.random.uniform(35, 80)  
            genome.food_count = np.random.randint(12, 35)  
            genome.global_energy_budget = np.random.uniform(1200, 3500)  
            genome.predation_rate = np.random.uniform(0.4, 1.3)  
            genome.predation_range = np.random.uniform(2, 7)  
            genome.winter_multiplier = np.random.uniform(0.08, 0.25)  
            genome.season_length = np.random.randint(600, 2500)  
        else:
            # 标准参数范围
            genome.base_metabolism = np.random.uniform(0.005, 0.04)
            genome.neural_cost = np.random.uniform(0.0003, 0.004)
            genome.food_energy = np.random.uniform(25, 100)
            genome.food_count = np.random.randint(8, 50)
            genome.global_energy_budget = np.random.uniform(800, 4000)
            genome.predation_rate = np.random.uniform(0.5, 1.8)
            genome.predation_range = np.random.uniform(2, 8)
            genome.season_length = np.random.randint(400, 3000)
            genome.winter_multiplier = np.random.uniform(0.03, 0.25)
        
        # 共同参数
        genome.energy_recirculation_ratio = np.random.uniform(0.3, 0.8)
        genome.crowding_radius = np.random.uniform(3, 15)
        genome.mutation_rate = np.random.uniform(0.1, 0.5)
        
        # 随机开关（高压模式下更倾向启用）
        if high_pressure:
            genome.soft_carrying_cap = random.random() > 0.2
            genome.crowding_penalty_enabled = random.random() > 0.2
            genome.predation_enabled = random.random() > 0.1  # 倾向启用
            genome.seasons_enabled = random.random() > 0.2
        else:
            genome.soft_carrying_cap = random.random() > 0.3
            genome.crowding_penalty_enabled = random.random() > 0.3
            genome.predation_enabled = random.random() > 0.3
            genome.seasons_enabled = random.random() > 0.3
        genome.use_dynamic_energy = random.random() > 0.7
        
        return genome
    
    def mutate(self, sigma: float = 0.15) -> 'EnvironmentGenome':
        """对参数进行高斯变异"""
        new_genome = deepcopy(self)
        
        # 高斯变异数值参数
        def mutate_param(param, min_val=None, max_val=None):
            if random.random() < 0.8:
                new_val = param * (1 + np.random.normal(0, sigma))
                if min_val is not None:
                    new_val = max(min_val, new_val)
                if max_val is not None:
                    new_val = min(max_val, new_val)
                return new_val
            return param
        
        new_genome.base_metabolism = mutate_param(self.base_metabolism, 0.0005, 0.05)
        new_genome.neural_cost = mutate_param(self.neural_cost, 0.00005, 0.005)
        new_genome.food_energy = mutate_param(self.food_energy, 20, 200)
        new_genome.food_count = max(5, int(mutate_param(self.food_count, 3, 150)))
        new_genome.energy_recirculation_ratio = mutate_param(
            self.energy_recirculation_ratio, 0.1, 0.95
        )
        new_genome.global_energy_budget = mutate_param(
            self.global_energy_budget, 500, 15000
        )
        new_genome.crowding_radius = mutate_param(self.crowding_radius, 2, 25)
        new_genome.predation_rate = mutate_param(self.predation_rate, 0.1, 2.5)
        new_genome.predation_range = mutate_param(self.predation_range, 1, 15)
        new_genome.season_length = max(200, int(mutate_param(self.season_length, 100, 8000)))
        new_genome.winter_multiplier = mutate_param(self.winter_multiplier, 0.0, 0.6)
        
        # 小概率切换布尔开关
        if random.random() < 0.1:
            new_genome.soft_carrying_cap = not new_genome.soft_carrying_cap
        if random.random() < 0.1:
            new_genome.crowding_penalty_enabled = not new_genome.crowding_penalty_enabled
        if random.random() < 0.1:
            new_genome.predation_enabled = not new_genome.predation_enabled
        if random.random() < 0.1:
            new_genome.seasons_enabled = not new_genome.seasons_enabled
        if random.random() < 0.05:
            new_genome.use_dynamic_energy = not new_genome.use_dynamic_energy
        
        new_genome.generation = self.generation + 1
        
        return new_genome
    
    def apply_to_config(self, config: type) -> None:
        """将环境参数应用到PoolConfig子类"""
        config.BASE_METABOLISM = self.base_metabolism
        config.NEURAL_COST = self.neural_cost
        config.ACTIVATION_COST = self.activation_cost
        
        # 能量系统
        config.ENERGY_RECIRCULATION_RATIO = self.energy_recirculation_ratio
        
        # 生态压力
        config.SOFT_CARRYING_CAP = self.soft_carrying_cap
        config.GLOBAL_ENERGY_BUDGET = self.global_energy_budget
        config.CROWDING_PENALTY_ENABLED = self.crowding_penalty_enabled
        config.CROWDING_RADIUS = self.crowding_radius
        
        # 捕食者
        config.PREDATION_ENABLED = self.predation_enabled
        config.PREDATION_RATE = self.predation_rate
        config.PREDATION_RANGE = self.predation_range
        config.PREDATION_COST = self.predation_cost
        
        # 动态环境
        config.SEASONS_ENABLED = self.seasons_enabled
        config.SEASON_LENGTH = self.season_length
        config.WINTER_MULTIPLIER = self.winter_multiplier
        config.SUMMER_MULTIPLIER = self.summer_multiplier
        
        # 动态能量源
        config.USE_DYNAMIC_ENERGY_SOURCE = self.use_dynamic_energy
        config.ENERGY_MOVE_INTERVAL = self.energy_move_interval
        config.ENERGY_JUMP_PROB = self.energy_jump_prob
    
    def __str__(self) -> str:
        return (f"G{self.generation}.U{self.universe_id}: "
                f"metab={self.base_metabolism:.4f}, "
                f"neural={self.neural_cost:.5f}, "
                f"food_e={self.food_energy:.1f}, "
                f"food_n={self.food_count}, "
                f"budget={self.global_energy_budget:.0f}, "
                f"pred={self.predation_rate:.2f}, "
                f"season={self.season_length}, "
                f"dyn={self.use_dynamic_energy}")


# ============================================================================
# 复杂度与适应度计算
# ============================================================================

def get_population_complexity(agents: BatchedAgents) -> float:
    """获取种群平均复杂度"""
    if not hasattr(agents, 'alive_mask'):
        return 0.0
    
    active_indices = torch.where(agents.alive_mask)[0]
    if len(active_indices) == 0:
        return 0.0
    
    node_counts = agents.state.node_counts[active_indices].float()
    return node_counts.mean().item()


def get_population_stats(agents: BatchedAgents) -> Dict:
    """获取种群统计信息"""
    alive_indices = torch.where(agents.alive_mask)[0]
    if len(alive_indices) == 0:
        return {
            'mean_nodes': 0.0,
            'max_nodes': 0,
            'survival_rate': 0.0,
            'mean_energy': 0.0,
        }
    
    node_counts = agents.state.node_counts[alive_indices].float()
    energies = agents.state.energies[alive_indices]
    
    return {
        'mean_nodes': node_counts.mean().item(),
        'max_nodes': node_counts.max().item().item() if hasattr(node_counts.max().item(), 'item') else int(node_counts.max().item()),
        'survival_rate': len(alive_indices) / agents.max_agents,
        'mean_energy': energies.mean().item(),
    }


def evaluate_environment_fitness(
    initial_complexity: float,
    final_complexity: float,
    survival_rate: float,
    max_nodes: int,
    energy_efficiency: float = 1.0,
) -> float:
    """
    环境适应度 = 复杂度增量 * 生存率调整因子 * 能量效率约束
    
    修正要点：
    1. 复杂度增量为负 → 环境太恶劣 → 惩罚
    2. 复杂度增量为0 → 环境太安逸，躺平 → 低分
    3. 防止臃肿：节点数上限约束
    4. 能量效率：低效网络惩罚（防止单纯长节点不做事）
    """
    complexity_delta = final_complexity - initial_complexity
    
    # 生存率惩罚因子
    # 太高(>80%)=太容易   太低(<20%)=太难
    if survival_rate > 0.8:
        survival_factor = 0.5  # 太容易
    elif survival_rate < 0.1:
        survival_factor = survival_rate  # 太难，全灭风险
    else:
        survival_factor = 1.0
    
    # 节点数防止臃肿 (12节点以上开始惩罚)
    node_factor = 1.0 if max_nodes < 12 else max(0.2, 1.0 - (max_nodes - 12) * 0.15)
    
    # 能量效率惩罚：效率低于1.0说明入不敷出，高于3.0可能是异常
    # 使用log惩罚：效率越接近1.0越好
    if energy_efficiency < 0.5:
        efficiency_factor = 0.3  # 严重入不敷出
    elif energy_efficiency > 3.0:
        efficiency_factor = 0.8  # 异常高效率，可能有问题
    else:
        efficiency_factor = min(1.0, energy_efficiency)  # 正常范围
    
    # 核心公式
    if complexity_delta <= 0:
        # 负增长：惩罚，但保留一点分数避免完全淘汰
        fitness = complexity_delta * survival_factor * 0.5 * efficiency_factor
    else:
        fitness = complexity_delta * survival_factor * node_factor * efficiency_factor
    
    return fitness


# ============================================================================
# 宇宙运行函数
# ============================================================================

def create_simulation(genome: EnvironmentGenome, n_agents: int, device: str, width: float = 100.0, height: float = 100.0, brain_templates: List = None):
    """创建配置好的模拟环境"""
    
    # 创建配置类
    class EvolvedConfig(PoolConfig):
        pass
    
    # 应用基因组参数
    genome.apply_to_config(EvolvedConfig)
    
    # 设置基础参数
    EvolvedConfig.DEVICE = device
    EvolvedConfig.MAX_AGENTS = n_agents
    
    # 关闭不想要的复杂机制以简化
    EvolvedConfig.T_MAZE_ENABLED = False
    EvolvedConfig.SUPERNODE_ENABLED = False
    EvolvedConfig.HEBBIAN_ENABLED = False
    EvolvedConfig.CAMBRIAN_INIT = True
    EvolvedConfig.METABOLIC_GRACE = True
    
    config = EvolvedConfig()
    
    # 保存brain_templates供后续使用
    saved_templates = brain_templates
    
    # 创建环境
    env = EnvironmentGPU(
        width=width,
        height=height,
        device=device,
        energy_field_enabled=True,
        impedance_field_enabled=False,
        stigmergy_field_enabled=False
    )
    
    # 创建Agent池
    agents = BatchedAgents(
        initial_population=n_agents,
        max_agents=n_agents + 50,  # 预留空间
        env_width=width,
        env_height=height,
        device=device,
        init_energy=config.INITIAL_ENERGY if hasattr(config, 'INITIAL_ENERGY') else 80.0,
        config=config,
        env=env
    )
    
    # 🔥 火炬传递：加载最优大脑模板
    if brain_templates and len(brain_templates) > 0:
        print(f"   🔥 加载 {len(brain_templates)} 个大脑模板...")
        # 使用set_brains方法加载模板
        n_load = min(len(brain_templates), n_agents)
        templates_to_load = []
        for i in range(n_load):
            template = brain_templates[i % len(brain_templates)]
            templates_to_load.append(template.copy())
        
        # 调用set_brains加载模板
        agents.set_brains(templates_to_load)
        
        # 确保这些agent是活的
        agents.alive_mask[:n_agents] = True
        agents._indices_dirty = True
    
    # 热力学系统会在agents.step中自动处理
    thermo = None
    
    return env, agents, config


def run_universe(args: Tuple) -> Dict:
    """
    在一个宇宙中运行演化
    支持火炬传递：brain_templates用于初始化
    """
    universe_id, genome, inner_steps, device, width, height, n_agents, brain_templates = args
    
    # 设置随机种子
    seed = universe_id * 12345 + genome.generation * 1000 + (hash(str(brain_templates)) % 10000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    genome.universe_id = universe_id
    
    try:
        # 创建模拟（带火炬传递）
        env, agents, config = create_simulation(
            genome, n_agents, device, width, height, brain_templates
        )
        
        # 记录初始复杂度
        initial_stats = get_population_stats(agents)
        initial_complexity = initial_stats['mean_nodes']
        
        # 能量追踪（用于效率计算）
        total_energy_gathered = 0.0
        total_energy_consumed = 0.0
        initial_total_energy = agents.state.energies[:n_agents].sum().item()
        
        # 运行内层循环 - 使用内置step方法
        for step in range(inner_steps):
            stats = agents.step(env=env, dt=1.0)
            
            if stats.get('n_alive', 0) == 0:
                break
            
            # 追踪能量变化
            current_total = agents.state.energies[:n_agents].sum().item()
            # 获取本step的净能量变化
            net_change = current_total - (agents.prev_energy_sum if hasattr(agents, 'prev_energy_sum') else initial_total_energy)
            total_energy_gathered += max(0, net_change)  # 获取的能量
            total_energy_consumed += abs(min(0, net_change))  # 消耗的能量
        
        # 计算能量效率
        energy_efficiency = 1.0
        if total_energy_consumed > 0:
            energy_efficiency = total_energy_gathered / (total_energy_consumed + 1e-5)
        energy_efficiency = min(energy_efficiency, 5.0)  # 防止极端值
        
        # 最终状态
        final_stats = get_population_stats(agents)
        final_complexity = final_stats['mean_nodes']
        survival_rate = final_stats['survival_rate']
        
        # 适应度（加入能量效率约束）
        fitness = evaluate_environment_fitness(
            initial_complexity,
            final_complexity,
            survival_rate,
            final_stats['max_nodes'],
            energy_efficiency
        )
        
        # 获取存活的最优结构模板 (用于火炬传递)
        best_template = None
        alive_indices = torch.where(agents.alive_mask)[0]
        if len(alive_indices) > 0:
            energies = agents.state.energies[alive_indices]
            _, top_idx = energies.topk(min(5, len(alive_indices)))
            top_agent_indices = alive_indices[top_idx]
            
            # 提取一个最优genome作为模板
            for idx in top_agent_indices:
                if idx.item() in agents.genomes:
                    best_template = agents.genomes[idx.item()].copy()
                    break
        
        result = {
            'universe_id': universe_id,
            'genome': genome,
            'fitness': fitness,
            'initial_complexity': initial_complexity,
            'final_complexity': final_complexity,
            'complexity_delta': final_complexity - initial_complexity,
            'survival_rate': survival_rate,
            'max_nodes': final_stats['max_nodes'],
            'mean_energy': final_stats['mean_energy'],
            'alive_count': len(alive_indices),
            'best_genome_template': best_template,
            'success': True,
        }
        
        return result
        
    except Exception as e:
        import traceback
        return {
            'universe_id': universe_id,
            'genome': genome,
            'fitness': -999.0,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False,
        }


# ============================================================================
# 环境演化策略 (μ + λ)
# ============================================================================

class EnvironmentEvolutionaryStrategy:
    """环境遗传算法 - μ + λ 策略"""
    
    def __init__(
        self,
        population_size: int = 4,
        elite_count: int = 1,
        mutation_strength: float = 0.15,
    ):
        self.pop_size = population_size
        self.elite_count = elite_count
        self.mut_sigma = mutation_strength
        
        # 初始化随机环境种群
        self.population = [EnvironmentGenome.random(high_pressure=True) for _ in range(population_size)]
        for i, g in enumerate(self.population):
            g.universe_id = i
    
    def select_and_mutate(self, fitnesses: List[float]) -> List[EnvironmentGenome]:
        """选择+变异生成下一代"""
        # 处理None值
        valid_fitnesses = [f if f is not None else -999 for f in fitnesses]
        
        # 按适应度排序
        sorted_indices = sorted(
            range(len(valid_fitnesses)), 
            key=lambda i: valid_fitnesses[i], 
            reverse=True
        )
        
        # 保留精英
        elites = [self.population[i] for i in sorted_indices[:self.elite_count]]
        
        print(f"    🏆 Elite fitness: {valid_fitnesses[sorted_indices[0]]:.4f}")
        
        # 生成变异后代
        offspring = []
        for i in range(self.pop_size - self.elite_count):
            parent = random.choice(elites)
            offspring.append(parent.mutate(sigma=self.mut_sigma))
        
        # 重新分配universe_id
        new_pop = elites + offspring
        for i, g in enumerate(new_pop):
            g.universe_id = i
            g.generation = self.population[0].generation + 1
        
        self.population = new_pop
        return self.population


# ============================================================================
# 主循环
# ============================================================================

def run_coevolution(
    n_universes: int = 4,
    inner_steps: int = 350,  # 扩展规模
    outer_generations: int = 20,
    device: str = 'cuda:0',
    width: float = 100.0,
    height: float = 100.0,
    n_agents: int = 50,
    save_dir: str = "outputs/v18_coevolution",
):
    """主循环：双重演化"""
    
    print("=" * 70)
    print("🚀 v18 复杂性梯度驱动的环境协同演化 (CDC)")
    print("=" * 70)
    print(f"  并行宇宙: {n_universes}")
    print(f"  内层步数: {inner_steps}")
    print(f"  外层代数: {outer_generations}")
    print(f"  每宇宙Agent数: {n_agents}")
    print(f"  设备: {device}")
    print(f"  保存目录: {save_dir}")
    print("=" * 70)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 初始化环境EA
    env_ea = EnvironmentEvolutionaryStrategy(
        population_size=n_universes,
        elite_count=max(1, n_universes // 4),
        mutation_strength=0.15,
    )
    
    # 追踪历史
    history = {
        'generations': [],
        'best_genomes': [],
        'best_fitnesses': [],
        'avg_complexity': [],
        'avg_survival': [],
    }
    
    best_fitness_overall = -999.0
    best_genome_overall = None
    best_genome_template = None  # 火炬传递：最优大脑模板
    
    for gen in range(outer_generations):
        gen_start = time.time()
        print(f"\n{'='*70}")
        print(f"🌍 环境演化代 {gen}/{outer_generations}")
        if best_genome_template:
            print(f"   🔥 火炬传递: {len(best_genome_template)} 个最优大脑模板")
        else:
            print(f"   🔥 火炬传递: 无（初始代）")
        print(f"{'='*70}")
        
        # 准备当前代的宇宙参数（包含火炬传递的brain_templates）
        universe_args = []
        for i, genome in enumerate(env_ea.population):
            genome.generation = gen
            universe_args.append((
                i, genome, inner_steps, device, width, height, n_agents, best_genome_template
            ))
        
        # 串行运行每个宇宙
        results = []
        for args in universe_args:
            print(f"  🚀 Universe {args[0]}: {args[1]}")
            result = run_universe(args)
            results.append(result)
            
            if result['success']:
                print(f"      fitness={result.get('fitness', -999):.4f}, "
                      f"Δcomplexity={result.get('complexity_delta', 0):.2f}, "
                      f"survival={result.get('survival_rate', 0):.2%}, "
                      f"alive={result.get('alive_count', 0)}, "
                      f"max_nodes={result.get('max_nodes', 0)}")
            else:
                print(f"      ❌ Error: {result.get('error', 'unknown')}")
        
        # 提取适应度
        fitnesses = [r.get('fitness', -999) if r.get('success') else -999 for r in results]
        
        # 找出最佳宇宙
        best_idx = np.argmax(fitnesses)
        best_result = results[best_idx]
        
        print(f"\n  🏆 Best Universe: {best_idx}")
        print(f"      Fitness: {best_result.get('fitness', -999):.4f}")
        print(f"      Complexity Δ: {best_result.get('complexity_delta', 0):.2f}")
        print(f"      Survival: {best_result.get('survival_rate', 0):.2%}")
        print(f"      Genome: {best_result['genome']}")
        
        # 更新最佳总体
        if best_result.get('fitness', -999) > best_fitness_overall:
            best_fitness_overall = best_result.get('fitness', -999)
            best_genome_overall = best_result['genome']
        
        # 🔥 火炬传递：从最佳宇宙提取最优大脑模板
        current_best_template = best_result.get('best_genome_template')
        if current_best_template is not None:
            # 包装成列表（即使只有一个）
            if not isinstance(current_best_template, list):
                current_best_template = [current_best_template]
            best_genome_template = current_best_template
            print(f"   🔥 火炬更新: 传递 {len(best_genome_template)} 个大脑模板")
        
        # 记录历史
        history['generations'].append(gen)
        history['best_genomes'].append(str(best_result['genome']))
        history['best_fitnesses'].append(best_result.get('fitness', -999))
        
        avg_complexity = np.mean([r.get('final_complexity', 0) for r in results if r.get('success')])
        avg_survival = np.mean([r.get('survival_rate', 0) for r in results if r.get('success')])
        history['avg_complexity'].append(avg_complexity)
        history['avg_survival'].append(avg_survival)
        
        # 环境演化：淘汰+变异
        print(f"\n  🔄 环境演化...")
        env_ea.population = env_ea.select_and_mutate(fitnesses)
        
        # 保存历史
        with open(f"{save_dir}/history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        gen_time = time.time() - gen_start
        print(f"\n  📊 代 {gen} 总结 (耗时 {gen_time:.1f}s):")
        print(f"      最佳适应度: {best_fitness_overall:.4f}")
        print(f"      平均复杂度: {avg_complexity:.2f}")
        print(f"      平均存活率: {avg_survival:.2%}")
        print(f"      最佳环境: {best_genome_overall}")
    
    print("\n" + "=" * 70)
    print("🎉 演化完成!")
    print(f"  最佳适应度: {best_fitness_overall:.4f}")
    print(f"  最佳环境: {best_genome_overall}")
    print("=" * 70)
    
    # 保存最终结果
    final_result = {
        'best_genome': str(best_genome_overall) if best_genome_overall else None,
        'best_fitness': best_fitness_overall,
        'history': history,
    }
    with open(f"{save_dir}/final_result.json", 'w') as f:
        json.dump(final_result, f, indent=2, default=str)
    
    return best_genome_overall, history


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="v18 复杂性梯度驱动的环境协同演化")
    parser.add_argument("--universes", type=int, default=4, help="并行宇宙数")
    parser.add_argument("--inner-steps", type=int, default=200, help="内层循环步数")
    parser.add_argument("--generations", type=int, default=20, help="外层循环代数")
    parser.add_argument("--n-agents", type=int, default=50, help="每宇宙Agent数")
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备")
    parser.add_argument("--save-dir", type=str, default="outputs/v18_coevolution", help="保存目录")
    
    args = parser.parse_args()
    
    run_coevolution(
        n_universes=args.universes,
        inner_steps=args.inner_steps,
        outer_generations=args.generations,
        n_agents=args.n_agents,
        device=args.device,
        save_dir=args.save_dir,
    )