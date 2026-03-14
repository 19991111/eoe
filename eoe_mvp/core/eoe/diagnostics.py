"""
v14.1 演化诊断监控系统
========================
用于排查:
1. 鲍德温效应反噬 (学习成本过高)
2. 演化棘轮锁死 (陷入局部最优)  
3. 共生场应力反馈 (链式崩溃)
4. 基因组坍缩

关键指标:
- learning_energy_cost: 学习消耗的能量
- energy_acquired: 获取的能量
- mutation_rejection_rate: 突变拒绝率
- genetic_diversity: 基因多样性
- stress_field_intensity: 应力场强度
- genome_length_dist: 基因组长度分布
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import deque
import json
import os


class EvolutionDiagnostics:
    """演化诊断器 - 实时监控关键指标"""
    
    def __init__(
        self,
        max_agents: int = 10000,
        device: str = 'cpu',
        log_interval: int = 100,
        history_size: int = 1000
    ):
        self.max_agents = max_agents
        self.device = device
        self.log_interval = log_interval
        self.history_size = history_size
        
        # 初始化历史记录
        self.step_history = deque(maxlen=history_size)
        self.genome_history = deque(maxlen=history_size)
        self.mutation_history = deque(maxlen=history_size)
        self.energy_history = deque(maxlen=history_size)
        self.stress_history = deque(maxlen=history_size)
        self.supernode_history = deque(maxlen=history_size)
        
        # 累计统计
        self.total_mutations = 0
        self.rejected_mutations = 0
        self.total_learning_cost = 0.0
        self.total_energy_acquired = 0.0
        
        # 记录开关
        self.enabled = True
        self._step_count = 0
        
    def record_step(
        self,
        step: int,
        n_alive: int,
        energies: torch.Tensor,
        node_counts: torch.Tensor,
        hebbian_active: int,
        hebbian_cost: float,
        energy_gained: float,
        n_mutations: int = 0,
        n_rejections: int = 0,
        n_supernodes: int = 0,
        stress_values: Optional[torch.Tensor] = None,
        genome_lengths: Optional[torch.Tensor] = None
    ):
        """记录单步诊断数据"""
        if not self.enabled:
            return
            
        self._step_count += 1
        
        # 1. 能量统计
        energy_stats = {
            'step': step,
            'n_alive': n_alive,
            'mean_energy': energies.mean().item() if n_alive > 0 else 0,
            'max_energy': energies.max().item() if n_alive > 0 else 0,
            'min_energy': energies.min().item() if n_alive > 0 else 0,
            'std_energy': energies.std().item() if n_alive > 0 else 0,
            'hebbian_active': hebbian_active,
            'hebbian_cost': hebbian_cost,
            'energy_gained': energy_gained,
            'learning_cost_ratio': hebbian_cost / max(energy_gained, 0.1) if energy_gained > 0 else 0
        }
        self.energy_history.append(energy_stats)
        
        # 累计
        self.total_learning_cost += hebbian_cost
        self.total_energy_acquired += energy_gained
        
        # 2. 基因组统计
        if genome_lengths is not None and n_alive > 0:
            genome_stats = {
                'step': step,
                'mean_nodes': node_counts.float().mean().item(),
                'max_nodes': node_counts.max().item(),
                'min_nodes': node_counts.min().item(),
                'std_nodes': node_counts.float().std().item(),
                'genome_length_dist': self._compute_length_dist(genome_lengths)
            }
            self.genome_history.append(genome_stats)
        
        # 3. 突变统计
        self.total_mutations += n_mutations
        self.rejected_mutations += n_rejections
        rejection_rate = n_rejections / max(n_mutations, 1)
        
        mutation_stats = {
            'step': step,
            'n_mutations': n_mutations,
            'n_rejections': n_rejections,
            'rejection_rate': rejection_rate,
            'cumulative_rejection_rate': self.rejected_mutations / max(self.total_mutations, 1)
        }
        self.mutation_history.append(mutation_stats)
        
        # 4. 应力场统计
        if stress_values is not None:
            stress_stats = {
                'step': step,
                'mean_stress': stress_values.mean().item(),
                'max_stress': stress_values.max().item(),
                'min_stress': stress_values.min().item(),
                'std_stress': stress_values.std().item()
            }
            self.stress_history.append(stress_stats)
        
        # 5. SuperNode统计
        supernode_stats = {
            'step': step,
            'n_supernodes': n_supernodes,
            'total_savings': 0  # 后续填充
        }
        self.supernode_history.append(supernode_stats)
        
        # 定期打印警告
        if step % self.log_interval == 0:
            self._check_warnings(step, energy_stats, mutation_stats)
    
    def _compute_length_dist(self, lengths: torch.Tensor) -> Dict[str, int]:
        """计算长度分布"""
        arr = lengths.cpu().numpy()
        return {
            '3_nodes': int((arr == 3).sum()),
            '4_nodes': int((arr == 4).sum()),
            '5_nodes': int((arr == 5).sum()),
            '6_nodes': int((arr == 6).sum()),
            '7+_nodes': int((arr >= 7).sum())
        }
    
    def _check_warnings(self, step: int, energy_stats: Dict, mutation_stats: Dict):
        """检查并打印警告"""
        warnings = []
        
        # 1. 学习成本过高
        lr = energy_stats.get('learning_cost_ratio', 0)
        if lr > 0.5:  # 学习成本超过50%获取能量
            warnings.append(f"⚠️ 学习成本过高: {lr:.1%}")
            
        # 2. 突变拒绝率过高
        rr = mutation_stats.get('cumulative_rejection_rate', 0)
        if rr > 0.8:  # 80%突变被拒绝
            warnings.append(f"⚠️ 演化棘轮锁死: 拒绝率 {rr:.1%}")
            
        # 3. 能量过低
        if energy_stats['mean_energy'] < 10:
            warnings.append(f"⚠️ 能量危机: 平均 {energy_stats['mean_energy']:.1f}")
            
        # 4. 基因组异常 (在genome_history中检查)
        if len(self.genome_history) > 10:
            recent = self.genome_history[-1]
            if recent['mean_nodes'] < 3.5:
                warnings.append(f"⚠️ 基因组坍缩: 平均 {recent['mean_nodes']:.1f} 节点")
        
        if warnings:
            print(f"\n🔍 [Step {step}] 诊断警告:")
            for w in warnings:
                print(f"   {w}")
    
    def get_report(self) -> Dict:
        """生成诊断报告"""
        report = {
            'summary': {
                'total_steps': self._step_count,
                'total_mutations': self.total_mutations,
                'total_rejections': self.rejected_mutations,
                'final_rejection_rate': self.rejected_mutations / max(self.total_mutations, 1),
                'total_learning_cost': self.total_learning_cost,
                'total_energy_acquired': self.total_energy_acquired,
                'overall_learning_ratio': self.total_learning_cost / max(self.total_energy_acquired, 0.1)
            },
            'trends': {}
        }
        
        # 计算趋势
        if len(self.energy_history) >= 10:
            early = list(self.energy_history)[:5]
            late = list(self.energy_history)[-5:]
            
            report['trends']['energy'] = {
                'early_mean': np.mean([e['mean_energy'] for e in early]),
                'late_mean': np.mean([e['mean_energy'] for e in late]),
                'trend': 'increasing' if late[-1]['mean_energy'] > early[0]['mean_energy'] else 'decreasing'
            }
            
            report['trends']['hebbian'] = {
                'early_active': np.mean([e['hebbian_active'] for e in early]),
                'late_active': np.mean([e['hebbian_active'] for e in late])
            }
            
        if len(self.genome_history) >= 10:
            early = list(self.genome_history)[:5]
            late = list(self.genome_history)[-5:]
            
            report['trends']['genome'] = {
                'early_nodes': np.mean([g['mean_nodes'] for g in early]),
                'late_nodes': np.mean([g['mean_nodes'] for g in late])
            }
            
        if len(self.mutation_history) >= 10:
            early = list(self.mutation_history)[:5]
            late = list(self.mutation_history)[-5:]
            
            report['trends']['mutation'] = {
                'early_rejection': np.mean([m['rejection_rate'] for m in early]),
                'late_rejection': np.mean([m['rejection_rate'] for m in late])
            }
            
        return report
    
    def print_report(self):
        """打印诊断报告"""
        report = self.get_report()
        
        print("\n" + "="*60)
        print("📊 演化诊断报告")
        print("="*60)
        
        s = report['summary']
        print(f"\n总步数: {s['total_steps']}")
        print(f"总突变: {s['total_mutations']}, 拒绝: {s['total_rejections']}")
        print(f"总学习成本: {s['total_learning_cost']:.2f}")
        print(f"总能量获取: {s['total_energy_acquired']:.2f}")
        print(f"学习成本比: {s['overall_learning_ratio']:.1%}")
        
        if 'energy' in report['trends']:
            t = report['trends']['energy']
            print(f"\n📈 能量趋势: {t['trend']}")
            print(f"   早期平均: {t['early_mean']:.1f}")
            print(f"   晚期平均: {t['late_mean']:.1f}")
            
        if 'hebbian' in report['trends']:
            t = report['trends']['hebbian']
            print(f"\n🧠 Hebbian学习:")
            print(f"   早期活跃: {t['early_active']:.0f}")
            print(f"   晚期活跃: {t['late_active']:.0f}")
            
        if 'genome' in report['trends']:
            t = report['trends']['genome']
            print(f"\n🧬 基因组:")
            print(f"   早期节点: {t['early_nodes']:.1f}")
            print(f"   晚期节点: {t['late_nodes']:.1f}")
            
        if 'mutation' in report['trends']:
            t = report['trends']['mutation']
            print(f"\n🔄 突变:")
            print(f"   早期拒绝率: {t['early_rejection']:.1%}")
            print(f"   晚期拒绝率: {t['late_rejection']:.1%}")
        
        print("\n" + "="*60)
    
    def save_to_file(self, filepath: str):
        """保存诊断数据到文件"""
        data = {
            'energy_history': list(self.energy_history),
            'genome_history': list(self.genome_history),
            'mutation_history': list(self.mutation_history),
            'stress_history': list(self.stress_history),
            'summary': self.get_report()['summary']
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 诊断数据已保存到: {filepath}")


class DiagnosticRunner:
    """诊断运行器 - 带诊断的实验运行"""
    
    def __init__(
        self,
        n_steps: int = 20000,
        initial_population: int = 500,
        device: str = 'cuda:0',
        log_interval: int = 500,
        save_path: Optional[str] = None
    ):
        self.n_steps = n_steps
        self.initial_population = initial_population
        self.device = device
        self.log_interval = log_interval
        self.save_path = save_path
        
    def run(self):
        """运行带诊断的实验"""
        import torch
        from core.eoe.batched_agents import BatchedAgents, PoolConfig
        from core.eoe.environment_gpu import EnvironmentGPU
        from core.eoe.node import NodeType, Node
        from core.eoe.genome import OperatorGenome
        import numpy as np
        
        print("="*60)
        print("🔬 带诊断的演化实验")
        print("="*60)
        
        # 初始化诊断器
        diagnostics = EvolutionDiagnostics(
            max_agents=10000,
            device=self.device,
            log_interval=self.log_interval
        )
        
        # 创建环境
        env = EnvironmentGPU(
            width=100.0, height=100.0, resolution=1.0,
            device=self.device,
            energy_field_enabled=True,
            seasons_enabled=True,
            season_length=3000,
            winter_multiplier=0.15,
            summer_multiplier=1.8
        )
        
        # 注入能量
        for (x, y), v in [((50, 50), 200.0), ((25, 25), 100.0), ((75, 75), 100.0)]:
            env.energy_field.field[0, 0, y, x] = v
        
        # 配置
        config = PoolConfig()
        config.HEBBIAN_ENABLED = True
        config.HEBBIAN_REWARD_MODULATION = True
        config.SUPERNODE_ENABLED = True
        config.SUPERNODE_DETECTION_FREQUENCY = 500
        config.AGE_ENABLED = True
        config.PREDATION_ENABLED = True
        
        # 创建Agent池
        agents = BatchedAgents(
            initial_population=self.initial_population,
            max_agents=10000,
            env_width=100.0, env_height=100.0,
            device=self.device,
            init_energy=100.0,
            config=config,
            env=env
        )
        
        # 寒武纪初始化
        for i in range(self.initial_population):
            g = OperatorGenome()
            n_nodes = np.random.randint(3, 8)
            types = [NodeType.SENSOR]
            for _ in range(n_nodes - 2):
                types.append(np.random.choice([
                    NodeType.ADD, NodeType.MULTIPLY,
                    NodeType.THRESHOLD, NodeType.DELAY
                ]))
            types.append(NodeType.ACTUATOR)
            
            for j, t in enumerate(types):
                g.add_node(Node(j, t))
            for src in range(len(types) - 1):
                if np.random.random() < 0.7:
                    g.add_edge(src, src+1, 0.001)
            g.energy = 100.0
            
            agents.genomes[i] = g
            agents.state.node_counts[i] = len(g.nodes)
        
        print(f"已初始化 {self.initial_population} 个Agent")
        print(f"运行 {self.n_steps} 步...")
        
        last_refill = 0
        energy_refill_step = 100
        
        for step in range(self.n_steps):
            # 能量补充
            if step - last_refill >= energy_refill_step:
                env.energy_field.field[0, 0, 50, 50] = 200.0
                env.energy_field.field[0, 0, 25, 25] = 100.0
                env.energy_field.field[0, 0, 75, 75] = 100.0
                last_refill = step
            
            # Step
            result = agents.step(env=env, dt=0.1)
            
            # 能量奖励 (触发Hebbian学习)
            if step % 10 == 0:
                batch = agents.get_active_batch()
                if batch.n > 0:
                    reward_mask = torch.rand(batch.n, device=self.device) < 0.1
                    reward_indices = batch.indices[reward_mask]
                    agents.state.energies[reward_indices] += 15.0
            
            # 记录诊断
            if step % self.log_interval == 0:
                batch = agents.get_active_batch()
                
                # Hebbian统计
                hebbian_active = 0
                hebbian_cost = 0.0
                if hasattr(agents, '_hebbian_progress') and agents._hebbian_progress is not None:
                    hebbian_active = (agents._hebbian_progress.abs() > 0.01).sum().item()
                    # 估算学习成本
                    hebbian_cost = hebbian_active * 0.01
                
                # 能量获取
                energy_gained = result.get('energy_gained', 0)
                
                # SuperNode
                n_supernodes = 0
                if hasattr(agents, 'supernode_registry'):
                    stats = agents.supernode_registry.get_stats()
                    n_supernodes = stats.get('n_supernodes', 0)
                
                diagnostics.record_step(
                    step=step,
                    n_alive=result['n_alive'],
                    energies=batch.energies if batch.n > 0 else torch.tensor([0]),
                    node_counts=agents.state.node_counts[agents.alive_mask] if result['n_alive'] > 0 else torch.tensor([0]),
                    hebbian_active=hebbian_active,
                    hebbian_cost=hebbian_cost,
                    energy_gained=energy_gained,
                    n_supernodes=n_supernodes,
                    genome_lengths=agents.state.node_counts[agents.alive_mask]
                )
                
                print(f"Step {step:5d} | 存活: {result['n_alive']:4d} | "
                      f"Hebbian: {hebbian_active:4d} | "
                      f"SuperNode: {n_supernodes:2d}")
        
        # 最终报告
        diagnostics.print_report()
        
        # 保存
        if self.save_path:
            diagnostics.save_to_file(self.save_path)
        
        return diagnostics


if __name__ == "__main__":
    import sys
    
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    runner = DiagnosticRunner(
        n_steps=n_steps,
        initial_population=500,
        device=device,
        log_interval=500,
        save_path=f"diagnostics_run_{n_steps}.json"
    )
    runner.run()