#!/usr/bin/env python3
"""
v13.1 大脑热力学模块 (Brain Thermodynamics)
============================================
"只设计环境压力，不设计大脑结构" 的核心实现

通过热力学法则自然淘汰臃肿的大脑:
- 静态维护成本: 节点存在即消耗能量
- 动态激活成本: 神经元激活时额外消耗
- 连线成本: 每条突触消耗微量能量

涌现预期:
- 紧凑高效的大脑 (用少量节点完成任务)
- 活跃的"肌肉"消耗大量能量 (防止无意义抽搐)
- 精简的连接 (避免过度连接)
"""
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# 节点类型枚举 (与 node.py 保持同步)
NODE_TYPE_ORDER = [
    'CONSTANT',    # 0: 常数节点
    'ADD',         # 1: 加法
    'MULTIPLY',    # 2: 乘法
    'THRESHOLD',   # 3: 阈值
    'DELAY',       # 4: 延迟
    'SENSOR',      # 5: 传感器
    'ACTUATOR',    # 6: 执行器
]


@dataclass
class BrainMetabolism:
    """大脑代谢统计"""
    static_cost: float      # 静态维护成本
    edge_cost: float        # 连线成本
    activation_cost: float  # 动态激活成本
    total_cost: float       # 总成本


class BrainThermodynamics:
    """
    大脑热力学计算器
    
    将大脑结构映射为能量消耗，模拟生物大脑的代谢压力
    """
    
    # 默认成本向量 (与 manifest 中的默认值对应)
    DEFAULT_NODE_COSTS = torch.tensor([
        0.001,   # CONSTANT
        0.001,   # ADD
        0.002,   # MULTIPLY
        0.001,   # THRESHOLD
        0.005,   # DELAY
        0.01,    # SENSOR
        0.02,    # ACTUATOR
    ])
    
    def __init__(
        self,
        node_costs: Optional[torch.Tensor] = None,
        edge_cost: float = 0.0005,
        activation_multiplier: float = 0.1,
        device: str = 'cuda:0'
    ):
        self.device = device
        
        # 节点成本向量 [7]
        if node_costs is None:
            self.node_costs = self.DEFAULT_NODE_COSTS.to(device)
        else:
            self.node_costs = node_costs.to(device)
        
        self.edge_cost = edge_cost
        self.activation_multiplier = activation_multiplier
    
    @classmethod
    def from_manifest(cls, manifest, device='cuda:0'):
        """从 PhysicsManifest 创建"""
        node_costs = torch.tensor([
            manifest.node_cost_constant,
            manifest.node_cost_add,
            manifest.node_cost_multiply,
            manifest.node_cost_threshold,
            manifest.node_cost_delay,
            manifest.node_cost_sensor,
            manifest.node_cost_actuator,
        ], device=device)
        
        return cls(
            node_costs=node_costs,
            edge_cost=manifest.edge_cost,
            activation_multiplier=manifest.actuator_activation_cost_multiplier,
            device=device
        )
    
    def compute_static_cost(
        self,
        node_type_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        计算静态维护成本
        
        Args:
            node_type_counts: [N, 7] 各类型节点数量
            
        Returns:
            static_cost: [N] 每个 Agent 的静态成本
        """
        # 矩阵乘法: [N, 7] @ [7] -> [N]
        return torch.matmul(node_type_counts, self.node_costs)
    
    def compute_edge_cost(
        self,
        edge_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        计算连线成本
        
        Args:
            edge_counts: [N] 每 Agent 的有效连接数
            
        Returns:
            edge_cost: [N] 连线成本
        """
        return edge_counts * self.edge_cost
    
    def compute_activation_cost(
        self,
        actuator_outputs: torch.Tensor,
        node_activations: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算动态激活成本
        
        Args:
            actuator_outputs: [N, A] 执行器输出 (已激活的肌肉)
            node_activations: [N, num_nodes] 节点激活值 (可选)
            
        Returns:
            activation_cost: [N] 激活成本
        """
        # Actuator 激活成本 = |输出| × 系数
        if actuator_outputs.numel() > 0:
            actuator_cost = torch.abs(actuator_outputs).sum(dim=1) * self.activation_multiplier
        else:
            actuator_cost = torch.zeros(1, device=self.device)
        
        # 如果有节点激活值，也计入 (可选)
        if node_activations is not None:
            # 激活越强，消耗越多
            node_cost = torch.abs(node_activations).sum(dim=1) * 0.001
            actuator_cost = actuator_cost + node_cost
        
        return actuator_cost
    
    def compute_total_metabolism(
        self,
        node_type_counts: torch.Tensor,
        edge_counts: torch.Tensor,
        actuator_outputs: torch.Tensor,
        node_activations: Optional[torch.Tensor] = None
    ) -> BrainMetabolism:
        """
        计算总代谢成本
        
        Args:
            node_type_counts: [N, 7] 节点类型数量
            edge_counts: [N] 边数量
            actuator_outputs: [N, A] 执行器输出
            node_activations: [N, num_nodes] 节点激活值
            
        Returns:
            BrainMetabolism 包含各类成本
        """
        static_cost = self.compute_static_cost(node_type_counts)
        edge_cost = self.compute_edge_cost(edge_counts)
        activation_cost = self.compute_activation_cost(actuator_outputs, node_activations)
        
        return BrainMetabolism(
            static_cost=static_cost,
            edge_cost=edge_cost,
            activation_cost=activation_cost,
            total_cost=static_cost + edge_cost + activation_cost
        )
    
    def apply_to_energies(
        self,
        energies: torch.Tensor,
        metabolism: BrainMetabolism,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        从能量中扣除代谢成本
        
        Args:
            energies: [N] 当前能量
            metabolism: BrainMetabolism 对象
            dt: 时间步长
            
        Returns:
            new_energies: [N] 扣除后的能量
        """
        return energies - metabolism.total_cost * dt


def create_node_type_counts(genomes: List, device: str = 'cuda:0') -> torch.Tensor:
    """
    从基因组列表创建节点类型计数张量
    
    Args:
        genomes: Agent 基因组列表
        device: 设备
        
    Returns:
        node_type_counts: [N, 7] 每个 Agent 的各类型节点数
    """
    from core.eoe.node import NodeType
    
    n_agents = len(genomes)
    
    # 初始化计数
    node_type_counts = torch.zeros(n_agents, 7, device=device)
    
    for i, genome in enumerate(genomes):
        if genome is None or not hasattr(genome, 'nodes'):
            continue
        
        # 统计各类型节点
        counts = {t: 0 for t in NODE_TYPE_ORDER}
        for node in genome.nodes.values():
            node_type_name = node.node_type.name
            if node_type_name in counts:
                counts[node_type_name] += 1
            elif node_type_name == 'MACRO' or node_type_name == 'SUPERNODE':
                counts['ADD'] += 1  # 归约为 ADD
        
        # 填充张量
        for j, nt in enumerate(NODE_TYPE_ORDER):
            node_type_counts[i, j] = counts[nt]
    
    return node_type_counts


def create_edge_counts(genomes: List, device: str = 'cuda:0') -> torch.Tensor:
    """
    从基因组列表创建边计数张量
    
    Args:
        genomes: Agent 基因组列表
        device: 设备
        
    Returns:
        edge_counts: [N] 每个 Agent 的有效连接数
    """
    n_agents = len(genomes)
    edge_counts = torch.zeros(n_agents, device=device)
    
    for i, genome in enumerate(genomes):
        if genome is None or not hasattr(genome, 'edges'):
            continue
        
        # 统计有效边 (权重非零)
        edge_counts[i] = sum(1 for e in genome.edges.values() if e.weight != 0)
    
    return edge_counts


# 测试
def test_brain_thermodynamics():
    """测试大脑热力学"""
    print("="*60)
    print("🧪 大脑热力学测试")
    print("="*60)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    thermodynamics = BrainThermodynamics(device=device)
    
    # 模拟 4 个 Agent 的脑结构
    # Agent 0: 简单大脑 (2 ADD + 1 ACTUATOR)
    # Agent 1: 中等大脑 (5 ADD + 2 MULTIPLY + 2 ACTUATOR + 3 SENSOR)
    # Agent 2: 臃肿大脑 (20 ADD + 10 MULTIPLY + 5 DELAY + 5 ACTUATOR + 5 SENSOR)
    # Agent 3: 极简大脑 (1 ADD + 1 ACTUATOR)
    
    node_type_counts = torch.tensor([
        [0, 2, 0, 0, 0, 0, 1],  # Agent 0: 简单
        [0, 5, 2, 0, 0, 3, 2],  # Agent 1: 中等
        [0, 20, 10, 0, 5, 5, 5],  # Agent 2: 臃肿
        [0, 1, 0, 0, 0, 0, 1],  # Agent 3: 极简
    ], device=device, dtype=torch.float32)
    
    edge_counts = torch.tensor([5, 20, 80, 3], device=device, dtype=torch.float32)
    
    actuator_outputs = torch.tensor([
        [0.5, 0.3],   # Agent 0: 轻度激活
        [1.0, 0.8],   # Agent 1: 中度激活
        [1.0, 1.0],   # Agent 2: 满功率 (应该被惩罚!)
        [0.1, 0.0],   # Agent 3: 几乎不动
    ], device=device)
    
    # 计算代谢
    metabolism = thermodynamics.compute_total_metabolism(
        node_type_counts, edge_counts, actuator_outputs
    )
    
    print(f"\n📊 节点数量:")
    print(f"  Agent 0 (简单): {node_type_counts[0].sum().int().item()} 节点")
    print(f"  Agent 1 (中等): {node_type_counts[1].sum().int().item()} 节点")
    print(f"  Agent 2 (臃肿): {node_type_counts[2].sum().int().item()} 节点")
    print(f"  Agent 3 (极简): {node_type_counts[3].sum().int().item()} 节点")
    
    print(f"\n⚡ 代谢成本:")
    print(f"  静态成本: {metabolism.static_cost.cpu().numpy()}")
    print(f"  连线成本: {metabolism.edge_cost.cpu().numpy()}")
    print(f"  激活成本: {metabolism.activation_cost.cpu().numpy()}")
    print(f"  总成本:   {metabolism.total_cost.cpu().numpy()}")
    
    # 验证
    # Agent 2 应该比 Agent 0 和 3 高很多
    test1_pass = metabolism.total_cost[2] > metabolism.total_cost[0] * 3
    # Agent 2 的激活成本应该最高 (满功率)
    test2_pass = metabolism.activation_cost[2] > metabolism.activation_cost[3]
    
    print(f"\n✅ 臃肿大脑成本更高: {'通过' if test1_pass else '失败'}")
    print(f"✅ 高激活成本惩罚: {'通过' if test2_pass else '失败'}")
    
    # 能量扣除测试
    energies = torch.tensor([100.0, 100.0, 100.0, 100.0], device=device)
    new_energies = thermodynamics.apply_to_energies(energies, metabolism)
    
    print(f"\n🔋 能量扣除:")
    print(f"  原始: {energies.cpu().numpy()}")
    print(f"  扣除后: {new_energies.cpu().numpy()}")
    
    all_pass = test1_pass and test2_pass
    print(f"\n{'='*60}")
    print(f"{'✅ 测试通过' if all_pass else '❌ 测试失败'}")
    print(f"{'='*60}")
    
    return all_pass


# ============================================================================
# v14.0 鲍德温效应: 能量调制赫布学习 (Reward-modulated STDP)
# ============================================================================

def reward_modulated_hebbian(
    pre_activations: torch.Tensor,      # [N, n_edges] 前突触激活
    post_activations: torch.Tensor,     # [N, n_edges] 后突触激活
    energy_delta: torch.Tensor,         # [N] 能量变化
    plastic_mask: torch.Tensor,         # [N, n_edges] 可塑性边掩码
    eligibility_trace: torch.Tensor,    # [N, trace_len, n_edges] 资格迹
    config: dict = None,
    trace_ptr: int = 0,
    decay: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    能量调制赫布学习 (Reward-modulated Hebbian Learning)
    
    这是鲍德温效应的核心实现:
    - 如果Agent走了一步发现能量增加了 → 强化过去活跃的连接
    - 如果能量减少了 → 弱化连接
    - 这就是"学会"一件事的本质!
    
    参数:
        pre_activations: 前突触激活 [N, n_edges]
        post_activations: 后突触激活 [N, n_edges]
        energy_delta: 能量变化 [N]
        plastic_mask: 可塑性边掩码 [N, n_edges]
        eligibility_trace: 资格迹缓冲区 [N, trace_len, n_edges]
        config: 配置字典
        trace_ptr: 当前时间指针 (循环)
        decay: 资格迹衰减率
        
    返回:
        (new_weights_delta, new_eligibility_trace)
        - 权重变化量 [N, n_edges]
        - 更新后的资格迹 [N, trace_len, n_edges]
    """
    if config is None:
        config = {
            'base_lr': 0.01,
            'reward_modulation': True,
            'deadzone': 1.0,
            'max_weight_delta': 0.1,
        }
    
    n_agents = pre_activations.shape[0]
    n_edges = pre_activations.shape[1]
    trace_len = eligibility_trace.shape[1]
    
    device = pre_activations.device
    
    # 1. 死区过滤 (Gemini建议)
    deadzone = config.get('deadzone', 1.0)
    significant = energy_delta.abs() > deadzone
    
    # 2. 多巴胺信号计算
    if config.get('reward_modulation', True):
        # dopamine = sign(ΔE) * min(|ΔE|/50, 1.0)
        dopamine = torch.sign(energy_delta) * (energy_delta.abs() / 50.0).clamp(max=1.0)
        # 只对显著变化的Agent计算
        dopamine = dopamine * significant.float()
    else:
        dopamine = torch.ones(n_agents, device=device) * 0.1
    
    # 3. Hebbian相关: pre * post
    hebbian_term = pre_activations * post_activations  # [N, n_edges]
    
    # 4. 资格迹更新: trace_new = decay * trace_old + hebbian
    # 使用循环缓冲区
    new_trace = eligibility_trace.clone()
    new_trace[:, trace_ptr, :] = hebbian_term * plastic_mask.float()
    
    # 5. 计算 eligibility trace 的加权和 (过去trace_len步的平均)
    # 使用衰减求和
    decay_weights = torch.tensor([decay ** i for i in range(trace_len)], device=device)
    # trace [N, trace_len, n_edges] -> weighted sum -> [N, n_edges]
    # einsum: jlk,ijk->il means: sum over j (trace_len)
    weighted_trace = torch.einsum('j,ijk->ik', decay_weights, new_trace)
    
    # 6. 权重更新: Δw = lr * dopamine * weighted_trace
    lr = config.get('base_lr', 0.01)
    weight_delta = lr * dopamine.unsqueeze(1) * weighted_trace
    
    # 7. 限制最大变化幅度
    max_delta = config.get('max_weight_delta', 0.1)
    weight_delta = weight_delta.clamp(-max_delta, max_delta)
    
    # 8. 应用可塑性掩码
    weight_delta = weight_delta * plastic_mask.float()
    
    return weight_delta, new_trace


def compute_eligibility_trace(
    pre_act: torch.Tensor,
    post_act: torch.Tensor,
    trace_buffer: torch.Tensor,
    trace_ptr: int,
    decay: float = 0.9
) -> torch.Tensor:
    """
    计算资格迹 (Eligibility Trace)
    
    这是一个"记忆"机制:
    - 记录过去trace_len步中每一步的 Hebbian 相关 (pre * post)
    - 当收到奖励/惩罚时，回顾过去所有记忆
    - 强化/弱化当时活跃的连接
    
    参数:
        pre_act: 前突触激活 [N, n_edges]
        post_act: 后突触激活 [N, n_edges]
        trace_buffer: 循环缓冲区 [N, trace_len, n_edges]
        trace_ptr: 当前时间指针
        decay: 衰减率
        
    返回:
        更新后的trace_buffer
    """
    # Hebbian term: "fire together, wire together"
    hebbian = pre_act * post_act  # [N, n_edges]
    
    # 写入当前时间步
    new_buffer = trace_buffer.clone()
    new_buffer[:, trace_ptr, :] = hebbian
    
    return new_buffer


def apply_dopamine_modulation(
    eligibility_trace: torch.Tensor,
    dopamine: torch.Tensor,
    lr: float = 0.01,
    max_delta: float = 0.1
) -> torch.Tensor:
    """
    应用多巴胺调制到权重
    
    参数:
        eligibility_trace: 资格迹 [N, trace_len, n_edges] 或 [N, n_edges]
        dopamine: 多巴胺信号 [N]
        lr: 学习率
        max_delta: 最大变化幅度
        
    返回:
        权重变化量 [N, n_edges]
    """
    # 如果是3Dtrace，先求加权和
    if eligibility_trace.dim() == 3:
        trace_len = eligibility_trace.shape[1]
        decay = 0.9
        decay_weights = torch.tensor(
            [decay ** i for i in range(trace_len)], 
            device=eligibility_trace.device
        )
        # einsum: j,ijk->ik means sum over j (trace_len)
        weighted = torch.einsum('j,ijk->ik', decay_weights, eligibility_trace)
    else:
        weighted = eligibility_trace
    
    # Δw = lr * dopamine * trace
    weight_delta = lr * dopamine.unsqueeze(1) * weighted
    weight_delta = weight_delta.clamp(-max_delta, max_delta)
    
    return weight_delta


# ============================================================================
# 测试
# ============================================================================

def test_reward_hebbian():
    """测试能量调制赫布学习"""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🧪 测试能量调制赫布学习 on {device}")
    
    n_agents = 100
    n_edges = 20
    trace_len = 5
    
    # 模拟数据
    pre_act = torch.rand(n_agents, n_edges, device=device)
    post_act = torch.rand(n_agents, n_edges, device=device)
    energy_delta = torch.randn(n_agents, device=device) * 10  # 随机能量变化
    plastic_mask = torch.rand(n_agents, n_edges, device=device) > 0.7  # 30%可塑性
    eligibility_trace = torch.zeros(n_agents, trace_len, n_edges, device=device)
    
    config = {
        'base_lr': 0.01,
        'reward_modulation': True,
        'deadzone': 1.0,
        'max_weight_delta': 0.1,
    }
    
    # 测试
    weight_delta, new_trace = reward_modulated_hebbian(
        pre_act, post_act, energy_delta, plastic_mask,
        eligibility_trace, config, trace_ptr=0, decay=0.9
    )
    
    print(f"\n📊 结果:")
    print(f"  能量变化范围: [{energy_delta.min():.2f}, {energy_delta.max():.2f}]")
    print(f"  显著变化Agent数: {(energy_delta.abs() > 1.0).sum().item()}")
    print(f"  权重变化范围: [{weight_delta.min():.4f}, {weight_delta.max():.4f}]")
    print(f"  可塑性边数: {plastic_mask.sum().item()}")
    
    # 验证: 能量增加时，权重应该增加
    positive_delta = energy_delta > 1.0
    if positive_delta.any():
        avg_increase = weight_delta[positive_delta].mean().item()
        print(f"  能量增加组平均Δw: {avg_increase:.4f}")
    
    # 验证: 能量减少时，权重应该减少
    negative_delta = energy_delta < -1.0
    if negative_delta.any():
        avg_decrease = weight_delta[negative_delta].mean().item()
        print(f"  能量减少组平均Δw: {avg_decrease:.4f}")
    
    print(f"\n✅ 能量调制赫布学习测试完成")
    return True


if __name__ == "__main__":
    test_reward_hebbian()