"""
v14.0: 异步连续繁衍与"鲸落"机制
===============================
彻底抛弃代际循环，让时间变成连续的线。

核心特性:
- 预分配张量池 (Object Pool) + 掩码 (Mask)
- 能量驱动的自我繁衍 (Energy-Driven Mitosis)
- 鲸落机制 (Whale Fall) - 死亡反哺环境
- 持续运行，真实生态系统

Author: 104助手
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass


# ============================================================================
# 配置参数
# ============================================================================

class AsynchConfig:
    """异步池配置"""
    # 池大小
    MAX_AGENTS = 10000
    
    # 繁衍参数
    REPRODUCTION_THRESHOLD = 180.0  # 能量超过此值触发分裂
    CHILD_ENERGY_RATIO = 0.5        # 子代获得能量比例
    MUTATION_RATE = 0.1             # 基因变异概率
    MIN_REPRO_ENERGY = 30.0         # 能量低于此值不分裂
    SPAWN_RADIUS = 0.5              # 子代出生位置偏移半径
    
    # 鲸落参数
    WHALE_RETURN_RATIO = 0.8        # 死亡能量回归环境比例
    BIOMASS_PER_NODE = 10.0         # 每个脑节点的基础生物量能量
    
    # 代谢参数
    BASE_METABOLISM = 0.5           # 基础代谢率 (每步)
    ACTIVATION_COST = 0.01          # 激活能耗系数


# ============================================================================
# Agent 状态
# ============================================================================

@dataclass
class AgentState:
    """Agent 状态容器 (GPU 张量, 预分配 MAX_AGENTS 大小)"""
    positions: torch.Tensor      # [MAX_AGENTS, 2] (x, y)
    linear_velocity: torch.Tensor   # [MAX_AGENTS] 线速度
    angular_velocity: torch.Tensor  # [MAX_AGENTS] 角速度
    thetas: torch.Tensor         # [MAX_AGENTS] 朝向角
    
    # 能量系统 (v14.0)
    active_energy: torch.Tensor  # [MAX_AGENTS] 活动能量 (用于判断生死)
    structural_energy: torch.Tensor # [MAX_AGENTS] 结构能量 (躯体生物量)
    
    # 生物学特征
    node_counts: torch.Tensor    # [MAX_AGENTS] 脑节点数量 (用于计算 Biomass)
    
    @property
    def total_energy(self) -> torch.Tensor:
        """总能量 = 活动能量 + 结构能量"""
        return self.active_energy + self.structural_energy


class AsynchronousPool:
    """
    异步连续智能体池
    =================
    预分配内存池 + 掩码机制，支持持续运行的生态系统
    """
    
    def __init__(
        self,
        init_agents: int = 300,
        env_width: float = 100.0,
        env_height: float = 100.0,
        device: str = 'cuda:0',
        config: AsynchConfig = None
    ):
        self.config = config or AsynchConfig()
        self.env_width = env_width
        self.env_height = env_height
        self.device = device
        
        print(f"[AsynchronousPool] 初始化 on {device}")
        print(f"  池大小: {self.config.MAX_AGENTS}")
        print(f"  初始Agent: {init_agents}")
        
        # 预分配状态张量
        self._init_state_tensor()
        
        # 存活掩码
        self.alive_mask = torch.zeros(
            self.config.MAX_AGENTS, 
            device=device, 
            dtype=torch.bool
        )
        
        # 活跃索引缓存 (用于加速)
        self._active_indices: Optional[torch.Tensor] = None
        self._indices_dirty = True
        
        # 基因组字典 {idx: OperatorGenome}
        self.genomes: Dict[int, 'OperatorGenome'] = {}
        
        # 大脑矩阵
        self.brain_matrix: Optional[torch.Tensor] = None
        self.brain_masks: Optional[torch.Tensor] = None
        
        # 初始化第一批 Agent
        self._spawn_initial_agents(init_agents)
        
        print(f"  ✅ 初始化完成")
    
    def _init_state_tensor(self):
        """初始化预分配状态张量"""
        max_agents = self.config.MAX_AGENTS
        
        self.state = AgentState(
            positions = torch.zeros(max_agents, 2, device=self.device),
            linear_velocity = torch.zeros(max_agents, device=self.device),
            angular_velocity = torch.zeros(max_agents, device=self.device),
            thetas = torch.zeros(max_agents, device=self.device),
            active_energy = torch.zeros(max_agents, device=self.device),
            structural_energy = torch.zeros(max_agents, device=self.device),
            node_counts = torch.zeros(max_agents, device=self.device, dtype=torch.long)
        )
        
        print(f"  ✅ 预分配张量: {self.state.positions.shape}")
    
    def _spawn_initial_agents(self, n: int):
        """生成初始 Agent"""
        # 随机位置
        self.state.positions[:n] = torch.rand(n, 2, device=self.device) * \
            torch.tensor([self.env_width, self.env_height], device=self.device)
        
        # 随机朝向
        self.state.thetas[:n] = torch.rand(n, device=self.device) * 2 * np.pi
        
        # 初始能量
        init_energy = 100.0
        self.state.active_energy[:n] = init_energy
        self.state.structural_energy[:n] = init_energy * 0.5  # 结构能量
        
        # 节点数量 (简单大脑: 3-5 个节点)
        self.state.node_counts[:n] = torch.randint(3, 6, (n,), device=self.device)
        
        # 激活掩码
        self.alive_mask[:n] = True
        self._indices_dirty = True
        
        print(f"  ✅ 生成初始群体: {n}")
    
    # ============================================================================
    # 核心 API
    # ============================================================================
    
    def get_active_batch(self) -> 'ActiveBatch':
        """
        获取当前活跃 Agent 的切片
        
        Returns:
            ActiveBatch: 包含活跃状态切片的对象
        """
        if self._indices_dirty or self._active_indices is None:
            self._active_indices = self.alive_mask.nonzero(as_tuple=True)[0]
            self._indices_dirty = False
        
        n_active = len(self._active_indices)
        if n_active == 0:
            return ActiveBatch.empty(self.device)
        
        # 提取活跃状态
        idx = self._active_indices
        
        return ActiveBatch(
            indices=idx,
            positions=self.state.positions[idx],
            linear_velocity=self.state.linear_velocity[idx],
            angular_velocity=self.state.angular_velocity[idx],
            thetas=self.state.thetas[idx],
            active_energy=self.state.active_energy[idx],
            structural_energy=self.state.structural_energy[idx],
            node_counts=self.state.node_counts[idx],
        )
    
    def step_continuous(
        self,
        env: 'EnvironmentGPU',
        dt: float = 0.1,
        brain_fn: Optional[Callable] = None
    ) -> Dict[str, any]:
        """
        连续步进：物理 + 代谢 + 生死
        
        Args:
            env: GPU 环境
            dt: 时间步长
            brain_fn: 大脑前向传播函数 (可选)
            
        Returns:
            dict: 统计信息
        """
        batch = self.get_active_batch()
        if batch.n == 0:
            return {'n_alive': 0, 'births': 0, 'deaths': 0}
        
        n_before = batch.n
        
        # 1. 大脑推理 (如果有)
        if brain_fn is not None:
            brain_outputs = brain_fn(batch)
        else:
            # 默认零输出 (停止不动)
            brain_outputs = torch.zeros(batch.n, 5, device=self.device)
        
        # 2. 物理更新
        self._apply_physics(batch, brain_outputs, dt)
        
        # 3. 代谢扣除
        self._apply_metabolism(batch, dt)
        
        # 4. 环境交互 (摄食)
        self._apply_environment_interaction(batch, env)
        
        # 5. 鲸落处理 (死亡)
        deaths = self._process_deaths(batch, env)
        
        # 6. 分裂处理 (繁衍)
        births = self._process_reproduction(batch)
        
        # 7. 边界处理
        self._apply_boundaries(batch)
        
        n_after = self.get_active_batch().n
        
        return {
            'n_alive': n_after,
            'births': births,
            'deaths': deaths,
            'n_before': n_before,
            'n_after': n_after
        }
    
    def _apply_physics(self, batch: 'ActiveBatch', outputs: torch.Tensor, dt: float):
        """应用物理"""
        # 简化的物理模型
        # 输出: [permeability, thrust, turn, signal, defense]
        thrust = torch.tanh(outputs[:, 0]) * 5.0
        turn = torch.tanh(outputs[:, 1]) * 2.0
        
        # 线速度更新
        batch.linear_velocity *= 0.9  # 阻尼
        batch.linear_velocity += thrust * dt
        
        # 角速度更新
        batch.angular_velocity *= 0.9
        batch.angular_velocity += turn * dt
        
        # 位置更新
        batch.positions[:, 0] += batch.linear_velocity * torch.cos(batch.thetas) * dt
        batch.positions[:, 1] += batch.linear_velocity * torch.sin(batch.thetas) * dt
        
        # 朝向更新
        batch.thetas += batch.angular_velocity * dt
        
        # 写回状态
        self._write_back(batch)
    
    def _apply_metabolism(self, batch: 'ActiveBatch', dt: float):
        """代谢能耗"""
        # 基础代谢 + 运动代谢
        metabolic_cost = self.config.BASE_METABOLISM * dt
        kinetic_cost = (batch.linear_velocity.abs() + batch.angular_velocity.abs()) * \
                       self.config.ACTIVATION_COST * dt
        
        total_cost = metabolic_cost + kinetic_cost
        
        # 从活动能量扣除
        batch.active_energy = batch.active_energy - total_cost
        
        self._write_back(batch)
    
    def _apply_environment_interaction(self, batch: 'ActiveBatch', env: 'EnvironmentGPU'):
        """环境交互 - 摄食"""
        # 采样能量场
        if hasattr(env, 'energy_field') and env.energy_field is not None:
            try:
                energy_values = env.energy_field.sample_batch(batch.positions)
                # 摄取能量 (最多到满)
                feed_amount = energy_values * 0.3  # 30% 萃取率
                batch.active_energy = batch.active_energy + feed_amount
            except Exception:
                pass
        
        self._write_back(batch)
    
    def _process_deaths(self, batch: 'ActiveBatch', env: 'EnvironmentGPU') -> int:
        """
        处理死亡 - 鲸落机制
        
        鲸落能量 = Biomass (节点数×10) + max(0, 当前活动能量)
        """
        # 判定死亡
        death_mask = batch.active_energy <= 0
        
        if not death_mask.any():
            return 0
        
        death_indices = batch.indices[death_mask]
        n_deaths = len(death_indices)
        
        # 计算鲸落能量
        node_counts = self.state.node_counts[death_indices]
        biomass_energy = node_counts.float() * self.config.BIOMASS_PER_NODE
        
        # 活动能量 (取正值部分)
        active_energy = torch.clamp(self.state.active_energy[death_indices], min=0)
        
        # 总鲸落能量
        whale_energy = biomass_energy + active_energy * self.config.WHALE_RETURN_RATIO
        
        # 写入环境 (通过 scatter_add)
        if hasattr(env, 'energy_field') and env.energy_field is not None:
            try:
                death_positions = self.state.positions[death_indices]
                env.energy_field.scatter_add_(death_positions, whale_energy)
            except Exception:
                pass
        
        # 标记死亡
        self.alive_mask[death_indices] = False
        self._indices_dirty = True
        
        # 从基因组字典移除
        for idx in death_indices.tolist():
            if idx in self.genomes:
                del self.genomes[idx]
        
        return n_deaths
    
    def _process_reproduction(self, batch: 'ActiveBatch') -> int:
        """
        处理分裂 - 能量驱动的有丝分裂
        
        条件: active_energy > REPRODUCTION_THRESHOLD
        """
        # 判定分裂
        repro_mask = (batch.active_energy > self.config.REPRODUCTION_THRESHOLD) & \
                     (batch.active_energy > self.config.MIN_REPRO_ENERGY)
        
        if not repro_mask.any():
            return 0
        
        parent_indices = batch.indices[repro_mask]
        n_parents = len(parent_indices)
        
        # 寻找空槽位
        empty_slots = (~self.alive_mask).nonzero(as_tuple=True)[0]
        
        if len(empty_slots) == 0:
            return 0  # 池已满
        
        # 容量限制
        n_spawn = min(n_parents, len(empty_slots))
        
        parent_indices = parent_indices[:n_spawn]
        child_indices = empty_slots[:n_spawn]
        
        # 父代能量扣除 (50% 给子代)
        parent_energy = self.state.active_energy[parent_indices]
        child_energy = parent_energy * self.config.CHILD_ENERGY_RATIO
        
        self.state.active_energy[parent_indices] = parent_energy * (1 - self.config.CHILD_ENERGY_RATIO)
        self.state.active_energy[child_indices] = child_energy
        
        # 结构能量继承 (略有损耗)
        self.state.structural_energy[child_indices] = self.state.structural_energy[parent_indices]
        
        # 位置偏移 (避免数值奇点)
        offset = torch.randn(n_spawn, 2, device=self.device) * self.config.SPAWN_RADIUS
        self.state.positions[child_indices] = self.state.positions[parent_indices] + offset
        
        # 环形边界
        self.state.positions[child_indices, 0] = self.state.positions[child_indices, 0] % self.env_width
        self.state.positions[child_indices, 1] = self.state.positions[child_indices, 1] % self.env_height
        
        # 朝向继承 + 小扰动
        self.state.thetas[child_indices] = self.state.thetas[parent_indices] + \
            torch.randn(n_spawn, device=self.device) * 0.1
        
        # 速度继承
        self.state.linear_velocity[child_indices] = self.state.linear_velocity[parent_indices] * 0.5
        self.state.angular_velocity[child_indices] = self.state.angular_velocity[parent_indices] * 0.5
        
        # 节点数量继承 (可能突变)
        self.state.node_counts[child_indices] = self.state.node_counts[parent_indices]
        
        # 标记子代存活
        self.alive_mask[child_indices] = True
        self._indices_dirty = True
        
        # 基因组复制与变异 (TODO: 优化为 GPU 张量操作)
        for pi, ci in zip(parent_indices.tolist(), child_indices.tolist()):
            if pi in self.genomes:
                # 深拷贝 + 变异
                parent_genome = self.genomes[pi]
                child_genome = parent_genome.mutate(rate=self.config.MUTATION_RATE)
                self.genomes[ci] = child_genome
        
        return n_spawn
    
    def _apply_boundaries(self, batch: 'ActiveBatch'):
        """环形世界边界"""
        batch.positions[:, 0] = batch.positions[:, 0] % self.env_width
        batch.positions[:, 1] = batch.positions[:, 1] % self.env_height
        
        self._write_back(batch)
    
    def _write_back(self, batch: 'ActiveBatch'):
        """写回状态"""
        idx = batch.indices
        self.state.positions[idx] = batch.positions
        self.state.linear_velocity[idx] = batch.linear_velocity
        self.state.angular_velocity[idx] = batch.angular_velocity
        self.state.thetas[idx] = batch.thetas
        self.state.active_energy[idx] = batch.active_energy
        self.state.structural_energy[idx] = batch.structural_energy
    
    def get_population_stats(self) -> Dict:
        """获取种群统计"""
        batch = self.get_active_batch()
        if batch.n == 0:
            return {'n_alive': 0, 'mean_energy': 0, 'max_energy': 0}
        
        return {
            'n_alive': batch.n,
            'mean_energy': batch.active_energy.mean().item(),
            'max_energy': batch.active_energy.max().item(),
            'min_energy': batch.active_energy.min().item(),
            'mean_nodes': batch.node_counts.float().mean().item()
        }


@dataclass
class ActiveBatch:
    """活跃 Agent 的批量切片"""
    indices: torch.Tensor
    positions: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor
    thetas: torch.Tensor
    active_energy: torch.Tensor
    structural_energy: torch.Tensor
    node_counts: torch.Tensor
    
    @property
    def n(self) -> int:
        return len(self.indices)
    
    @staticmethod
    def empty(device: str) -> 'ActiveBatch':
        empty_tensor = torch.tensor([], device=device)
        return ActiveBatch(
            indices=empty_tensor,
            positions=empty_tensor.unsqueeze(0),
            linear_velocity=empty_tensor,
            angular_velocity=empty_tensor,
            thetas=empty_tensor,
            active_energy=empty_tensor,
            structural_energy=empty_tensor,
            node_counts=empty_tensor.long()
        )


# ============================================================================
# 测试
# ============================================================================

def test_asynchronous_pool():
    """测试异步池"""
    import time
    
    print("\n" + "="*60)
    print("🧪 异步连续池测试")
    print("="*60)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 创建环境 (简化版)
    from core.eoe.environment_gpu import EnvironmentGPU
    env = EnvironmentGPU(
        width=100, height=100,
        device=device,
        energy_field_enabled=True
    )
    
    # 创建异步池
    pool = AsynchronousPool(
        init_agents=100,
        env_width=100,
        env_height=100,
        device=device
    )
    
    print(f"\n初始种群: {pool.get_population_stats()}")
    
    # 运行一段时间
    print(f"\n运行 100 步...")
    for step in range(100):
        stats = pool.step_continuous(env, dt=0.1)
        
        if step % 20 == 0:
            print(f"  Step {step}: {stats}")
    
    final_stats = pool.get_population_stats()
    print(f"\n📊 最终统计: {final_stats}")
    
    # 测试分裂和死亡
    print("\n" + "="*60)
    print("🔥 压力测试: 强制提高能量触发分裂")
    print("="*60)
    
    # 手动给一些 Agent 加能量
    batch = pool.get_active_batch()
    if batch.n > 0:
        pool.state.active_energy[batch.indices[:10]] = 200.0
    
    for step in range(50):
        stats = pool.step_continuous(env, dt=0.1)
    
    print(f"分裂后统计: {pool.get_population_stats()}")
    
    print("\n✅ 测试完成")
    return pool, env


if __name__ == "__main__":
    test_asynchronous_pool()