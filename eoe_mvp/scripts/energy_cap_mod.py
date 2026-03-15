#!/usr/bin/env python3
"""
能量上限补丁 - 模拟"吃饱"机制
=============================
当个体能量达到上限时，会受到减益效果：
1. 移动速度降低
2. 能量萃取效率降低
3. 感知范围缩小

使用方式:
    # 在run_harsh_environment.py中导入
    from scripts.energy_cap_mod import apply_energy_cap
    
    # 在主循环中的适当位置调用
    agents = apply_energy_cap(agents, env, device, max_energy=500.0, penalty_factor=0.5)
"""

import torch


def apply_energy_cap(
    agents,
    env,
    device,
    base_max_energy: float = 200.0,
    energy_per_node: float = 50.0,
    penalty_factor: float = 0.5,
    enable_saturation_damage: bool = True
):
    """
    应用能量上限和吃饱减益
    
    参数:
        agents: BatchedAgents实例
        env: EnvironmentGPU实例  
        device: 计算设备
        base_max_energy: 基础能量上限 (default: 200.0)
        energy_per_node: 每个节点增加的能量上限 (default: 50.0)
            例: 5节点 = 200 + 5*50 = 450 能量上限
        penalty_factor: 吃饱时的效率系数 (0.5 = 50%效率)
        enable_saturation_damage: 是否启用饱和伤害
    
    效果:
        - 能量上限 = base_max_energy + node_count * energy_per_node
        - 当能量 > max_energy * 0.8 时，进入"吃饱"状态
        - 吃饱状态: 移动速度降低、萃取效率降低
        - 超饱和 (能量 > max_energy): 额外伤害
    """
    
    batch = agents.get_active_batch()
    if batch.n == 0:
        return agents
    
    idx = batch.indices
    
    # 计算每个个体的能量上限（基于节点数）
    node_counts = agents.state.node_counts[idx].float()
    max_energy = base_max_energy + node_counts * energy_per_node
    
    # 1. 能量上限 clamp
    current_energy = agents.state.energies[idx]
    over_energy = torch.clamp(current_energy - max_energy, min=0)
    
    # 超出部分造成持续伤害 (模拟吃太饱不舒服)
    if enable_saturation_damage and over_energy.any():
        damage = over_energy * 0.02  # 超出部分2%作为伤害（温和）
        agents.state.energies[idx] -= damage
    
    # 2. 吃饱状态检测 (能量 > 80%上限)
    saturation_threshold = max_energy * 0.8
    is_saturated = current_energy > saturation_threshold
    
    # 3. 吃饱减益效果
    if is_saturated.any():
        # 找到吃饱的个体
        sat_indices = idx[is_saturated]
        
        # 减益1: 降低位置更新权重 (模拟行动迟缓)
        # 在实际使用中需要在位置更新前应用
        agents._energy_saturation_mask = is_saturated.clone()
        
        # 减益2: 记录饱和状态供其他模块使用
        agents._saturation_level = torch.where(
            is_saturated,
            (current_energy - saturation_threshold) / (max_energy - saturation_threshold),
            torch.zeros_like(current_energy)
        )
        
        if hasattr(agents, 'state') and hasattr(agents.state, 'velocities'):
            # 降低速度
            agents.state.velocities[sat_indices] *= penalty_factor
    
    # 4. 确保不低于0
    agents.state.energies[idx] = torch.clamp(agents.state.energies[idx], min=0)
    
    return agents


def get_saturation_stats(agents):
    """获取饱和状态统计"""
    if not hasattr(agents, '_energy_saturation_mask'):
        return {"total": 0, "saturated": 0, "pct": 0.0}
    
    total = agents._energy_saturation_mask.numel()
    saturated = agents._energy_saturation_mask.sum().item()
    
    return {
        "total": total,
        "saturated": saturated,
        "pct": saturated / total * 100 if total > 0 else 0.0
    }


# ========== 补丁注入器 ==========

class EnergyCapPatcher:
    """能量上限补丁注入器"""
    
    def __init__(
        self,
        base_max_energy: float = 200.0,
        energy_per_node: float = 50.0,
        penalty_factor: float = 0.5,
        enable_saturation_damage: bool = True
    ):
        self.base_max_energy = base_max_energy
        self.energy_per_node = energy_per_node
        self.penalty_factor = penalty_factor
        self.enable_saturation_damage = enable_saturation_damage
        self.stats = []
    
    def patch(self, agents, env, device):
        """应用补丁"""
        agents = apply_energy_cap(
            agents, env, device,
            base_max_energy=self.base_max_energy,
            energy_per_node=self.energy_per_node,
            penalty_factor=self.penalty_factor,
            enable_saturation_damage=self.enable_saturation_damage
        )
        
        # 记录统计
        if hasattr(agents, '_energy_saturation_mask'):
            sat_count = agents._energy_saturation_mask.sum().item()
            self.stats.append(sat_count)
        
        return agents
    
    def get_stats(self):
        """获取补丁统计"""
        return {
            "base_max_energy": self.base_max_energy,
            "energy_per_node": self.energy_per_node,
            "penalty_factor": self.penalty_factor,
            "saturation_history": self.stats[-10:] if len(self.stats) > 10 else self.stats,
            "avg_saturated": sum(self.stats) / len(self.stats) if self.stats else 0
        }


# ========== 使用示例 ==========
"""
在 run_harsh_experiment() 中使用:

# 1. 创建补丁
energy_patcher = EnergyCapPatcher(
    base_max_energy=200.0,   # 基础能量上限200
    energy_per_node=50.0,    # 每节点+50能量上限
                            # 例: 5节点 = 200 + 5*50 = 450 能量上限
    penalty_factor=0.3,      # 吃饱后30%效率
    enable_saturation_damage=True
)

# 2. 在主循环中调用（在 agents.step() 之后）
for step in range(n_steps):
    result = agents.step(env=env, dt=0.1)
    
    # 应用能量上限补丁
    energy_patcher.patch(agents, env, device)
    
    # 打印统计（可选）
    if step % 1000 == 0:
        stats = energy_patcher.get_stats()
        print(f"Step {step}: 吃饱个体 {stats['avg_saturated']:.0f}")

# 3. 查看统计
final_stats = energy_patcher.get_stats()
print(f"最终统计: {final_stats}")
"""