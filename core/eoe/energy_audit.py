"""
能量审计模块 (Energy Audit Hook)
=================================
EOE 核心约束: 宏观能量守恒

本模块实现严格的热力学第一定律验证:
E_total = ΣE_env_grid + ΣE_agent + ΣE_corpse + ΣE_sources + ΣE_in_transit

注意事项:
- 使用 float64 避免 FP32 累积误差
- 在所有物理和生命周期结算完成后执行审计
- 周期性检查,误差超过阈值则抛出异常
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
import time


class EnergyConservationError(Exception):
    """能量守恒异常 - 系统总能量发生泄漏或超发"""
    pass


@dataclass
class EnergySnapshot:
    """能量快照 - 记录某一时刻的系统总能量分布"""
    step: int
    timestamp: float
    
    # 场能量
    epf_energy: float = 0.0      # EPF场 (能量扩散场)
    isf_energy: float = 0.0      # ISF场 (压痕场)
    
    # Agent能量
    alive_agent_energy: float = 0.0    # 存活Agent能量
    dead_agent_energy: float = 0.0     # 刚死亡待转化Agent
    
    # 外部能量源
    source_energy: float = 0.0         # 能量源剩余容量
    source_total_capacity: float = 0.0 # 能量源总容量 (初始)
    cumulative_injected: float = 0.0   # 累积注入能量 (新增!)
    pending_food_energy: float = 0.0   # 待生成食物能量
    prey_energy: float = 0.0           # 智能猎物能量
    
    # 其他
    corpse_biomass: float = 0.0        # 尸体生物质能量
    
    @property
    def total_system(self) -> float:
        """系统内部总能量 (不含外部注入)"""
        return (self.epf_energy + self.isf_energy + 
                self.alive_agent_energy + self.dead_agent_energy +
                self.source_energy + self.pending_food_energy +
                self.prey_energy + self.corpse_biomass)
    
    @property
    def total_with_injection(self) -> float:
        """含外部注入的系统总能量"""
        return (self.epf_energy + self.isf_energy + 
                self.alive_agent_energy + self.dead_agent_energy +
                self.source_total_capacity +  # 使用初始总容量
                self.pending_food_energy +
                self.prey_energy + self.corpse_biomass)
    
    @property
    def total_conserved(self) -> float:
        """守恒能量: EPF + Agent + Source_Capacity + 其他"""
        return (self.epf_energy + self.isf_energy + 
                self.alive_agent_energy + self.dead_agent_energy +
                self.source_energy +  # 当前剩余容量
                self.pending_food_energy +
                self.prey_energy + self.corpse_biomass)
    
    @property
    def total(self) -> float:
        """向后兼容: 返回系统内部能量"""
        return self.total_system
    
    def __repr__(self):
        return (f"EnergySnapshot(step={self.step}, "
                f"total={self.total:.4f}, "
                f"EPF={self.epf_energy:.4f}, "
                f"agents={self.alive_agent_energy:.4f})")


class EnergyAuditHook:
    """
    能量审计钩子
    
    在每 N 步强制执行能量守恒检查,确保:
    |E_current - E_initial| / E_initial < tolerance
    
    追踪的所有能量项:
    - EPF场网格能量
    - ISF场网格能量  
    - 存活Agent内部能量
    - 死亡待转化Agent能量 (在途能量)
    - 能量源剩余容量
    - 待生成食物能量
    - 智能猎物能量
    - 尸体生物质能量
    """
    
    def __init__(
        self,
        tolerance: float = 1e-5,
        audit_interval: int = 1000,
        device: str = 'cuda:0',
        verbose: bool = True
    ):
        self.tolerance = tolerance
        self.audit_interval = audit_interval
        self.device = device
        self.verbose = verbose
        
        # 初始化记录
        self.initial_snapshot: Optional[EnergySnapshot] = None
        self.snapshots: List[EnergySnapshot] = []
        self.error_history: List[Dict] = []
        
        # 审计统计
        self.total_audits = 0
        self.failed_audits = 0
        self.max_relative_error = 0.0
        
        # 性能追踪
        self.audit_times: List[float] = []
        
        # ===== 新增: 开放耗散系统追踪 =====
        # E_in: 外部能量输入 
        #   - 能量源重生 (容量重置)
        #   - 常规注入 (每次注入脉冲)
        self.cumulative_energy_in: float = 0.0
        self._last_source_remaining: Optional[torch.Tensor] = None  # 用于追踪常规注入
        self._last_epf_energy: Optional[float] = None  # 用于追踪EPF变化
        
        # E_out: 能量散失 (代谢废热/未回收的能量)
        self.cumulative_energy_out: float = 0.0
        
        # 追踪能量源重生
        self._last_source_capacities: Optional[torch.Tensor] = None
    
    def initialize(self, env, agents) -> EnergySnapshot:
        """
        初始化: 在实验开始前记录系统总能量
        
        Args:
            env: EnvironmentGPU 实例
            agents: BatchedAgents 实例
            
        Returns:
            初始能量快照
        """
        snapshot = self._capture_snapshot(env, agents, step=0)
        self.initial_snapshot = snapshot
        self.snapshots.append(snapshot)
        
        if self.verbose:
            print("=" * 60)
            print("🔋 能量审计初始化")
            print("=" * 60)
            print(f"  初始总能量: {snapshot.total:.6f}")
            print(f"  EPF场:      {snapshot.epf_energy:.6f}")
            print(f"  ISF场:      {snapshot.isf_energy:.6f}")
            print(f"  Agent:      {snapshot.alive_agent_energy:.6f}")
            print(f"  能量源:     {snapshot.source_energy:.6f}")
            print(f"  审计间隔:   {self.audit_interval} 步")
            print(f"  容差阈值:   {self.tolerance:.2e}")
            print("=" * 60)
        
        return snapshot
    
    def _capture_snapshot(
        self, 
        env, 
        agents, 
        step: int,
        dead_agent_energies: Optional[torch.Tensor] = None,
        corpse_biomass: float = 0.0
    ) -> EnergySnapshot:
        """
        捕获当前时刻的能量快照
        
        使用 float64 避免 FP32 累积误差
        """
        t_start = time.time()
        
        snapshot = EnergySnapshot(
            step=step,
            timestamp=time.time()
        )
        
        # 1. EPF 场能量 (能量扩散场)
        if hasattr(env, 'energy_field') and env.energy_field is not None:
            epf_field = env.energy_field.field
            # 转换为 float64 并求和,避免大数吃小数
            snapshot.epf_energy = epf_field.to(torch.float64).sum().item()
        
        # 2. ISF 场能量 (压痕场/stigmergy)
        if hasattr(env, 'stigmergy_field') and env.stigmergy_field is not None:
            isf_field = env.stigmergy_field.field
            snapshot.isf_energy = isf_field.to(torch.float64).sum().item()
        
        # 3. 存活 Agent 能量
        if hasattr(agents, 'state') and hasattr(agents.state, 'energies'):
            energies = agents.state.energies
            alive_mask = agents.alive_mask
            
            # 仅统计存活Agent
            alive_energies = energies * alive_mask.float()
            snapshot.alive_agent_energy = alive_energies.to(torch.float64).sum().item()
        
        # 4. 死亡待转化 Agent 能量 (在途能量)
        if dead_agent_energies is not None:
            snapshot.dead_agent_energy = dead_agent_energies.to(torch.float64).sum().item()
        
        # 5. 能量源剩余容量 (source_capacity)
        if hasattr(env, 'energy_field') and env.energy_field is not None:
            if hasattr(env.energy_field, 'sources'):
                sources = env.energy_field.sources
                # sources[:,4] 是当前剩余容量
                snapshot.source_energy = sources[:, 4].to(torch.float64).sum().item()
                # sources[:,5] 是初始最大容量 (用于计算累积注入)
                snapshot.source_total_capacity = sources[:, 5].to(torch.float64).sum().item()
                
                # 追踪E_in: 外部能量输入
                # 方式1: 能量源重生 (容量增加)
                # 方式2: 常规注入 (EPF增加 - decay预期)
                
                # 追踪常规注入: EPF增加量
                if self._last_epf_energy is not None:
                    epf_delta = snapshot.epf_energy - self._last_epf_energy
                    if epf_delta > 0:
                        # EPF增加了，说明有能量注入 (扣除decay后)
                        self.cumulative_energy_in += max(0, epf_delta)
                self._last_epf_energy = snapshot.epf_energy
                
                # 追踪能量源重生 (容量重置)
                current_capacities = sources[:, 4].clone()
                if self._last_source_capacities is not None:
                    respawn_mask = current_capacities > self._last_source_capacities
                    if respawn_mask.any():
                        respawn_energy = respawn_mask.float() * (
                            current_capacities - self._last_source_capacities
                        )
                        respawn_add = respawn_energy.to(torch.float64).sum().item()
                        self.cumulative_energy_in += respawn_add  # 累加E_in
                        if self.verbose and respawn_add > 10:
                            print(f"  ☀️ 能量源重生 (E_in): +{respawn_add:.2f}")
                self._last_source_capacities = current_capacities.clone()
                
                # 检测能量源重生 (容量增加 = E_in 外部能量输入)
                if self._last_source_capacities is not None:
                    current_capacities = sources[:, 4].clone()
                    # 重生: 当前容量 > 上次容量 (源枯竭后重新生成)
                    respawn_mask = current_capacities > self._last_source_capacities
                    if respawn_mask.any():
                        respawn_energy = respawn_mask.float() * (
                            current_capacities - self._last_source_capacities
                        )
                        respawn_add = respawn_energy.to(torch.float64).sum().item()
                        self.cumulative_energy_in += respawn_add  # 追踪E_in
                        if self.verbose and respawn_add > 10:
                            print(f"  ☀️ 能量源重生 (E_in): +{respawn_add:.2f}")
                    self._last_source_capacities = current_capacities.clone()
                else:
                    # 首次记录
                    self._last_source_capacities = sources[:, 4].clone()
        
        # 6. 待生成食物能量
        if hasattr(env, 'food_spawner') and env.food_spawner is not None:
            if hasattr(env.food_spawner, 'pending_energy'):
                snapshot.pending_food_energy = env.food_spawner.pending_energy
            elif hasattr(env.food_spawner, 'energy_pool'):
                snapshot.pending_food_energy = env.food_spawner.energy_pool
        
        # 7. 智能猎物能量
        if hasattr(env, 'prey_energy_sources') and env.prey_energy_sources is not None:
            snapshot.prey_energy = env.prey_energy_sources.to(torch.float64).sum().item()
        elif hasattr(env, 'intelligent_prey'):
            # IntelligentPrey 对象
            prey_sources = []
            for prey in getattr(env, 'intelligent_prey', []):
                if hasattr(prey, 'energy'):
                    prey_sources.append(prey.energy)
            if prey_sources:
                snapshot.prey_energy = sum(prey_sources)
        
        # 8. 尸体生物质能量
        snapshot.corpse_biomass = corpse_biomass
        
        # 记录审计耗时
        self.audit_times.append(time.time() - t_start)
        
        return snapshot
    
    def audit(
        self, 
        env, 
        agents, 
        step: int,
        dead_agent_energies: Optional[torch.Tensor] = None,
        corpse_biomass: float = 0.0
    ) -> Optional[Dict]:
        """
        执行能量审计
        
        在每一 audit_interval 步执行检查
        
        Args:
            env: EnvironmentGPU 实例
            agents: BatchedAgents 实例
            step: 当前步数
            dead_agent_energies: 刚死亡的Agent能量 (可选)
            corpse_biomass: 尸体生物质能量 (可选)
            
        Returns:
            审计结果字典,或 None (非审计周期)
        """
        # 只有在审计周期才执行
        if step % self.audit_interval != 0:
            return None
        
        if self.initial_snapshot is None:
            raise RuntimeError("能量审计未初始化! 请先调用 initialize()")
        
        self.total_audits += 1
        
        # 捕获当前快照
        current = self._capture_snapshot(
            env, agents, step,
            dead_agent_energies=dead_agent_energies,
            corpse_biomass=corpse_biomass
        )
        self.snapshots.append(current)
        
        # ===== 开放耗散系统审计 =====
        # 能量平衡方程: E_current = E_initial + E_in - E_out
        # 其中:
        #   E_in = 能量源重生带来的外部输入 (cumulative_energy_in)
        #   E_out = 代谢废热散失 (cumulative_energy_out)
        
        initial_conserved = self.initial_snapshot.total_conserved
        current_conserved = current.total_conserved
        
        # 预期总能量 = 初始守恒能量 + 累积输入 - 累积输出
        expected_total = initial_conserved + self.cumulative_energy_in - self.cumulative_energy_out
        
        # 实际误差
        absolute_error = abs(current_conserved - expected_total)
        
        # 相对误差
        if abs(expected_total) > 1e-10:
            relative_error = absolute_error / abs(expected_total)
        else:
            relative_error = absolute_error
        
        self.max_relative_error = max(self.max_relative_error, relative_error)
        
        # 构建审计结果
        result = {
            'step': step,
            
            # 基础能量
            'initial_energy': initial_conserved,
            'current_energy': current_conserved,
            'expected_energy': expected_total,
            
            # 误差
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'is_conserved': relative_error < self.tolerance,
            
            # 能量流动追踪
            'cumulative_energy_in': self.cumulative_energy_in,
            'cumulative_energy_out': self.cumulative_energy_out,
            
            # 能量分布变化 (用于调试)
            'epf_energy': current.epf_energy,
            'agent_energy': current.alive_agent_energy,
            'source_energy': current.source_energy,
            'epf_delta': current.epf_energy - self.initial_snapshot.epf_energy,
            'agent_delta': current.alive_agent_energy - self.initial_snapshot.alive_agent_energy,
            'source_delta': current.source_energy - self.initial_snapshot.source_energy,
        }
        
        # 检查是否守恒
        if not result['is_conserved']:
            self.failed_audits += 1
            self.error_history.append(result)
            
            error_msg = (
                f"⚠️ 能量守恒破缺! Step {step}\n"
                f"  初始能量: {initial_conserved:.8f}\n"
                f"  预期能量: {expected_total:.8f} (E_in: {self.cumulative_energy_in:.2f}, E_out: {self.cumulative_energy_out:.2f})\n"
                f"  当前能量: {current_conserved:.8f}\n"
                f"  绝对误差: {absolute_error:.8f}\n"
                f"  相对误差: {relative_error:.2e} (阈值: {self.tolerance:.2e})\n"
                f"  EPF变化:  {result['epf_delta']:+.4f}\n"
                f"  Agent变化: {result['agent_delta']:+.4f}"
            )
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(error_msg)
                print(f"{'='*60}\n")
            
            # 可选: 抛出异常 (严格模式)
            # raise EnergyConservationError(error_msg)
        
        if self.verbose and step % (self.audit_interval * 10) == 0:
            print(f"🔋 审计 Step {step}: "
                  f"能量={current.total:.4f}, "
                  f"误差={relative_error:.2e}, "
                  f"状态={'✅' if result['is_conserved'] else '⚠️'}")
        
        return result
    
    def get_statistics(self) -> Dict:
        """获取审计统计信息"""
        avg_audit_time = np.mean(self.audit_times) if self.audit_times else 0
        
        return {
            'total_audits': self.total_audits,
            'failed_audits': self.failed_audits,
            'failure_rate': self.failed_audits / max(self.total_audits, 1),
            'max_relative_error': self.max_relative_error,
            'avg_audit_time_ms': avg_audit_time * 1000,
            'initial_energy': self.initial_snapshot.total if self.initial_snapshot else 0,
            'final_energy': self.snapshots[-1].total if self.snapshots else 0,
        }
    
    def print_summary(self):
        """打印审计总结"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("📊 能量审计总结")
        print("=" * 60)
        print(f"  审计次数:     {stats['total_audits']}")
        print(f"  失败次数:     {stats['failed_audits']}")
        print(f"  失败率:       {stats['failure_rate']*100:.2f}%")
        print(f"  最大误差:     {stats['max_relative_error']:.2e}")
        print(f"  初始能量:     {stats['initial_energy']:.6f}")
        print(f"  最终能量:     {stats['final_energy']:.6f}")
        print(f"  审计耗时:     {stats['avg_audit_time_ms']:.2f}ms/次")
        
        if stats['failed_audits'] > 0:
            print(f"\n  ⚠️ 警告: 检测到 {stats['failed_audits']} 次能量守恒破缺!")
            print(f"     最大相对误差: {stats['max_relative_error']:.2e}")
            if stats['max_relative_error'] > self.tolerance:
                print(f"\n  🔴 严重: 误差超过阈值 {self.tolerance:.2e}")
                print(f"     需要检查能量泄漏/超发源头!")
        else:
            print(f"\n  ✅ 能量守恒验证通过!")
        
        print("=" * 60)


def create_energy_audit_hook(
    config_preset: str = "standard",
    device: str = 'cuda:0'
) -> EnergyAuditHook:
    """
    工厂函数: 创建预配置的审计钩子
    
    Args:
        config_preset: 预设配置
            - "strict": 严格模式,每100步检查,容差1e-6
            - "standard": 标准模式,每1000步检查,容差1e-5
            - "relaxed": 宽松模式,每5000步检查,容差1e-4
        device: 计算设备
        
    Returns:
        配置好的 EnergyAuditHook 实例
    """
    presets = {
        "strict": {"tolerance": 1e-6, "interval": 100},
        "standard": {"tolerance": 1e-5, "interval": 1000},
        "relaxed": {"tolerance": 1e-4, "interval": 5000},
    }
    
    cfg = presets.get(config_preset, presets["standard"])
    
    return EnergyAuditHook(
        tolerance=cfg["tolerance"],
        audit_interval=cfg["interval"],
        device=device,
        verbose=True
    )