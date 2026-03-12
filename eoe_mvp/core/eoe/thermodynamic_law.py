"""
v13.0 热力学物理法则 (ThermodynamicLaw)

能量交换物理法则 - 将能量从"奖励数值"还原为"物理流体"

核心公式:
- 能量交换: E_exchange = κ × (E_field - E_agent)
- 渗透膜代价: E_cost = κ × permeability_cost
- 移动做功: E_move = c × |F|²
- 废热排放: E_waste = E_metabolic × waste_heat_ratio

物理意义:
- κ = 0 (封闭): 不与环境交换能量
- κ = 1 (开放): 完全与环境能量场平衡
- 在高能区开启 κ > 0 → 能量流入
- 在低能区开启 κ > 0 → 能量倒灌流失

依赖:
- EnergyField: 能量场
- Agent: 智能体
"""

from typing import Tuple
import numpy as np


class ThermodynamicLaw:
    """
    v13.0 能量交换物理法则
    
    物理本质:
    - 能量守恒: 能量不会凭空消失，而是转化
    - 渗透压平衡: 能量从高浓度流向低浓度
    - 做功能耗: 移动输出功，消耗能量
    - 废热循环: 代谢产物以低价值能量排入环境
    """
    
    def __init__(
        self,
        permeability_cost: float = 0.01,
        waste_heat_ratio: float = 0.3,
        move_cost_coeff: float = 0.1,
        interaction_range: float = 5.0,
        theft_efficiency: float = 0.5
    ):
        """
        初始化热力学法则
        
        参数:
            permeability_cost: 维持渗透膜的代价 (每帧)
            waste_heat_ratio: 代谢转化为废热的比例 [0,1]
            move_cost_coeff: 移动做功能量系数 c
            interaction_range: Agent间交互距离
            theft_efficiency: 能量窃取效率
        """
        self.permeability_cost = permeability_cost
        self.waste_heat_ratio = waste_heat_ratio
        self.move_cost_coeff = move_cost_coeff
        self.interaction_range = interaction_range
        self.theft_efficiency = theft_efficiency
        
    def compute_energy_exchange(
        self,
        agent,
        field
    ) -> float:
        """
        计算 Agent 与能量场的能量交换
        
        公式: E_exchange = κ × (E_field - E_agent) - membrane_cost
        
        参数:
            agent: Agent实例，需有 permeability, internal_energy 属性
            field: EnergyField实例
        
        返回:
            能量变化量 (正值=流入, 负值=流出)
        """
        if field is None:
            return 0.0
            
        # 采样所在位置的环境能量
        field_energy = field.sample(agent.x, agent.y)
        agent.field_energy = field_energy  # 记录供传感器使用
        
        # 能量交换公式: E = κ × (E_field - E_agent)
        kappa = getattr(agent, 'permeability', 0.0)
        exchange = kappa * (field_energy - agent.internal_energy)
        
        # 渗透膜维持代价
        membrane_cost = kappa * self.permeability_cost
        
        return exchange - membrane_cost
        
    def compute_move_cost(
        self,
        left_force: float,
        right_force: float,
        agent_energy: float = 0.0
    ) -> float:
        """
        计算移动做功的能量消耗
        
        公式: E = c × |F|² × (1 + mass_penalty)
        
        物理意义:
        - 速度越快，阻力越大
        - 消耗呈非线性增长
        - 高能量(肥胖)会增加移动能耗 (质量惩罚)
        
        参数:
            left_force, right_force: 左右推进器输出力
            agent_energy: Agent内部能量 (用于计算质量惩罚)
        
        返回:
            能量消耗量
        """
        # 合力
        force = (abs(left_force) + abs(right_force)) / 2.0
        
        # ============================================================
        # v13.0 质量-能耗惩罚 (Mass-Energy Penalty)
        # 体内能量越高，移动消耗越大
        # 演化博弈: 必须卸载能量才能保持机动性
        # ============================================================
        mass_penalty = 1.0 + (agent_energy * 0.01)  # 每100能量 +100%能耗
        
        # 非线性消耗 (平方关系) × 质量惩罚
        return self.move_cost_coeff * (force ** 2) * mass_penalty
    
    def compute_signal_cost(
        self,
        signal_intensity: float
    ) -> float:
        """
        v13.0 计算信息场信号释放能耗
        
        公式: E = λ² × signal_cost_coeff (非线性)
        
        物理意义:
        - 信号强度越高，消耗呈平方增长
        - 与移动推力解绑，允许"隐身策略"
        
        参数:
            signal_intensity: λ ∈ [0, 1]
        
        返回:
            能量消耗量
        """
        # 非线性代价 (平方关系)
        signal_cost_coeff = 0.02
        return (signal_intensity ** 2) * signal_cost_coeff
    
    def compute_defense_reduction(
        self,
        attack_amount: float,
        defense_rigidity: float
    ) -> float:
        """
        v13.0 计算防御刚性对能量窃取的减伤
        
        公式: E_lost = attack × (1 - S)
        
        参数:
            attack_amount: 攻击方试图窃取的能量
            defense_rigidity: S ∈ [0, 1]
        
        返回:
            实际损失的能量 (防御后的值)
        """
        # S = 1: 完全防御，无损失
        # S = 0: 无防御，全额损失
        return attack_amount * (1.0 - defense_rigidity)
        
    def compute_waste_heat(
        self,
        metabolic_cost: float,
        move_cost: float
    ) -> float:
        """
        计算废热排放量
        
        代谢消耗的一部分以低价值能量形式排入环境
        
        参数:
            metabolic_cost: 大脑运行代谢消耗
            move_cost: 移动做功消耗
        
        返回:
            废热排放量
        """
        total_cost = metabolic_cost + move_cost
        return total_cost * self.waste_heat_ratio
        
    def compute_agent_interaction(
        self,
        agent_a,
        agent_b,
        width: float,
        height: float
    ) -> Tuple[float, float]:
        """
        计算两个 Agent 之间的能量流动 (掠夺机制)
        
        物理本质:
        - 渗透压竞争: 高能量方的能量自发流向低能量方
        - 取决于双方的渗透率差异
        - 防御刚性 S 可降低损失
        
        参数:
            agent_a, agent_b: 两个Agent实例
            width, height: 环境尺寸 (用于环形世界距离)
        
        返回:
            (energy_to_a, energy_to_b) 能量变化量
        """
        # 环形世界距离计算
        dx = agent_b.x - agent_a.x
        dy = agent_b.y - agent_a.y
        dx = dx - width * np.floor(dx / width + 0.5)
        dy = dy - height * np.floor(dy / height + 0.5)
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance > self.interaction_range:
            return (0.0, 0.0)
            
        # 能量差
        energy_diff = agent_a.internal_energy - agent_b.internal_energy
        
        # 渗透率差决定流动方向
        kappa_b = getattr(agent_b, 'permeability', 0.0)
        
        # 如果 B 的渗透率高于 A，且 B 的能量低于 A，则发生窃取
        if kappa_b > 0 and energy_diff > 0:
            # 窃取量 = 渗透率 × 能量差 × 效率
            steal_amount = kappa_b * energy_diff * self.theft_efficiency
            
            # A 的防御刚性 S 减伤
            defense_rigidity_a = getattr(agent_a, 'defense_rigidity', 0.0)
            steal_amount = self.compute_defense_reduction(steal_amount, defense_rigidity_a)
            
            # A 失去能量，B 获得能量
            return (-steal_amount, steal_amount)
            
        return (0.0, 0.0)
        
    def apply_to_agent(
        self,
        agent,
        field,
        left_force: float,
        right_force: float,
        metabolic_cost: float
    ) -> dict:
        """
        将热力学法则应用到单个 Agent
        
        参数:
            agent: Agent实例
            field: EnergyField实例 (可为None)
            left_force, right_force: 左右推进器输出
            metabolic_cost: 大脑运行代谢消耗
        
        返回:
            详细的能量变化分解字典
        """
        # 1. 能量场交换
        exchange = self.compute_energy_exchange(agent, field)
        
        # 2. 移动做功
        move_cost = self.compute_move_cost(left_force, right_force)
        
        # 3. 废热排放 (到环境)
        waste_heat = self.compute_waste_heat(metabolic_cost, move_cost)
        
        # 总能量变化
        total_delta = exchange - move_cost - metabolic_cost
        
        # 应用到 Agent
        agent.internal_energy += total_delta
        
        # 记录废热到环境场
        if field is not None and waste_heat > 0:
            gx = int(agent.x / field.resolution) % field.grid_width
            gy = int(agent.y / field.resolution) % field.grid_height
            field.field[gx, gy] += waste_heat
            
        # 更新统计
        if hasattr(agent, 'energy_spent'):
            agent.energy_spent += (move_cost + metabolic_cost)
        if hasattr(agent, 'energy_wasted'):
            agent.energy_wasted += waste_heat
        
        return {
            'exchange': exchange,
            'move_cost': move_cost,
            'metabolic_cost': metabolic_cost,
            'waste_heat': waste_heat,
            'total_delta': total_delta
        }
    
    def apply_v13_unified(
        self,
        agent,
        env
    ) -> dict:
        """
        v13.0 统一能量结算 - 所有能量得失通过场耦合计算
        
        四个物理参数:
            κ (permeability): 渗透率 → EPF能量交换
            F (thrust): 推力矢量 → KIF移动能耗
            λ (signal): 信号强度 → ISF信息释放能耗
            S (defense): 防御刚性 → 能量窃取减伤
        
        严禁手动修改 agent.energy，所有能量变化必须通过此方法
        """
        # === 1. EPF: 能量场交换 (通过 κ) ===
        exchange = 0.0
        if env.energy_field_enabled and env.energy_field:
            kappa = agent.permeability  # κ ∈ [0, 1]
            field_energy = env.energy_field.sample(agent.x, agent.y)
            agent.field_energy = field_energy
            
            # 热力学公式: ΔE = κ × (E_field - E_agent)
            # - 高能区: 能量流入 (进食)
            # - 低能区: 能量流出 (排泄 - 物理涌现!)
            exchange = kappa * (field_energy - agent.internal_energy)
            
            # 渗透膜维持代价
            membrane_cost = kappa * self.permeability_cost
            exchange -= membrane_cost
        
        # === 2. KIF: 移动能耗 (通过 F) ===
        move_cost = 0.0
        if env.impedance_field_enabled and env.kinetic_impedance_law:
            fx, fy = agent.thrust_vector  # F ∈ [-1, 1]²
            force_magnitude = np.sqrt(fx*fx + fy*fy)
            # 获取当前位置阻抗
            impedance = env.impedance_field.sample(agent.x, agent.y)
            
            # ============================================================
            # v13.0 质量惩罚: 高能量 = 高移动能耗
            # mass_penalty = 1 + E_agent * 0.01
            # ============================================================
            mass_penalty = 1.0 + (agent.internal_energy * 0.01)
            
            # 能耗 = c × |F|² × log(1+Z) × mass_penalty
            move_cost = self.move_cost_coeff * (force_magnitude ** 2) * np.log(1 + impedance) * mass_penalty
        
        # === 3. ISF: 信息场信号能耗 (通过 λ - 独立于移动!) ===
        signal_cost = 0.0
        if env.stigmergy_field_enabled and agent.signal_intensity > 0:
            signal_cost = self.compute_signal_cost(agent.signal_intensity)
            
            # 注入信号到压痕场
            if env.stigmergy_field:
                env.stigmergy_field.deposit(
                    agent.x, agent.y,
                    amount=agent.signal_intensity,
                    agent_energy=agent.internal_energy
                )
        
        # === 4. ESF: 代谢调制 (已由 env.step 处理) ===
        # stress_metabolic_multiplier 已在 environment.py 中应用
        
        # === 5. 基础代谢 ===
        metabolic_cost = 0.1  # 基础代谢
        
        # 总能量变化
        total_delta = exchange - move_cost - signal_cost - metabolic_cost
        
        # 应用到 Agent (严禁手动修改!)
        agent.internal_energy += total_delta
        
        # 更新统计
        if hasattr(agent, 'energy_spent'):
            agent.energy_spent += (move_cost + signal_cost + metabolic_cost)
        
        return {
            'exchange': exchange,         # EPF 能量交换
            'move_cost': move_cost,       # KIF 移动能耗
            'signal_cost': signal_cost,   # ISF 信号能耗
            'metabolic_cost': metabolic_cost,  # 基础代谢
            'total_delta': total_delta    # 总变化
        }


# ============================================================
# 便捷函数
# ============================================================

def create_thermodynamic_law(
    cold_start_mode: bool = False
) -> ThermodynamicLaw:
    """
    创建热力学法则实例
    
    参数:
        cold_start_mode: 是否启用冷启动模式 (更宽容的参数)
    
    冷启动模式:
        - 更高的初始能量注入
        - 更低的渗透膜代价
        - 给第一代"乱动"的智能体留出容错空间
    """
    if cold_start_mode:
        return ThermodynamicLaw(
            permeability_cost=0.005,   # 降低50%
            waste_heat_ratio=0.5,      # 提高废热回收
            move_cost_coeff=0.05,      # 降低移动代价
            interaction_range=3.0,     # 缩小交互范围
            theft_efficiency=0.3       # 降低窃取效率
        )
    else:
        return ThermodynamicLaw()