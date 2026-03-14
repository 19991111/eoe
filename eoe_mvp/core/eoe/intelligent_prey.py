"""
智能猎物 (Intelligent Prey) - Red Queen Dynamics
=================================================
能量源会主动躲避靠近的Agent，迫使Agent演化出更复杂的预测能力

核心机制:
1. 猎物感知: 检测一定范围内的Agent
2. 逃跑决策: 当检测到直线靠近的Agent时，启动逃跑模式
3. Z字形逃跑: 周期性改变逃跑方向，使简单预测失效
4. 疲劳恢复: 逃跑后会疲惫，短时间内不再逃跑

设计原理:
- 真正的红皇后假说: 猎物的"进化"(策略)与捕食者(Agent)协同演化
- Z字形模式需要Agent计算二阶导数(加速度)
- 简单2节点预测无法捕捉这种动态
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class IntelligentPreyConfig:
    """智能猎物配置"""
    # 感知参数
    detection_range: float = 25.0      # 感知Agent的范围
    escape_trigger_distance: float = 15.0  # 触发逃跑的距离
    
    # 逃跑参数
    escape_speed: float = 2.0          # 逃跑速度
    zigzag_period: int = 8              # Z字形周期 (步)
    zigzag_amplitude: float = 0.5       # Z字形幅度 (弧度)
    
    # 疲劳参数
    fatigue_duration: int = 30          # 逃跑后疲劳步数
    energy_cost_per_step: float = 0.5   # 每步逃跑能量消耗
    
    # 智能程度
    predict_linear: bool = True         # 能预测直线靠近
    react_delay: int = 2                # 反应延迟 (步)


class EnergySourcePrey:
    """
    智能能量源 (猎物)
    
    每个能量源都是一个"有智能的猎物"，会:
    1. 感知附近的Agent
    2. 判断是否直线靠近
    3. 决定逃跑方向和时机
    """
    
    def __init__(
        self,
        config: IntelligentPreyConfig,
        position: torch.Tensor,  # [2] x, y
        device: str = 'cpu'
    ):
        self.config = config
        self.device = device
        self.position = position.clone()  # [2]
        
        # 状态
        self.is_escaping = False
        self.escape_timer = 0              # 逃跑计时器
        self.fatigue_timer = 0             # 疲劳计时器
        self.zigzag_phase = 0              # Z字形相位
        self.escape_direction = torch.zeros(2, device=device)  # 逃跑方向
        self.last_agent_position = None    # 上次看到的Agent位置
        
    def perceive(
        self,
        agent_positions: torch.Tensor,    # [N, 2]
        agent_velocities: torch.Tensor,   # [N, 2]
        alive_mask: Optional[torch.Tensor] = None  # [N]
    ) -> dict:
        """
        感知附近的Agent
        
        Returns:
            dict: {
                'nearest_distance': float,
                'nearest_direction': [2],
                'is_approaching': bool,
                'approach_velocity': float,
            }
        """
        if agent_positions.shape[0] == 0:
            return {
                'nearest_distance': float('inf'),
                'nearest_direction': torch.zeros(2, device=self.device),
                'is_approaching': False,
                'approach_velocity': 0.0,
            }
        
        # 计算到所有Agent的距离
        diff = agent_positions - self.position.unsqueeze(0)  # [N, 2]
        distances = torch.norm(diff, dim=1)  # [N]
        
        # 过滤范围外的Agent
        cfg = self.config
        in_range = distances < cfg.detection_range
        
        if not in_range.any():
            return {
                'nearest_distance': float('inf'),
                'nearest_direction': torch.zeros(2, device=self.device),
                'is_approaching': False,
                'approach_velocity': 0.0,
            }
        
        # 找最近的
        valid_distances = distances.clone()
        valid_distances[~in_range] = float('inf')
        nearest_idx = torch.argmin(valid_distances)
        
        nearest_pos = agent_positions[nearest_idx]
        nearest_vel = agent_velocities[nearest_idx]
        nearest_dist = distances[nearest_idx]
        
        # 到最近Agent的方向
        direction = diff[nearest_idx] / (nearest_dist + 1e-6)  # [2] 归一化
        
        # 判断是否在靠近: 速度方向与方向向量的点积
        # 确保都是1D张量
        nearest_vel_2d = nearest_vel.reshape(-1) if nearest_vel.numel() == 2 else nearest_vel
        direction_2d = direction.reshape(-1) if direction.numel() == 2 else direction
        
        if nearest_vel_2d.dim() == 0 or direction_2d.dim() == 0:
            velocity_toward = 0.0
        else:
            velocity_toward = torch.dot(nearest_vel_2d, direction_2d).item()
        
        is_approaching = velocity_toward > 0 and nearest_dist < cfg.escape_trigger_distance
        
        return {
            'nearest_distance': nearest_dist.item(),
            'nearest_direction': direction,
            'is_approaching': is_approaching,
            'approach_velocity': velocity_toward,
        }
    
    def update(
        self,
        agent_positions: torch.Tensor,
        agent_velocities: torch.Tensor,
        bounds: Tuple[float, float, float, float],  # x_min, x_max, y_min, y_max
        dt: float = 1.0
    ) -> dict:
        """
        更新猎物状态
        
        Returns:
            dict: {
                'escaped': bool,
                'new_position': [2],
                'energy_cost': float,
            }
        """
        cfg = self.config
        x_min, x_max, y_min, y_max = bounds
        
        # 感知环境
        perception = self.perceive(agent_positions, agent_velocities)
        
        energy_cost = 0.0
        escaped = False
        
        # 状态机
        if self.fatigue_timer > 0:
            # 疲劳中，不逃跑
            self.fatigue_timer -= 1
            self.is_escaping = False
            
        elif self.is_escaping:
            # 正在逃跑
            self.escape_timer -= 1
            
            if self.escape_timer <= 0:
                # 逃跑结束，进入疲劳
                self.is_escaping = False
                self.fatigue_timer = cfg.fatigue_duration
            else:
                # 执行Z字形逃跑
                escaped = True
                energy_cost = cfg.energy_cost_per_step * dt
                
                # 计算Z字形方向
                self.zigzag_phase += 1
                zigzag_angle = torch.sin(
                    torch.tensor(self.zigzag_phase * 2 * torch.pi / cfg.zigzag_period)
                ) * cfg.zigzag_amplitude
                
                # 旋转基础逃跑方向
                cos_a = torch.cos(zigzag_angle)
                sin_a = torch.sin(zigzag_angle)
                
                # 旋转
                escaped_x = self.escape_direction[0] * cos_a - self.escape_direction[1] * sin_a
                escaped_y = self.escape_direction[0] * sin_a + self.escape_direction[1] * cos_a
                
                # 移动
                move = torch.stack([escaped_x, escaped_y]) * cfg.escape_speed * dt
                self.position += move
                
                # 边界反弹
                self.position[0] = torch.clamp(self.position[0], x_min, x_max)
                self.position[1] = torch.clamp(self.position[1], y_min, y_max)
        
        elif perception['is_approaching']:
            # 触发逃跑!
            self.is_escaping = True
            self.escape_timer = cfg.zigzag_period * 3  # 逃跑3个周期
            self.zigzag_phase = 0
            
            # 逃跑方向: 远离Agent + 随机扰动
            away_direction = -perception['nearest_direction']
            random_offset = torch.randn(2, device=self.device) * 0.2
            self.escape_direction = (away_direction + random_offset)
            self.escape_direction = self.escape_direction / (torch.norm(self.escape_direction) + 1e-6)
        
        return {
            'escaped': escaped,
            'new_position': self.position.clone(),
            'energy_cost': energy_cost,
            'is_escaping': self.is_escaping,
            'perception': perception,
        }


class IntelligentPreySystem:
    """
    智能猎物系统 - 管理所有能量源的智能行为
    """
    
    def __init__(
        self,
        config: IntelligentPreyConfig,
        source_positions: torch.Tensor,  # [N, 2]
        device: str = 'cpu'
    ):
        self.config = config
        self.device = device
        self.n_sources = source_positions.shape[0]
        
        # 为每个能量源创建智能猎物
        self.preys = [
            EnergySourcePrey(config, source_positions[i], device)
            for i in range(self.n_sources)
        ]
        
    def update(
        self,
        agent_positions: torch.Tensor,
        agent_velocities: torch.Tensor,
        bounds: Tuple[float, float, float, float],
        dt: float = 1.0
    ) -> dict:
        """更新所有猎物"""
        results = []
        
        for prey in self.preys:
            result = prey.update(agent_positions, agent_velocities, bounds, dt)
            results.append(result)
            
        return {
            'results': results,
            'n_escaping': sum(1 for r in results if r['escaped']),
        }
    
    def get_positions(self) -> torch.Tensor:
        """获取所有猎物的当前位置"""
        return torch.stack([prey.position for prey in self.preys])
    
    def apply_to_energy_field(self, energy_field) -> None:
        """将智能位置应用到能量场"""
        positions = self.get_positions()
        for i in range(self.n_sources):
            energy_field.sources[i, 0] = positions[i, 0]
            energy_field.sources[i, 1] = positions[i, 1]


# ============================================================================
# 集成到环境场的适配器
# ============================================================================

class IntelligentPreyAdapter:
    """
    将智能猎物行为适配到现有EnergyFieldGPU
    """
    
    def __init__(
        self,
        energy_field,
        config: Optional[IntelligentPreyConfig] = None
    ):
        self.energy_field = energy_field
        self.config = config or IntelligentPreyConfig()
        
        # 从能量场初始化智能猎物
        source_positions = energy_field.sources[:, :2].clone()
        self.prey_system = IntelligentPreySystem(
            self.config,
            source_positions,
            energy_field.device
        )
        
    def update(
        self,
        agent_positions: torch.Tensor,
        agent_velocities: torch.Tensor,
        dt: float = 1.0
    ) -> dict:
        """更新智能猎物并应用到能量场"""
        bounds = (
            0.0, self.energy_field.width,
            0.0, self.energy_field.height
        )
        
        # 更新智能猎物
        result = self.prey_system.update(
            agent_positions,
            agent_velocities,
            bounds,
            dt
        )
        
        # 应用新位置到能量场
        self.prey_system.apply_to_energy_field(self.energy_field)
        
        return result


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')
    
    # 测试
    config = IntelligentPreyConfig()
    
    # 模拟3个能量源
    positions = torch.tensor([[10.0, 10.0], [50.0, 50.0], [80.0, 20.0]])
    prey_system = IntelligentPreySystem(config, positions, 'cpu')
    
    # 模拟1个Agent从不同方向靠近
    agent_pos = torch.tensor([[15.0, 12.0]])  # 靠近第一个能量源
    agent_vel = torch.tensor([[1.0, 0.5]])    # 速度朝向能量源
    
    print("=" * 50)
    print("智能猎物测试")
    print("=" * 50)
    
    bounds = (0.0, 100.0, 0.0, 100.0)
    
    for step in range(20):
        result = prey_system.update(agent_pos, agent_vel, bounds, dt=1.0)
        
        pos = prey_system.get_positions()
        print(f"步{step+1}: 能耗={result['results'][0]['energy_cost']:.2f}, "
              f"位置=({pos[0,0]:.1f}, {pos[0,1]:.1f})")
        
        # Agent继续靠近
        agent_pos += agent_vel * 1.0
    
    print(f"\n最终位置: {prey_system.get_positions()[0].tolist()}")