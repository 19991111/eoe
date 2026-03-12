"""
v13.0 环境应力/温度场 (Environmental Stress Field, ESF)

"时间的节律" - 外部非线性干扰

物理模型:
- 空间异构: σ(x,y,t) = σ_base + wave(t) × noise(x,y)
- 场间耦合: 应力调制能量衰减、阻抗基准、代谢率
- 时间导数: dσ/dt (应力变化率)

核心思想:
- 全局波动 + 低频噪声 = 微气候
- 应力作为物理常数调制器
- Agent感知应力变化率建立"时间感"

依赖:
- Agent: 智能体
- Environment: 环境
"""

from typing import Dict, Tuple
import numpy as np


class Stressor:
    """单个应力源"""
    
    def __init__(
        self,
        name: str,
        base: float = 0.0,
        amplitude: float = 0.3,
        period: int = 200,
        phase: float = 0.0,
        noise_scale: float = 0.5  # 空间异构程度
    ):
        self.name = name
        self.base = base
        self.amplitude = amplitude
        self.period = period
        self.phase = phase
        self.noise_scale = noise_scale
        
    def compute(self, x: float, y: float, time: int, noise_field: np.ndarray = None) -> float:
        """计算应力值"""
        # 时间波动
        wave = self.amplitude * np.sin(2 * np.pi * time / self.period + self.phase)
        
        # 空间异构 (如果提供了噪声场)
        spatial = 1.0
        if noise_field is not None:
            gx = int(x) % noise_field.shape[0]
            gy = int(y) % noise_field.shape[1]
            spatial = 1.0 + self.noise_scale * (noise_field[gx, gy] - 0.5)
        
        return self.base + wave * spatial


class StressField:
    """
    环境应力场 - 空间异构时间波动
    
    物理本质:
    - σ(x,y,t): 随时间漂移的空间场
    - 创造"微气候"区域
    - 调制其他场的物理参数
    
    实现:
    - 多个应力源 (温度、代谢、扩散、阻抗)
    - 空间噪声场用于异构化
    - 时间导数计算
    """
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 2.0,  # 低分辨率足矣
        # 温度应力
        temp_amplitude: float = 0.5,
        temp_period: int = 200,
        # 代谢压力
        metabolic_amplitude: float = 0.3,
        metabolic_period: int = 150,
        # 扩散波动
        diffusion_amplitude: float = 0.1,
        diffusion_period: int = 100,
        # 阻抗波动
        impedance_amplitude: float = 0.2,
        impedance_period: int = 180
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # 生成空间噪声场 (低频，用于创造微气候)
        self.noise_field = self._generate_climate_noise()
        
        # 初始化各应力源
        self.stressors = {
            'temperature': Stressor(
                'temperature',
                base=0.0,
                amplitude=temp_amplitude,
                period=temp_period,
                phase=0.0,
                noise_scale=0.6  # 强空间异构
            ),
            'metabolic': Stressor(
                'metabolic',
                base=1.0,  # 基准代谢倍率
                amplitude=metabolic_amplitude,
                period=metabolic_period,
                phase=np.pi / 4,
                noise_scale=0.3
            ),
            'diffusion': Stressor(
                'diffusion',
                base=1.0,  # 基准扩散率
                amplitude=diffusion_amplitude,
                period=diffusion_period,
                phase=np.pi / 2,
                noise_scale=0.4
            ),
            'impedance': Stressor(
                'impedance',
                base=1.0,  # 基准阻抗乘数
                amplitude=impedance_amplitude,
                period=impedance_period,
                phase=np.pi,
                noise_scale=0.5
            )
        }
        
        # 记录上一帧应力用于计算导数
        self._prev_stress: Dict[str, float] = {}
        
    def _generate_climate_noise(self) -> np.ndarray:
        """生成低频气候噪声"""
        # 简化的多倍频正弦
        w, h = self.grid_width, self.grid_height
        x = np.linspace(0, 4 * np.pi, w)
        y = np.linspace(0, 4 * np.pi, h)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        
        # 低频叠加
        noise = np.sin(xx) * np.cos(yy) * 0.5
        noise += np.sin(xx * 0.5) * np.cos(yy * 0.5) * 0.3
        noise += np.sin(xx * 0.25) * np.cos(yy * 0.25) * 0.2
        
        # 归一化到 [0, 1]
        return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        
    def sample(self, x: float, y: float, time: int, stress_type: str = 'all') -> float:
        """
        采样指定位置的应力
        
        参数:
            x, y: 世界坐标
            time: 当前时间步
            stress_type: 'all' | 'temperature' | 'metabolic' | 'diffusion' | 'impedance'
        
        返回:
            应力值 (如果是'all'，返回总应力)
        """
        if stress_type == 'all':
            # 返回所有应力的加权和
            total = 0.0
            for name, stressor in self.stressors.items():
                total += stressor.compute(x, y, time, self.noise_field)
            return total / len(self.stressors)
        else:
            return self.stressors[stress_type].compute(x, y, time, self.noise_field)
            
    def sample_all(self, x: float, y: float, time: int) -> Dict[str, float]:
        """采样所有应力源"""
        result = {}
        for name, stressor in self.stressors.items():
            result[name] = stressor.compute(x, y, time, self.noise_field)
        return result
        
    def compute_derivative(self, agent, env) -> Dict[str, float]:
        """
        计算应力时间导数 dσ/dt
        
        用于Agent感知"应力正在上升/下降"
        """
        time = env.step_count if env else 0
        
        derivatives = {}
        for name in self.stressors:
            current = self.sample(agent.x, agent.y, time, name)
            prev = self._prev_stress.get(name, current)
            derivatives[name] = current - prev
            
            # 更新记录
            self._prev_stress[name] = current
            
        return derivatives
        
    def apply_coupling(self, env, time: int):
        """
        将应力应用到环境 (场间耦合)
        
        影响:
        - 能量场衰减率
        - 阻抗场基准值
        - 代谢倍率
        """
        # 获取当前Agent位置的应力 (使用第一个Agent或平均)
        if env.agents:
            sample_agent = env.agents[0]
            stress_vals = self.sample_all(sample_agent.x, sample_agent.y, time)
        else:
            stress_vals = {k: s.base for k, s in self.stressors.items()}
        
        # 1. 应力 → 能量熵增
        if env.energy_field:
            diff_stress = stress_vals.get('diffusion', 1.0)
            env.energy_field.diffusion_rate = env.base_diffusion_rate * diff_stress
            env.energy_field.decay_rate = env.base_decay_rate * (1 + stress_vals.get('temperature', 0) * 0.5)
        
        # 2. 应力 → 阻抗硬化
        if env.impedance_field:
            imp_stress = stress_vals.get('impedance', 1.0)
            env.impedance_field.field *= imp_stress
            # 限制最大增幅
            np.clip(env.impedance_field.field, 0, 1000, out=env.impedance_field.field)
        
        # 3. 应力 → 代谢倍率
        met_stress = stress_vals.get('metabolic', 1.0)
        env.metabolic_multiplier *= met_stress
        
    def get_heatmap_data(self) -> np.ndarray:
        """获取热力图数据"""
        return self.noise_field.T


# ============================================================
# StressLaw - 应力场物理法则
# ============================================================

class StressLaw:
    """
    应力场物理法则
    
    负责:
    - 采样应力传感器
    - 计算时间导数
    - 更新Agent的应力感知
    """
    
    def apply_to_agent(self, agent, env) -> dict:
        """
        将应力感知应用到Agent
        """
        if not hasattr(env, 'stress_field') or env.stress_field is None:
            return {'stress': 0.0, 'derivative': 0.0}
            
        time = env.step_count
        
        # 采样当前应力
        stress = env.stress_field.sample(agent.x, agent.y, time, 'all')
        
        # 计算时间导数
        derivatives = env.stress_field.compute_derivative(agent, env)
        total_derivative = sum(derivatives.values()) / len(derivatives)
        
        # 记录到Agent
        agent.current_stress = stress
        agent.stress_derivative = total_derivative
        agent._prev_stress = stress
        
        return {
            'stress': stress,
            'derivative': total_derivative,
            'details': derivatives
        }


def create_stress_field(
    width: float = 100.0,
    height: float = 100.0,
    season_length: int = 200
) -> StressField:
    """创建应力场 (季节长度可配置)"""
    return StressField(
        width=width, height=height,
        temp_period=season_length,
        metabolic_period=int(season_length * 0.75),
        diffusion_period=int(season_length * 0.5),
        impedance_period=int(season_length * 0.9)
    )