"""
EOE (Evolution of Emergence) v13.0
===================================
统一场物理的开放式演化引擎

模块:
    fields   - 物理场 (EPF, KIF, ISF, ESF)
    batch    - 批量GPU系统
    env      - 环境 (CPU/GPU)
    
核心类:
    Simulation - 统一仿真入口
    AgentState - Agent状态容器
    Field      - 场基类
"""

# 版本
__version__ = '13.0.0'

# 批量系统 (推荐使用)
from .batch import Simulation, AgentState, BatchedAgents, EnvironmentGPU
from .batch import ThermodynamicLaw, ThermoStats, SimState

# 物理场
from .fields import Field, EnergyField, ImpedanceField, StigmergyField
from .fields import DEFAULT_FIELD_CONFIG

# 保留旧版API (向后兼容)
from .agent import Agent
from .genome import OperatorGenome
from .node import NodeType
from .population import Population

__all__ = [
    # 版本
    '__version__',
    
    # 批量系统 (新API)
    'Simulation',
    'AgentState',
    'BatchedAgents', 
    'EnvironmentGPU',
    'ThermodynamicLaw',
    'ThermoStats',
    'SimState',
    
    # 物理场
    'Field',
    'EnergyField',
    'ImpedanceField',
    'StigmergyField',
    'DEFAULT_FIELD_CONFIG',
    
    # 旧版API (兼容)
    'Agent',
    'OperatorGenome', 
    'NodeType',
    'Population',
]


def quick_run(n_agents: int = 100, steps: int = 500, device: str = 'cuda:0'):
    """
    快速运行仿真
    
    Args:
        n_agents: Agent数量
        steps: 步数
        device: 设备
    """
    sim = Simulation(n_agents=n_agents, lifespan=steps, device=device)
    return sim.run()


# 便捷函数
run = quick_run