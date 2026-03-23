"""
EOE 共享类型定义
================

本模块提取所有模块间共享的数据类、TypedDict、Protocol。
不要在此模块中import任何其他eoe模块。

使用 TYPE_CHECKING 来避免运行时循环依赖:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from .agent import Agent
        from .genome import OperatorGenome
        from .environment import Environment
"""

from enum import Enum, auto
from typing import Protocol, Dict, List, Tuple, Optional, Any, Sequence, Iterator
from dataclasses import dataclass, field
from collections import deque


# ============================================================
# 枚举类型 (不依赖其他eoe模块)
# ============================================================


class NodeType(Enum):
    """
    EOE v17.0 算子库 - 8 个创世粒子 + 扩展

    核心哲学: "只设计环境压力，不设计大脑结构"

    物理接口 (连接数字大脑和物理世界):
    - SENSOR: 通用传感器 (带 receptor_key + spatial_offset)
    - ACTUATOR: 通用执行器 (带 emitter_key + spatial_offset)

    Level 1 - 线性与信号原语:
    - ADD: 加法算子 (空间整合)
    - DELAY: 延迟算子 (时间整合)

    Level 2 - 非线性与逻辑:
    - THRESHOLD: 阈值算子 (逻辑判断)
    - OSCILLATOR: 节律发生器 (CPG基础)

    Level 3 - 乘性与动态调节 (v17.0新增):
    - MULTIPLY: 乘法算子 (交互项)
    - MODULATOR: 调制器 (门控/动态路由) ← 新增!
    - CONSTANT: 常数节点 (偏置)

    演化涌现:
    - MACRO: 宏算子
    - SUPERNODE: 超级节点
    """

    # ===== 物理接口 =====
    SENSOR = auto()  # 通用传感器 (升级版)
    ACTUATOR = auto()  # 通用执行器 (待升级)

    # ===== Level 1: 线性与信号原语 =====
    ADD = auto()  # 加法算子 (空间整合)
    DELAY = auto()  # 延迟算子 (时间整合)

    # ===== Level 2: 非线性与逻辑 =====
    THRESHOLD = auto()  # 阈值算子 (逻辑判断)
    OSCILLATOR = auto()  # 节律发生器 (CPG基础)

    # ===== Level 3: 乘性与动态调节 (v17.0 新增) =====
    MULTIPLY = auto()  # 乘法算子 (交互项)
    MODULATOR = auto()  # 调制器 (门控/动态路由)
    CONSTANT = auto()  # 常数节点

    # ===== 演化涌现 =====
    MACRO = auto()  # 宏算子 (保留兼容)
    SUPERNODE = auto()  # 超级节点 (保留兼容)

    # ============================================================
    # v0.0 统一场物理系统 - 传感器节点
    # 输入层：直接感知四个场的中心值与梯度
    # ============================================================
    # EPF 能量场感知 (3节点)
    SENSE_EPF_CENTER = auto()  # 能量场中心值 E(x,y)
    SENSE_EPF_GRAD_X = auto()  # 能量梯度 ∂E/∂x
    SENSE_EPF_GRAD_Y = auto()  # 能量梯度 ∂E/∂y

    # KIF 阻抗场感知 (3节点)
    SENSE_KIF_CENTER = auto()  # 阻抗场中心值 Z(x,y)
    SENSE_KIF_GRAD_X = auto()  # 阻抗梯度 ∂Z/∂x
    SENSE_KIF_GRAD_Y = auto()  # 阻抗梯度 ∂Z/∂y

    # ISF 压痕场感知 (3节点)
    SENSE_ISF_CENTER = auto()  # 压痕场中心值 S(x,y)
    SENSE_ISF_GRAD_X = auto()  # 压痕梯度 ∂S/∂x
    SENSE_ISF_GRAD_Y = auto()  # 压痕梯度 ∂S/∂y

    # ESF 应力场感知 (1节点)
    SENSE_ESF_VAL = auto()  # 应力值 σ(t)

    # 内部状态感知 (1节点)
    SENSE_INTERNAL_ENERGY = auto()  # 体内能量余额

    # ============================================================
    # v0.0 统一场物理系统 - 执行器节点
    # 输出层：严格绑定激活函数的物理致动器
    # ============================================================
    ACTUATOR_PERMEABILITY = auto()  # κ 渗透率 [0,1] - Sigmoid
    ACTUATOR_THRUST_X = auto()  # Fx 推力X [-1,1] - Tanh
    ACTUATOR_THRUST_Y = auto()  # Fy 推力Y [-1,1] - Tanh
    ACTUATOR_SIGNAL = auto()  # λ 信号强度 [0,1] - ReLU/Sigmoid
    ACTUATOR_DEFENSE = auto()  # S 防御刚性 [0,1] - Sigmoid

    # v16.0: 构成性执行器
    ACTUATOR_CONSTRUCT = auto()  # 建造: 消耗能量生成物质块
    ACTUATOR_DECONSTRUCT = auto()  # 分解: 破坏物质块回收能量

    # ============================================================
    # v5.x 预测与通信节点
    # ============================================================
    PREDICTOR = auto()  # 预测节点 (预测下一时刻传感器值)
    ENTITY_RADAR = auto()  # 实体雷达 (感知附近Agent)

    # ============================================================
    # v5.6 物理输出端口
    # ============================================================
    PORT_MOTION = auto()  # 运动端口 (速度+转向)
    PORT_REPAIR = auto()  # 修复端口
    PORT_OFFENSE = auto()  # 攻击端口
    PORT_DEFENSE = auto()  # 防御端口
    PORT_SIGNAL = auto()  # 信号端口

    # ============================================================
    # v7.0 扩展传感器
    # ============================================================
    LIGHT_SENSOR = auto()  # 光源传感器
    AGENT_RADAR_SENSOR = auto()  # Agent雷达传感器 (社会信号)
    GPS_SENSOR = auto()  # GPS传感器 (位置信息)


# ============================================================
# 数据类 (不依赖其他eoe模块)
# ============================================================


@dataclass
class Edge:
    """基因组的边（连接）"""

    source_id: int
    target_id: int
    weight: float = 0.0
    enabled: bool = True
    innovation_id: Optional[int] = None


@dataclass
class NodeState:
    """节点状态快照"""

    node_id: int
    node_type: NodeType
    activation: float
    constant_value: float = 0.0


@dataclass
class GenomeState:
    """基因组状态快照"""

    nodes: List[NodeState]
    edges: List[Edge]
    fitness: float = 0.0


@dataclass
class AgentPhenotype:
    """Agent表型（运行时状态）"""

    x: float
    y: float
    theta: float
    internal_energy: float
    is_alive: bool
    age: float
    fitness: float


# ============================================================
# Protocol 定义 (用于类型注解，避免循环依赖)
# ============================================================


class AgentProtocol(Protocol):
    """Agent对象的最小接口协议"""

    id: int
    x: float
    y: float
    theta: float
    fitness: float
    internal_energy: float
    is_alive: bool
    age: float
    genome: "OperatorGenomeProtocol"

    def get_position(self) -> Tuple[float, float]: ...
    def is_alive(self) -> bool: ...


class OperatorGenomeProtocol(Protocol):
    """OperatorGenome对象的最小接口协议"""

    nodes: Dict[int, Any]
    edges: List[Edge]

    def forward(self, inputs: Any) -> Any: ...
    def get_topological_order(self) -> List[int]: ...


class EnvironmentProtocol(Protocol):
    """Environment对象的最小接口协议"""

    width: float
    height: float
    agents: List[Any]

    def step(self) -> None: ...
    def get_agent_sensors(self, agent: Any) -> Any: ...
    def apply_actuators(self, agent: Any, outputs: Any) -> None: ...


class NodeProtocol(Protocol):
    """Node对象的最小接口协议"""

    node_id: int
    node_type: NodeType
    activation: float

    def __repr__(self) -> str: ...


# ============================================================
# 类型别名
# ============================================================

# 节点ID
NodeId = int

# 边列表
EdgeList = List[Edge]

# 节点字典
NodeDict = Dict[int, NodeProtocol]

# 位置坐标
Position = Tuple[float, float]

# 传感器输入
SensorInput = List[float]

# 执行器输出
ActuatorOutput = List[float]


# ============================================================
# 迭代器类型
# ============================================================


class AgentIterator(Iterator[AgentProtocol]):
    """Agent迭代器协议"""

    pass


class NodeIterator(Iterator[NodeProtocol]):
    """Node迭代器协议"""

    pass
