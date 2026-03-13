from enum import Enum, auto
from typing import List, Dict, Optional, Any
from collections import deque
import numpy as np

class NodeType(Enum):
    """
    节点类型枚举 (EOE v4.0 - 功能器官版本)
    
    - SENSOR: 传感器节点，接收环境输入
    - ACTUATOR: 执行器节点，输出到物理世界
    - ADD: 加法算子
    - MULTIPLY: 乘法算子
    - CONSTANT: 常数节点
    - DELAY: 延迟算子，带可配置步长 (v4.0: 多尺度)
    - THRESHOLD: 阈值算子
    - PREDICTOR: 预测节点
    - ENTITY_RADAR: 社会传感器
    - MACRO: 宏算子 (封装子图)
    - SUPERNODE: 超级节点，冻结后的功能器官 (v4.0)
    - BAIT_SENSOR: 诱饵传感器，检测诱饵 (v4.0)
    """
    SENSOR = auto()
    ACTUATOR = auto()
    ADD = auto()
    MULTIPLY = auto()
    CONSTANT = auto()
    DELAY = auto()          # 多尺度时间动力学 (v4.0)
    THRESHOLD = auto()
    PREDICTOR = auto()
    ENTITY_RADAR = auto()
    MACRO = auto()          # 宏算子
    SUPERNODE = auto()      # 超级节点 - 冻结的功能器官 (v4.0)
    BAIT_SENSOR = auto()    # 诱饵传感器 (v4.0)
    
    # ============================================================
    # v6.0 GAIA 物理输出端口 (纯基础算子驱动)
    # 严禁添加逻辑节点！所有行为通过基础算子控制这些端口
    # ============================================================
    PORT_MOTION = auto()    # 基础移动 [速度, 转向]
    PORT_OFFENSE = auto()   # 攻击强度 (捕食能力)
    PORT_DEFENSE = auto()   # 防御硬度 (减少被掠夺)
    PORT_REPAIR = auto()    # 修复 (能量→寿命转化)
    PORT_SIGNAL = auto()    # 信号 (诱导/伪装)
    
    # ============================================================
    # v7.0 空间记忆 + 社会智能
    # ============================================================
    LIGHT_SENSOR = auto()   # 移动光源传感器 (v7.0)
    AGENT_RADAR_SENSOR = auto()  # 社会信号传感器 (v7.0)
    WALL_RADAR_SENSOR = auto()   # 墙壁雷达传感器 (v7.0)
    
    # ============================================================
    # v8.0 无尽边疆 - GPS坐标传感器
    # ============================================================
    GPS_SENSOR = auto()     # GPS坐标传感器 (v8.0)
    COMPASS_SENSOR = auto() # 指南针传感器 (v8.0)
    
    # ============================================================
    
    # ============================================================
    PHEROMONE_SENSOR = auto()  # 自身气味传感器
    
    # ============================================================
    
    # ============================================================
    META_NODE = auto()      # 元节点 (压缩的子网络)
    REWARD_PREDICTOR = auto()  # 奖励预测器 (v0.26)
    MODULATOR = auto()      # 神经调制器 (v0.26)
    SENSOR_CONTEXT = auto() # 元感知节点 (v0.27)
    BUFFER = auto()         # 循环记忆节点 (v0.24)
    COMM_OUT = auto()       # 通信输出
    COMM_IN = auto()        # 通信输入
    UPDATE_WEIGHT = auto()  # 权重更新
    POLY = auto()           # 多项式算子
    SWITCH = auto()         # 开关算子
    MACRO_EX = auto()       # 扩展宏算子
    
    # ============================================================
    # v0.0 统一场物理系统 - 传感器节点
    # 输入层：直接感知四个场的中心值与梯度
    # ============================================================
    # EPF 能量场感知 (3节点)
    SENSE_EPF_CENTER = auto()   # 能量场中心值 E(x,y)
    SENSE_EPF_GRAD_X = auto()   # 能量梯度 ∂E/∂x
    SENSE_EPF_GRAD_Y = auto()   # 能量梯度 ∂E/∂y
    
    # KIF 阻抗场感知 (3节点)
    SENSE_KIF_CENTER = auto()   # 阻抗场中心值 Z(x,y)
    SENSE_KIF_GRAD_X = auto()   # 阻抗梯度 ∂Z/∂x
    SENSE_KIF_GRAD_Y = auto()   # 阻抗梯度 ∂Z/∂y
    
    # ISF 压痕场感知 (3节点)
    SENSE_ISF_CENTER = auto()   # 压痕场中心值 S(x,y)
    SENSE_ISF_GRAD_X = auto()   # 压痕梯度 ∂S/∂x
    SENSE_ISF_GRAD_Y = auto()   # 压痕梯度 ∂S/∂y
    
    # ESF 应力场感知 (1节点)
    SENSE_ESF_VAL = auto()      # 应力值 σ(t)
    
    # 内部状态感知 (1节点)
    SENSE_INTERNAL_ENERGY = auto()  # 体内能量余额
    
    # ============================================================
    # v0.0 统一场物理系统 - 执行器节点
    # 输出层：严格绑定激活函数的物理致动器
    # ============================================================
    ACTUATOR_PERMEABILITY = auto()  # κ 渗透率 [0,1] - Sigmoid
    ACTUATOR_THRUST_X = auto()      # Fx 推力X [-1,1] - Tanh
    ACTUATOR_THRUST_Y = auto()      # Fy 推力Y [-1,1] - Tanh
    ACTUATOR_SIGNAL = auto()        # λ 信号强度 [0,1] - ReLU/Sigmoid
    ACTUATOR_DEFENSE = auto()       # S 防御刚性 [0,1] - Sigmoid


class Node:
    """
    基础计算节点 (v5.4)
    
    属性:
        node_id: 唯一标识符
        node_type: 节点类型 (NodeType)
        activation: 当前激活值
        constant_value: 常数值 (仅 CONSTANT 类型)
        name: 可选的友好名称
        delay_steps: 延迟步数 (v4.0: 多尺度时间动力学)
        delay_buffer: 延迟缓冲区 (支持多步延迟)
        is_frozen: 是否已冻结为SuperNode
        frozen_generation: 冻结发生的代
        fitness_history: 适应度贡献历史 (v4.0: 模块冻结用)
        
    v5.5 创新追踪:
        innovation_id: 全局创新序列号 (用于物种形成)
    """
    
    # v5.5 全局创新计数器
    _global_innovation_node = 0
    
    def __init__(
        self, 
        node_id: int, 
        node_type: NodeType, 
        constant_value: float = 0.0,
        name: Optional[str] = None,
        delay_steps: int = 1,  # v4.0: 默认1步，可配置
        innovation_id: Optional[int] = None  # v5.5: 创新序列号
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.activation = 0.0
        self.constant_value = constant_value
        self.name = name or f"{node_type.name}_{node_id}"
        
        # v5.5: 创新序列号
        if innovation_id is not None:
            self.innovation_id = innovation_id
        else:
            Node._global_innovation_node += 1
            self.innovation_id = Node._global_innovation_node
        
        # v4.0: 多尺度时间动力学
        self.delay_steps = max(1, delay_steps)  # 至少1步
        self.delay_buffer: deque = deque(maxlen=self.delay_steps)
        
        # v4.0: 兼容旧代码 (单步延迟)
        self.delay_state = 0.0
        
        # v4.0: 模块冻结追踪
        self.is_frozen = False
        self.frozen_generation = 0
        self.fitness_history: deque = deque(maxlen=25)  # 记录25代的贡献
        
        # v7.3: 熵增与腐蚀机制 - 节点稳定性
        self.stability = 1.0  # 初始稳定性
        self.corrosion_active = False  # 是否正在腐蚀
        self.steps_inactive = 0  # 不活跃步数计数
        
        # ============================================================
        
        # 每个传感器可以学习"注意"什么,而不需要预设类型
        # ============================================================
        if node_type in [NodeType.SENSOR]:
            # 方向注意力: 8个方向的重要性权重
            # 传感器会自动关注最重要的方向
            self.angle_weights = np.random.randn(8)
            self.angle_weights = self.angle_weights / (np.abs(self.angle_weights).sum() + 1e-8)
            
            # 距离注意力: 近/中/远
            self.distance_weights = np.random.randn(3)
            self.distance_weights = self.distance_weights / (np.abs(self.distance_weights).sum() + 1e-8)
            
            # 目标类型注意力: 食物/巢穴/敌对/墙壁
            self.target_type_weights = np.random.randn(4)
            self.target_type_weights = self.target_type_weights / (np.abs(self.target_type_weights).sum() + 1e-8)
        else:
            self.angle_weights = None
            self.distance_weights = None
            self.target_type_weights = None
        
        # 初始化延迟缓冲区
        for _ in range(self.delay_steps):
            self.delay_buffer.append(0.0)
    
    def __repr__(self):
        frozen_mark = "🔒" if self.is_frozen else ""
        return f"Node({self.node_id}, {self.node_type.name}{frozen_mark}, act={self.activation:.3f})"
    
    # ============================================================
    
    # ============================================================
    def mutate_sensor_weights(self, sigma: float = 0.1):
        """突变传感器的感知权重"""
        if self.node_type != NodeType.SENSOR:
            return
        
        # 方向权重突变
        if self.angle_weights is not None and np.random.random() < 0.3:
            noise = np.random.randn(8) * sigma
            self.angle_weights += noise
            # 归一化
            self.angle_weights = np.clip(self.angle_weights, -3, 3)
            self.angle_weights = self.angle_weights / (np.abs(self.angle_weights).sum() + 1e-8)
        
        # 距离权重突变
        if self.distance_weights is not None and np.random.random() < 0.3:
            noise = np.random.randn(3) * sigma
            self.distance_weights += noise
            self.distance_weights = np.clip(self.distance_weights, -3, 3)
            self.distance_weights = self.distance_weights / (np.abs(self.distance_weights).sum() + 1e-8)
        
        # 目标类型权重突变
        if self.target_type_weights is not None and np.random.random() < 0.3:
            noise = np.random.randn(4) * sigma
            self.target_type_weights += noise
            self.target_type_weights = np.clip(self.target_type_weights, -3, 3)
            self.target_type_weights = self.target_type_weights / (np.abs(self.target_type_weights).sum() + 1e-8)
    
    def get_sensor_attention(self, targets: List[Dict]) -> float:
        """
         使用学习到的注意力计算传感器输出
        
        targets: List of {angle, distance, type}
        - angle: 0-360度
        - distance: 距离
        - type: 0=food, 1=nest, 2=rival, 3=wall
        
        返回: 加权感知的激活值
        """
        if self.node_type != NodeType.SENSOR or not targets:
            return 0.0
        
        total_activation = 0.0
        
        for target in targets:
            # 1. 方向注意力
            angle_bin = int(target['angle'] / 45) % 8  # 8个方向bin
            angle_attention = self.angle_weights[angle_bin] if self.angle_weights is not None else 1.0
            
            # 2. 距离注意力
            if target['distance'] < 20:
                dist_bin = 0  # 近
            elif target['distance'] < 50:
                dist_bin = 1  # 中
            else:
                dist_bin = 2  # 远
            dist_attention = self.distance_weights[dist_bin] if self.distance_weights is not None else 1.0
            
            # 3. 目标类型注意力
            type_attention = self.target_type_weights[target['type']] if self.target_type_weights is not None else 1.0
            
            # 综合注意力
            attention = angle_attention * dist_attention * type_attention
            total_activation += attention
        
        return total_activation


# ============================================================
# 2. 超级节点 (SuperNode) - 冻结的功能器官
# ============================================================

class SuperNode:
    """
    超级节点 - 从优秀子图冻结而来的"功能器官"
    
    v4.0 新增:
    - 当某个子图在连续 20 代中持续贡献正向适应度时冻结
    - 冻结后不可拆分，整个作为单一模块使用
    - 可以参与递归突变（与基础算子连线）
    
    结构:
        - nodes: 内部节点列表 (已冻结)
        - edges: 内部边列表 (已冻结)
        - input_ids: 外部输入接口
        - output_ids: 外部输出接口
        - function_name: 功能名称 (如 "direction_filter", "pattern_detector")
    """
    
    _id_counter = 0
    
    def __init__(
        self,
        node_ids: List[int],
        edge_list: List[Dict],
        input_ids: List[int],
        output_ids: List[int],
        generation: int,
        function_name: Optional[str] = None
    ):
        SuperNode._id_counter += 1
        self.id = SuperNode._id_counter
        self.node_ids = set(node_ids)
        self.edges = edge_list
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.frozen_generation = generation
        self.function_name = function_name or f"function_{self.id}"
        
        # 冻结后的内部计算缓存
        self._cached_output: Optional[float] = None
        self._cache_valid = False
        
        # 统计
        self.usage_count = 0
        self.total_contribution = 0.0
    
    def compute(self, inputs: Dict[int, float]) -> Dict[int, float]:
        """
        计算超级节点的输出
        (内部子图的前向传播)
        """
        if self._cache_valid and self._cached_output is not None:
            return {oid: self._cached_output for oid in self.output_ids}
        
        # 简化的内部计算: 内部节点执行
        internal_activations = {nid: 0.0 for nid in self.node_ids}
        
        # 设置输入
        for iid in self.input_ids:
            if iid in inputs:
                internal_activations[iid] = inputs[iid]
        
        # 迭代计算 (简化版)
        for _ in range(3):  # 3轮迭代
            for edge in self.edges:
                src, tgt = edge['source'], edge['target']
                if src in internal_activations and tgt in internal_activations:
                    w = edge.get('weight', 1.0)
                    internal_activations[tgt] += internal_activations[src] * w
        
        # 提取输出
        outputs = {}
        for oid in self.output_ids:
            if oid in internal_activations:
                outputs[oid] = np.tanh(internal_activations[oid])
        
        self._cached_output = outputs.get(list(self.output_ids)[0], 0.0) if outputs else 0.0
        self._cache_valid = True
        
        return outputs
    
    def __repr__(self):
        return f"SuperNode(id={self.id}, nodes={len(self.node_ids)}, inputs={self.input_ids}, outputs={self.output_ids})"


# ============================================================
# 3. 子图追踪器 (SubgraphTracker) - 模块冻结机制
# ============================================================

class SubgraphTracker:
    """
    子图适应度追踪器 - 决定哪些子图应该被冻结
    
    v4.0 核心机制:
    - 追踪每个潜在模块的适应度贡献
    - 连续 20 代正向贡献 → 触发冻结
    - 使用滑动窗口评估稳定性
    """
    
    def __init__(
        self,
        freeze_threshold: int = 20,      # 连续正向贡献代数
        positive_threshold: float = 0.5   # 正向贡献阈值
    ):
        self.freeze_threshold = freeze_threshold
        self.positive_threshold = positive_threshold
        
        # 子图ID → 贡献历史
        # subgraph_id: frozenset of node_ids
        self.subgraph_contributions: Dict[frozenset, deque] = defaultdict(
            lambda: deque(maxlen=freeze_threshold + 5)
        )
        
        # 冻结队列
        self.frozen_subgraphs: Dict[frozenset, SuperNode] = {}
        self.pending_freeze: Dict[frozenset, int] = {}  # subgraph → 连续正向代数
    
    def record_contribution(self, subgraph: frozenset, contribution: float) -> None:
        """记录子图的适应度贡献"""
        self.subgraph_contributions[subgraph].append(contribution)
        
        # 检查是否应该冻结
        if contribution > self.positive_threshold:
            self.pending_freeze[subgraph] = self.pending_freeze.get(subgraph, 0) + 1
        else:
            self.pending_freeze[subgraph] = 0
        
        # 触发冻结
        if self.pending_freeze.get(subgraph, 0) >= self.freeze_threshold:
            self._freeze_subgraph(subgraph)
    
    def _freeze_subgraph(self, subgraph: frozenset) -> None:
        """冻结子图为超级节点"""
        if subgraph not in self.frozen_subgraphs:
            # 需要从genome获取详细信息
            pass  # 在OperatorGenome中调用时填充
    
    def should_freeze(self, subgraph: frozenset) -> bool:
        """检查子图是否应该被冻结"""
        return subgraph in self.frozen_subgraphs
    
    def get_frozen_subgraphs(self) -> Dict[frozenset, SuperNode]:
        """获取所有已冻结的子图"""
        return self.frozen_subgraphs.copy()


# ============================================================
# 4. 宏算子 (MacroOperator) - 基因硬化封装
# ============================================================

class MacroOperator:
    """
    宏算子 - 将表现优秀的子图封装为可复用的模块
    
    结构:
        - input_nodes: 输入节点ID列表
        - output_nodes: 输出节点ID列表  
        - internal_nodes: 内部节点ID列表
        - internal_edges: 内部边列表
        - fitness_contribution: 该子图对适应度的贡献
    """
    
    _id_counter = 0
    
    def __init__(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        input_ids: List[int],
        output_ids: List[int]
    ):
        MacroOperator._id_counter += 1
        self.id = MacroOperator._id_counter
        self.nodes = nodes  # 节点定义列表
        self.edges = edges  # 边定义列表
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.fitness_contribution = 0.0
        self.usage_count = 0
        
        # v5.0 新增: 代谢与生存追踪
        self.metabolic_cost = sum(1 for _ in nodes) + sum(1 for _ in edges)  # 节点+边成本
        self.generation_created = 0
        self.last_used_generation = 0
        self.consecutive_low_usage = 0  # 连续低使用代数
        self.avg_fitness_when_used = []  # 使用时的适应度记录
        
        # v5.2: 深度冷冻标志
        self.is_frozen = False  # 深度冷冻后禁止突变
        self.name = "Macro_" + str(self.id)
    
    def record_usage(self, generation: int, fitness: float) -> None:
        """记录使用情况"""
        self.usage_count += 1
        self.last_used_generation = generation
        self.avg_fitness_when_used.append(fitness)
        if len(self.avg_fitness_when_used) > 10:
            self.avg_fitness_when_used.pop(0)
        self.consecutive_low_usage = 0  # 重置
    
    def should_extinct(self, current_generation: int, usage_threshold: int = 5, idle_threshold: int = 50) -> bool:
        """
        判断是否应该灭绝
        
        条件:
        1. 连续50代未使用 OR
        2. 连续低使用(usage < threshold)超过50代 OR
        3. 平均适应度低于创建时
        """
        if self.usage_count == 0:
            self.consecutive_low_usage += 1
        else:
            self.consecutive_low_usage = 0
        
        # 50代未使用
        if current_generation - self.last_used_generation > idle_threshold:
            return True
        
        # 连续50代低使用
        if self.consecutive_low_usage > idle_threshold:
            return True
        
        # 平均适应度下降
        if len(self.avg_fitness_when_used) >= 5:
            recent_avg = np.mean(self.avg_fitness_when_used)
            if recent_avg < self.fitness_contribution * 0.5:  # 低于创建时50%
                return True
        
        return False
    
    def get_average_fitness(self) -> float:
        """获取平均适应度"""
        if self.avg_fitness_when_used:
            return np.mean(self.avg_fitness_when_used)
        return 0.0
    
    @staticmethod
    def extract_from_genome(genome: 'OperatorGenome', top_k: int = 3) -> List['MacroOperator']:
        """
        从表现优秀的基因组中提取宏算子
        
        算法:
            1. 找到高权重的连接模式
            2. 识别 3-5 节点的连通子图
            3. 封装为 MacroOperator
        """
        macros = []
        
        # 找到高权重边 (|weight| > 0.7)
        high_weight_edges = [
            e for e in genome.edges 
            if e['enabled'] and abs(e['weight']) > 0.7
        ]
        
        if len(high_weight_edges) < 2:
            return macros
        
        # 聚类形成子图
        node_groups = {}
        for edge in high_weight_edges:
            src, tgt = edge['source_id'], edge['target_id']
            if src not in node_groups:
                node_groups[src] = set()
            if tgt not in node_groups:
                node_groups[tgt] = set()
            node_groups[src].add(tgt)
            node_groups[tgt].add(src)
        
        # 提取连通分量 (3-5节点)
        visited = set()
        for start_node in node_groups:
            if start_node in visited:
                continue
            
            component = {start_node}
            queue = [start_node]
            
            while queue and len(component) < 5:
                node = queue.pop(0)
                for neighbor in node_groups.get(node, []):
                    if neighbor not in component:
                        component.add(neighbor)
                        queue.append(neighbor)
            
            if 3 <= len(component) <= 5:
                visited.update(component)
                
                # 提取子图
                sub_nodes = [genome.nodes[n] for n in component if n in genome.nodes]
                sub_edges = [e for e in high_weight_edges 
                           if e['source_id'] in component and e['target_id'] in component]
                
                # 找输入输出节点
                all_srcs = {e['source_id'] for e in sub_edges}
                all_tgts = {e['target_id'] for e in sub_edges}
                input_ids = list(all_srcs - all_tgts)[:2]  # 最多2输入
                output_ids = list(all_tgts - all_srcs)[:2]  # 最多2输出
                
                if sub_nodes and sub_edges:
                    macro = MacroOperator(
                        nodes=[{'node_id': n.node_id, 'node_type': n.node_type.name} for n in sub_nodes],
                        edges=sub_edges,
                        input_ids=input_ids,
                        output_ids=output_ids
                    )
                    macros.append(macro)
        
        return macros[:top_k]
    
    def apply_to(self, genome: 'OperatorGenome', insertion_point: int) -> bool:
        """
        将宏算子应用到基因组
        
        参数:
            genome: 目标基因组
            insertion_point: 插入起始节点ID
        """
        # 简单的宏应用: 复制所有节点和边
        node_id_offset = insertion_point
        id_mapping = {}
        
        # 复制节点
        for node_def in self.nodes:
            old_id = node_def['node_id']
            new_id = node_id_offset + old_id
            id_mapping[old_id] = new_id
            
            # 创建新节点
            new_node = Node(
                node_id=new_id,
                node_type=NodeType[node_def['node_type']] if isinstance(node_def['node_type'], str) else node_def['node_type'],
                name=f"MACRO_{self.id}_{new_id}"
            )
            genome.add_node(new_node)
        
        # 复制边 (更新ID)
        for edge in self.edges:
            new_src = id_mapping.get(edge['source_id'], edge['source_id'])
            new_tgt = id_mapping.get(edge['target_id'], edge['target_id'])
            try:
                genome.add_edge(new_src, new_tgt, edge['weight'], edge['enabled'])
            except:
                pass  # 边可能已存在
        
        self.usage_count += 1
        return True



# ============================================================
# Node JSON 序列化 (在Node类之后)
# ============================================================
def node_to_dict(node) -> dict:
    """导出节点为字典"""
    data = {
        'node_id': node.node_id,
        'node_type': node.node_type.name,
        'constant_value': node.constant_value,
        'name': node.name,
        'delay_steps': node.delay_steps,
        'innovation_id': node.innovation_id,
        'is_frozen': node.is_frozen,
        'stability': node.stability,
    }
    
    if node.angle_weights is not None:
        data['angle_weights'] = node.angle_weights.tolist()
    if node.distance_weights is not None:
        data['distance_weights'] = node.distance_weights.tolist()
    if node.target_type_weights is not None:
        data['target_type_weights'] = node.target_type_weights.tolist()
        
    return data

def node_from_dict(data: dict, NodeClass):
    """从字典导入节点"""
    from .node import NodeType
    node = NodeClass(
        node_id=data['node_id'],
        node_type=NodeType[data['node_type']],
        constant_value=data.get('constant_value', 0.0),
        name=data.get('name'),
        delay_steps=data.get('delay_steps', 1),
        innovation_id=data.get('innovation_id')
    )
    node.is_frozen = data.get('is_frozen', False)
    node.stability = data.get('stability', 1.0)
    
    if 'angle_weights' in data:
        import numpy as np
        node.angle_weights = np.array(data['angle_weights'])
    if 'distance_weights' in data:
        node.distance_weights = np.array(data['distance_weights'])
    if 'target_type_weights' in data:
        node.target_type_weights = np.array(data['target_type_weights'])
        
    return node
# 3. 算子基因组 (OperatorGenome)
# ============================================================

    # ============================================================
    # JSON 序列化支持
    # ============================================================
    def to_dict(self) -> Dict:
        """导出节点为字典"""
        data = {
            'node_id': self.node_id,
            'node_type': self.node_type.name,
            'constant_value': self.constant_value,
            'name': self.name,
            'delay_steps': self.delay_steps,
            'innovation_id': self.innovation_id,
            'is_frozen': self.is_frozen,
            'stability': self.stability,
        }
        
        # 传感器权重
        if self.angle_weights is not None:
            data['angle_weights'] = self.angle_weights.tolist()
        if self.distance_weights is not None:
            data['distance_weights'] = self.distance_weights.tolist()
        if self.target_type_weights is not None:
            data['target_type_weights'] = self.target_type_weights.tolist()
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Node':
        """从字典导入节点"""
        node = cls(
            node_id=data['node_id'],
            node_type=NodeType[data['node_type']],
            constant_value=data.get('constant_value', 0.0),
            name=data.get('name'),
            delay_steps=data.get('delay_steps', 1),
            innovation_id=data.get('innovation_id')
        )
        node.is_frozen = data.get('is_frozen', False)
        node.stability = data.get('stability', 1.0)
        
        # 传感器权重
        if 'angle_weights' in data:
            node.angle_weights = np.array(data['angle_weights'])
        if 'distance_weights' in data:
            node.distance_weights = np.array(data['distance_weights'])
        if 'target_type_weights' in data:
            node.target_type_weights = np.array(data['target_type_weights'])
            
        return node
