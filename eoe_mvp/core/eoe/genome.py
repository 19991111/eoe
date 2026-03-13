from .node import Node, node_to_dict, node_from_dict
from .node import node_to_dict, node_from_dict
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple
import numpy as np

from .node import Node, NodeType


class InnovationManager:
    """
    创新ID管理器 - 支持多进程并行和多次实验
    
    问题解决:
    - 类级别静态变量在多进程间不同步
    - 多次实验间ID污染
    
    方案:
    - 每个演化实验独立实例
    - 支持序列化/反序列化恢复ID
    """
    
    def __init__(self, start_id: int = 0):
        self._next_edge_id = start_id
        self._next_node_id = start_id
        self._edge_id_history: Dict[Tuple[int, int], int] = {}  # (src, tgt) -> id
    
    def get_edge_id(self, source_id: int, target_id: int, weight: float = None) -> int:
        """获取边的创新ID (如果边已存在则返回已有ID)"""
        edge_key = (source_id, target_id)
        if edge_key in self._edge_id_history:
            return self._edge_id_history[edge_key]
        
        # 新边，分配新ID
        new_id = self._next_edge_id
        self._next_edge_id += 1
        self._edge_id_history[edge_key] = new_id
        return new_id
    
    def get_node_id(self) -> int:
        """分配新的节点ID"""
        new_id = self._next_node_id
        self._next_node_id += 1
        return new_id
    
    def reset(self):
        """重置ID计数器"""
        self._next_edge_id = 0
        self._next_node_id = 0
        self._edge_id_history.clear()
    
    def get_state(self) -> Dict:
        """获取当前状态 (用于序列化)"""
        return {
            'next_edge_id': self._next_edge_id,
            'next_node_id': self._next_node_id,
            'edge_id_history': {str(k): v for k, v in self._edge_id_history.items()}
        }
    
    def set_state(self, state: Dict):
        """恢复状态 (从序列化恢复)"""
        self._next_edge_id = state.get('next_edge_id', 0)
        self._next_node_id = state.get('next_node_id', 0)
        self._edge_id_history = {eval(k): v for k, v in state.get('edge_id_history', {}).items()}

class OperatorGenome:
    """
    智能体的"大脑" - 带有权重的有向无环图 (DAG)
    
    结构:
        nodes: Dict[int, Node] - 所有节点的字典
        edges: List[Dict] - 边列表，每条边包含:
            - source_id: 源节点 ID
            - target_id: 目标节点 ID
            - weight: 权重
            - enabled: 是否启用
    
    方法:
        forward(sensor_inputs): 前向传播，返回执行器输出
    """
    
    def __init__(self, innovation_manager: Optional[InnovationManager] = None):
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Dict] = []
        self._adjacency_list: Dict[int, List[int]] = defaultdict(list)  # target -> sources
        self._reverse_adj: Dict[int, List[int]] = defaultdict(list)    # source -> targets
        self._topo_order: Optional[List[int]] = None
        
        # 创新ID管理器 (可共享)
        self._innovation_mgr = innovation_manager or InnovationManager()
    
    def add_node(self, node: Node) -> None:
        """添加一个节点到基因组"""
        self.nodes[node.node_id] = node
        self._topo_order = None  # 拓扑序失效，需要重新计算
    
    def add_edge(
        self, 
        source_id: int, 
        target_id: int, 
        weight: float = 1.0, 
        enabled: bool = True,
        innovation_id: Optional[int] = None  # v5.5: 创新序列号
    ) -> None:
        """
        添加一条边 (连接)
        
        参数:
            source_id: 源节点 ID
            target_id: 目标节点 ID
            weight: 连接权重 (默认为 1.0)
            enabled: 是否启用 (用于基因突变)
            innovation_id: v5.5 创新序列号 (可选)
        """
        if source_id not in self.nodes:
            raise ValueError(f"源节点 {source_id} 不存在")
        if target_id not in self.nodes:
            raise ValueError(f"目标节点 {target_id} 不存在")
        
        # v5.5: 分配创新ID (使用InnovationManager)
        if innovation_id is not None:
            edge_innovation = innovation_id
        else:
            edge_innovation = self._innovation_mgr.get_edge_id(source_id, target_id, weight)
        
        edge = {
            'source_id': source_id,
            'target_id': target_id,
            'weight': weight,
            'enabled': enabled,
            'learning_rate': 0.0,  # 赫布学习率 (0=不可学习)
            'innovation_id': edge_innovation  # v5.5: 创新序列号
        }
        self.edges.append(edge)
        
        # 更新邻接表
        if enabled:
            self._adjacency_list[target_id].append(source_id)
            self._reverse_adj[source_id].append(target_id)
        
        self._topo_order = None  # 拓扑序失效
    
    def _topological_sort(self) -> List[int]:
        """
        Kahn 算法拓扑排序
        
        返回:
            按计算顺序排列的节点 ID 列表
        """
        if self._topo_order is not None:
            return self._topo_order
        
        # 计算入度
        in_degree = defaultdict(int)
        for node_id in self.nodes:
            in_degree[node_id] = 0
        
        for edge in self.edges:
            if edge['enabled']:
                in_degree[edge['target_id']] += 1
        
        # 入度为 0 的节点队列
        queue = deque([nid for nid in self.nodes if in_degree[nid] == 0])
        topo_order = []
        
        while queue:
            node_id = queue.popleft()
            topo_order.append(node_id)
            
            # 更新相邻节点入度
            for neighbor in self._reverse_adj[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检测环
        if len(topo_order) != len(self.nodes):
            raise ValueError("图中存在环! 无法进行拓扑排序")
        
        self._topo_order = topo_order
        return topo_order
    
    def forward(self, sensor_inputs: np.ndarray) -> np.ndarray:
        """
        前向传播 - 根据输入计算输出
        
        特殊处理:
            - DELAY 节点: 输出上一时刻的输入值，并在计算后更新状态
            - 允许包含 DELAY 的路径形成环路 (通过延迟状态打破循环依赖)
        
        参数:
            sensor_inputs: 传感器输入数组
            
        返回:
            执行器输出数组
        """
        if not self.nodes:
            raise ValueError("基因组没有任何节点")
        
        # 获取拓扑排序顺序 (可能包含环，但 DELAY 会打破)
        try:
            topo_order = self._topological_sort()
        except ValueError as e:
            # 存在环，使用简化的计算顺序 (按节点 ID)
            topo_order = sorted(self.nodes.keys())
        
        # 重置非状态节点 (保留 DELAY 的延迟状态)
        for node in self.nodes.values():
            if node.node_type == NodeType.CONSTANT:
                node.activation = node.constant_value
            elif node.node_type == NodeType.DELAY:
                # DELAY 节点: 输出最旧的延迟值
                node.activation = node.delay_buffer[0] if node.delay_buffer else 0.0
                # 同步 delay_state 保持兼容
                node.delay_state = node.activation
            else:
                node.activation = 0.0
        
        # 收集传感器节点用于输入
        sensor_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.SENSOR]
        light_sensor_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.LIGHT_SENSOR]
        agent_radar_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.AGENT_RADAR_SENSOR]
        gps_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.GPS_SENSOR]
        compass_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.COMPASS_SENSOR]
        
        # ============================================================
        # v13.0: 统一场物理传感器节点
        # 11维输入: [EPF_CENTER, EPF_GRAD_X, EPF_GRAD_Y,
        #           KIF_CENTER, KIF_GRAD_X, KIF_GRAD_Y,
        #           ISF_CENTER, ISF_GRAD_X, ISF_GRAD_Y,
        #           ESF_VAL, INTERNAL_ENERGY]
        # ============================================================
        v13_sensor_nodes = {}
        v13_sensor_order = [
            NodeType.SENSE_EPF_CENTER, NodeType.SENSE_EPF_GRAD_X, NodeType.SENSE_EPF_GRAD_Y,
            NodeType.SENSE_KIF_CENTER, NodeType.SENSE_KIF_GRAD_X, NodeType.SENSE_KIF_GRAD_Y,
            NodeType.SENSE_ISF_CENTER, NodeType.SENSE_ISF_GRAD_X, NodeType.SENSE_ISF_GRAD_Y,
            NodeType.SENSE_ESF_VAL, NodeType.SENSE_INTERNAL_ENERGY
        ]
        for nt in v13_sensor_order:
            nodes_of_type = [n for n in self.nodes.values() if n.node_type == nt]
            if nodes_of_type:
                v13_sensor_nodes[nt] = nodes_of_type[0]
        
        # v7.0: 设置传感器输入
        # 传感器输入格式: [SENSOR_0, SENSOR_1, LIGHT_0, LIGHT_1, RADAR_0, RADAR_1, RADAR_2]
        
        # 基础食物传感器 (node_id 0, 1)
        for i, sensor_node in enumerate(sensor_nodes):
            if i < len(sensor_inputs):
                sensor_node.activation = sensor_inputs[i]
        
        # 光源传感器 (node_id 6, 7) - 对应 sensor_inputs[2], sensor_inputs[3]
        for i, node in enumerate(light_sensor_nodes):
            idx = 2 + i  # 偏移2位
            if idx < len(sensor_inputs):
                node.activation = sensor_inputs[idx]
        
        # 社会雷达传感器 (node_id 8, 9, 10) - 对应 sensor_inputs[4], [5], [6]
        for i, node in enumerate(agent_radar_nodes):
            idx = 4 + i  # 偏移4位
            if idx < len(sensor_inputs):
                node.activation = sensor_inputs[idx]
        
        # ============================================================
        # v13.0: 设置统一场物理传感器输入
        # 格式: [EPF×3, KIF×3, ISF×3, ESF×1, ENERGY×1] = 11维
        # ============================================================
        v13_input_offset = len(sensor_inputs) - 11 if len(sensor_inputs) >= 11 else 0
        if v13_input_offset >= 0:
            v13_inputs = sensor_inputs[v13_input_offset:]
            v13_keys = list(v13_sensor_nodes.keys())
            for i, nt in enumerate(v13_keys):
                if i < len(v13_inputs):
                    v13_sensor_nodes[nt].activation = v13_inputs[i]
        
        # v8.0: GPS传感器 (node_id 11-14) - 从扩展输入获取
        # 格式: [GPS_X, GPS_Y, GPS_DIST, GPS_BEARING]
        # v8.0: COMPASS传感器 (node_id 15-17) - 从扩展输入获取
        # sensor_inputs扩展部分应该包含GPS和COMPASS值
        
        # 按拓扑顺序计算每个节点
        for node_id in topo_order:
            node = self.nodes[node_id]
            
            # 跳过输入节点 (已设置)
            if node.node_type in (NodeType.SENSOR, NodeType.LIGHT_SENSOR, 
                                  NodeType.AGENT_RADAR_SENSOR, NodeType.GPS_SENSOR,
                                  NodeType.COMPASS_SENSOR,
                                  # v13.0: 统一场物理传感器
                                  NodeType.SENSE_EPF_CENTER, NodeType.SENSE_EPF_GRAD_X, NodeType.SENSE_EPF_GRAD_Y,
                                  NodeType.SENSE_KIF_CENTER, NodeType.SENSE_KIF_GRAD_X, NodeType.SENSE_KIF_GRAD_Y,
                                  NodeType.SENSE_ISF_CENTER, NodeType.SENSE_ISF_GRAD_X, NodeType.SENSE_ISF_GRAD_Y,
                                  NodeType.SENSE_ESF_VAL, NodeType.SENSE_INTERNAL_ENERGY):
                continue
            
            # 跳过常数节点 (已设置)
            if node.node_type == NodeType.CONSTANT:
                continue
            
            # DELAY 节点: 多尺度时间动力学 (v4.0)
            if node.node_type == NodeType.DELAY:
                input_edges = [
                    e for e in self.edges 
                    if e['enabled'] and e['target_id'] == node_id
                ]
                if input_edges:
                    # 计算当前输入
                    input_val = 0.0
                    for edge in input_edges:
                        source_node = self.nodes[edge['source_id']]
                        input_val += source_node.activation * edge['weight']
                    
                    # 更新延迟缓冲区
                    node.delay_buffer.append(input_val)
                
                # 输出延迟缓冲区最旧的 value (实现多尺度延迟)
                node.activation = node.delay_buffer[0] if node.delay_buffer else 0.0
                continue
            
            # 获取所有输入边
            input_edges = [
                e for e in self.edges 
                if e['enabled'] and e['target_id'] == node_id
            ]
            
            if not input_edges:
                # 无输入边，保持为 0
                continue
            
            # 收集输入值
            input_values = []
            for edge in input_edges:
                source_node = self.nodes[edge['source_id']]
                input_values.append(source_node.activation * edge['weight'])
            
            # 根据节点类型计算输出
            if node.node_type == NodeType.ADD:
                node.activation = sum(input_values)
            
            elif node.node_type == NodeType.MULTIPLY:
                node.activation = 1.0
                for val in input_values:
                    node.activation *= val
            
            elif node.node_type == NodeType.THRESHOLD:
                # 阈值算子: Output = 1.0 if Input > 0.5 else 0.0
                total_input = sum(input_values)
                node.activation = 1.0 if total_input > 0.5 else 0.0
            
            elif node.node_type == NodeType.PREDICTOR:
                # 预测节点: 简单地输出输入的加权和（可用于学习预测）
                # 在训练阶段，这会尝试匹配下一时刻的传感器值
                node.activation = sum(input_values) if input_values else 0.0
            
            elif node.node_type == NodeType.ENTITY_RADAR:
                # 社会传感器: ENTITY_RADAR 在前向传播时需要环境信息
                # 这里使用节点的 radar_data 属性（在Environment中设置）
                # 默认输出 0（表示无检测）
                node.activation = getattr(node, 'radar_data', 0.0)
            
            elif node.node_type == NodeType.MACRO:
                # 宏算子: 执行预封装的子图
                # 从 macro_subgraph 获取输出
                node.activation = getattr(node, 'macro_output', 0.0)
            
            elif node.node_type == NodeType.ACTUATOR:
                # 执行器：对所有输入求和 ( 添加ReLU确保非负)
                node.activation = max(0.0, sum(input_values))
                                                                                                
            # ============================================================
            # v6.0 GAIA 物理输出端口 (纯基础算子驱动)
            # 基础算子通过连接这些端口来控制物理行为
            # ============================================================
            elif node.node_type == NodeType.PORT_MOTION:
                # 运动 [速度, 转向]: 激活值控制移动
                node.activation = sum(input_values) if input_values else 0.0
            
            elif node.node_type == NodeType.PORT_OFFENSE:
                # 攻击强度: 碰撞时对敌方的能量掠夺力
                node.activation = sum(input_values) if input_values else 0.0
            
            elif node.node_type == NodeType.PORT_DEFENSE:
                # 防御硬度: 被撞击时的能量损失降低
                node.activation = sum(input_values) if input_values else 0.0
            
            elif node.node_type == NodeType.PORT_REPAIR:
                # 修复: 能量→寿命转化
                node.activation = sum(input_values) if input_values else 0.0
            
            elif node.node_type == NodeType.PORT_SIGNAL:
                # 信号: 改变感知频率 (诱导/伪装)
                node.activation = sum(input_values) if input_values else 0.0
        
        # 收集执行器输出 + 预测器输出
        actuator_nodes = [
            n for n in self.nodes.values() 
            if n.node_type in (NodeType.ACTUATOR, 
                               NodeType.PORT_MOTION, NodeType.PORT_OFFENSE,
                               NodeType.PORT_DEFENSE, NodeType.PORT_REPAIR,
                               NodeType.PORT_SIGNAL,
                               # v13.0: 统一场物理执行器
                               NodeType.ACTUATOR_PERMEABILITY, NodeType.ACTUATOR_THRUST_X,
                               NodeType.ACTUATOR_THRUST_Y, NodeType.ACTUATOR_SIGNAL,
                               NodeType.ACTUATOR_DEFENSE)
        ]
        predictor_nodes = [
            n for n in self.nodes.values()
            if n.node_type == NodeType.PREDICTOR
        ]
        
        # 按 node_id 排序确保一致性
        actuator_nodes.sort(key=lambda n: n.node_id)
        predictor_nodes.sort(key=lambda n: n.node_id)
        
        # ============================================================
        # v13.0: 激活函数钳制
        # - ACTUATOR_PERMEABILITY, ACTUATOR_DEFENSE → Sigmoid [0,1]
        # - ACTUATOR_THRUST_X/Y → Tanh [-1,1]
        # - ACTUATOR_SIGNAL → ReLU [0,1]
        # ============================================================
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        
        def apply_activation(node_type: str, value: float) -> float:
            """应用物理致动器激活函数"""
            if node_type in ('ACTUATOR_PERMEABILITY', 'ACTUATOR_DEFENSE', 'PORT_DEFENSE'):
                return sigmoid(value)
            elif node_type in ('ACTUATOR_THRUST_X', 'ACTUATOR_THRUST_Y', 'PORT_MOTION'):
                return np.tanh(value)
            elif node_type in ('ACTUATOR_SIGNAL', 'PORT_SIGNAL'):
                return max(0.0, min(1.0, value))  # ReLU + clamp
            else:
                return np.tanh(value)  # 默认Tanh
        
        # 应用激活函数
        actuator_outputs = [apply_activation(n.node_type.name, n.activation) for n in actuator_nodes]
        predictor_outputs = [n.activation for n in predictor_nodes]
        
        outputs = np.array(actuator_outputs + predictor_outputs)
        
        # ============================================================
        # v7.3: 熵增与腐蚀 - 更新节点稳定性
        # ============================================================
        self._update_node_stability(topo_order)
        
        return outputs
    
    # ============================================================
    # v7.3: 熵增与腐蚀机制
    # ============================================================
    
    def _update_node_stability(self, topo_order: List[int]):
        """
        更新所有节点的稳定性
        
        规则:
        - 若 abs(output) < 0.001: stability *= 0.99
        - 若 abs(output) >= 0.001: stability = 1.0
        """
        for node_id in topo_order:
            node = self.nodes[node_id]
            
            # 跳过传感器、常量、执行器端口 (不参与腐蚀)
            if node.node_type in (NodeType.SENSOR, NodeType.LIGHT_SENSOR, 
                                  NodeType.AGENT_RADAR_SENSOR, NodeType.GPS_SENSOR,
                                  NodeType.COMPASS_SENSOR, NodeType.CONSTANT,
                                  NodeType.ACTUATOR, NodeType.PORT_MOTION,
                                  NodeType.PORT_OFFENSE, NodeType.PORT_DEFENSE,
                                  NodeType.PORT_REPAIR, NodeType.PORT_SIGNAL,
                                  # v13.0: 统一场物理传感器
                                  NodeType.SENSE_EPF_CENTER, NodeType.SENSE_EPF_GRAD_X, NodeType.SENSE_EPF_GRAD_Y,
                                  NodeType.SENSE_KIF_CENTER, NodeType.SENSE_KIF_GRAD_X, NodeType.SENSE_KIF_GRAD_Y,
                                  NodeType.SENSE_ISF_CENTER, NodeType.SENSE_ISF_GRAD_X, NodeType.SENSE_ISF_GRAD_Y,
                                  NodeType.SENSE_ESF_VAL, NodeType.SENSE_INTERNAL_ENERGY,
                                  # v13.0: 统一场物理执行器
                                  NodeType.ACTUATOR_PERMEABILITY, NodeType.ACTUATOR_THRUST_X,
                                  NodeType.ACTUATOR_THRUST_Y, NodeType.ACTUATOR_SIGNAL,
                                  NodeType.ACTUATOR_DEFENSE):
                continue
            
            # 检查输出是否活跃
            if abs(node.activation) < 0.001:
                # 不活跃: 衰减稳定性
                node.stability *= 0.99  # 可调衰减率
                node.steps_inactive += 1
            else:
                # 活跃: 重置稳定性
                node.stability = 1.0
                node.steps_inactive = 0
    
    def apply_functional_corrosion(self) -> Dict:
        """
        应用功能腐蚀 (每帧调用)
        
        噪声阶段: stability < 0.5 时，输出叠加噪声
        坍塌阶段: stability < 0.2 时，1%概率算子突变
        
        返回:
            腐蚀统计 {"noisy_nodes": int, "mutated_nodes": int}
        """
        stats = {"noisy_nodes": 0, "mutated_nodes": 0}
        
        for node in self.nodes.values():
            # 跳过传感器和端口
            if node.node_type in (NodeType.SENSOR, NodeType.LIGHT_SENSOR,
                                  NodeType.AGENT_RADAR_SENSOR, NodeType.GPS_SENSOR,
                                  NodeType.COMPASS_SENSOR, NodeType.CONSTANT,
                                  NodeType.ACTUATOR, NodeType.PORT_MOTION,
                                  NodeType.PORT_OFFENSE, NodeType.PORT_DEFENSE,
                                  NodeType.PORT_REPAIR, NodeType.PORT_SIGNAL,
                                  # v13.0: 统一场物理
                                  NodeType.SENSE_EPF_CENTER, NodeType.SENSE_EPF_GRAD_X, NodeType.SENSE_EPF_GRAD_Y,
                                  NodeType.SENSE_KIF_CENTER, NodeType.SENSE_KIF_GRAD_X, NodeType.SENSE_KIF_GRAD_Y,
                                  NodeType.SENSE_ISF_CENTER, NodeType.SENSE_ISF_GRAD_X, NodeType.SENSE_ISF_GRAD_Y,
                                  NodeType.SENSE_ESF_VAL, NodeType.SENSE_INTERNAL_ENERGY,
                                  NodeType.ACTUATOR_PERMEABILITY, NodeType.ACTUATOR_THRUST_X,
                                  NodeType.ACTUATOR_THRUST_Y, NodeType.ACTUATOR_SIGNAL,
                                  NodeType.ACTUATOR_DEFENSE):
                continue
            
            # 噪声阶段: stability < 0.5
            if node.stability < 0.5 and node.stability >= 0.2:
                noise_std = 1.0 - node.stability  # stability越低，噪声越大
                noise = np.random.normal(0, noise_std)
                node.activation += noise
                node.corrosion_active = True
                stats["noisy_nodes"] += 1
            
            # 坍塌阶段: stability < 0.2
            elif node.stability < 0.2:
                node.corrosion_active = True
                # 1% 概率算子突变
                if np.random.random() < 0.01:
                    self._mutate_operator_type(node)
                    stats["mutated_nodes"] += 1
        
        return stats
    
    def _mutate_operator_type(self, node: Node):
        """
        随机改变算子类型 (坍塌阶段)
        """
        operator_types = [NodeType.ADD, NodeType.MULTIPLY, NodeType.DELAY, NodeType.THRESHOLD]
        
        # 过滤掉当前类型
        available_types = [t for t in operator_types if t != node.node_type]
        if available_types:
            new_type = np.random.choice(available_types)
            node.node_type = new_type
            node.name = f"{new_type.name}_{node.node_id}"
            # 重置延迟缓冲区
            if new_type == NodeType.DELAY:
                
                node.delay_steps = np.random.randint(8, 51)
                node.delay_buffer = deque(maxlen=max(1, node.delay_steps))
                for _ in range(node.delay_steps):
                    node.delay_buffer.append(0.0)
    
    def _mutate_delay_steps(self, node: Node):
        """
         突变DELAY节点的delay_steps
        允许Agent演化出更长的记忆和计划能力
        """
        if node.node_type != NodeType.DELAY:
            return
        
        # 随机增加或减少delay_steps
        if np.random.random() < 0.5:
            # 增加延迟 (最高50帧)
            node.delay_steps = min(50, node.delay_steps + np.random.randint(1, 4))
        else:
            # 减少延迟 (最低4帧)
            node.delay_steps = max(4, node.delay_steps - np.random.randint(1, 3))
        
        # 重建缓冲区
        old_buffer = list(node.delay_buffer) if node.delay_buffer else []
        node.delay_buffer = deque(maxlen=node.delay_steps)
        
        # 保留部分旧数据
        for i, val in enumerate(old_buffer[:min(len(old_buffer), node.delay_steps)]):
            node.delay_buffer.append(val)
        while len(node.delay_buffer) < node.delay_steps:
            node.delay_buffer.append(0.0)
    
    def structural_pruning(self) -> int:
        """
        物理清除: 删除 stability < 0.01 的节点及相连边
        
        返回:
            被删除的节点数
        """
        nodes_to_remove = []
        
        for node_id, node in self.nodes.items():
            # 跳过传感器和端口
            if node.node_type in (NodeType.SENSOR, NodeType.LIGHT_SENSOR,
                                  NodeType.AGENT_RADAR_SENSOR, NodeType.GPS_SENSOR,
                                  NodeType.COMPASS_SENSOR, NodeType.CONSTANT,
                                  NodeType.ACTUATOR, NodeType.PORT_MOTION,
                                  NodeType.PORT_OFFENSE, NodeType.PORT_DEFENSE,
                                  NodeType.PORT_REPAIR, NodeType.PORT_SIGNAL):
                continue
            
            # 跳过传感器和端口
            if node.node_type in (NodeType.SENSOR, NodeType.LIGHT_SENSOR,
                                  NodeType.AGENT_RADAR_SENSOR, NodeType.GPS_SENSOR,
                                  NodeType.COMPASS_SENSOR, NodeType.CONSTANT,
                                  NodeType.ACTUATOR, NodeType.PORT_MOTION,
                                  NodeType.PORT_OFFENSE, NodeType.PORT_DEFENSE,
                                  NodeType.PORT_REPAIR, NodeType.PORT_SIGNAL,
                                  # v13.0: 统一场物理
                                  NodeType.SENSE_EPF_CENTER, NodeType.SENSE_EPF_GRAD_X, NodeType.SENSE_EPF_GRAD_Y,
                                  NodeType.SENSE_KIF_CENTER, NodeType.SENSE_KIF_GRAD_X, NodeType.SENSE_KIF_GRAD_Y,
                                  NodeType.SENSE_ISF_CENTER, NodeType.SENSE_ISF_GRAD_X, NodeType.SENSE_ISF_GRAD_Y,
                                  NodeType.SENSE_ESF_VAL, NodeType.SENSE_INTERNAL_ENERGY,
                                  NodeType.ACTUATOR_PERMEABILITY, NodeType.ACTUATOR_THRUST_X,
                                  NodeType.ACTUATOR_THRUST_Y, NodeType.ACTUATOR_SIGNAL,
                                  NodeType.ACTUATOR_DEFENSE):
                continue
            
            # 检查稳定性阈值
            if node.stability < 0.01:
                nodes_to_remove.append(node_id)
        
        # 删除节点和边
        for node_id in nodes_to_remove:
            # 删除所有相连的边
            self.edges = [e for e in self.edges 
                         if e['source_id'] != node_id and e['target_id'] != node_id]
            # 删除节点
            del self.nodes[node_id]
        
        return len(nodes_to_remove)
    
    def get_info(self) -> Dict:
        """获取基因组的简要信息"""
        node_counts = defaultdict(int)
        for node in self.nodes.values():
            node_counts[node.node_type.name] += 1
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'enabled_edges': sum(1 for e in self.edges if e['enabled']),
            'node_counts': dict(node_counts)
        }
    
    # ============================================================
    # 结构突变方法 (Structural Mutation)
    # ============================================================
    
    def mutate_add_node(self) -> bool:
        """
        随机选择一条现有的边，在中间插入一个全新的算子节点 (ADD 或 MULTIPLY)
        
        过程:
            1. 随机选择一条已启用的边
            2. 禁用该边
            3. 创建新节点 (ADD 或 MULTIPLY)
            4. 添加边: source → new_node 和 new_node → target
        
        返回:
            是否成功执行突变
        """
        # 找已启用的边
        enabled_edges = [e for e in self.edges if e['enabled']]
        if not enabled_edges:
            return False
        
        # 随机选择一条边
        edge = enabled_edges[np.random.randint(len(enabled_edges))]
        
        # 获取新的节点 ID
        new_node_id = max(self.nodes.keys()) + 1 if self.nodes else 0
        
        # 随机选择节点类型 (ADD, MULTIPLY, DELAY, THRESHOLD)
        rand = np.random.random()
        if rand < 0.3:
            node_type = NodeType.ADD
        elif rand < 0.5:
            node_type = NodeType.MULTIPLY
        elif rand < 0.7:
            node_type = NodeType.DELAY
        elif rand < 0.85:
            node_type = NodeType.THRESHOLD  # 阈值算子
        else:
            node_type = NodeType.PREDICTOR  # 预测节点
        
        # ============================================================
        
        # 新节点初始时在数学上等同于"直通"，避免破坏性突变
        # ============================================================
        if node_type == NodeType.DELAY:
            # DELAY: 初始 delay_steps=1（最小延迟），避免杀死救命信号
            delay_steps = 1  # 原来是 np.random.randint(8, 51)
            new_node = Node(node_id=new_node_id, node_type=node_type, 
                          name=f"{node_type.name}_{new_node_id}", delay_steps=delay_steps)
        else:
            new_node = Node(node_id=new_node_id, node_type=node_type, name=f"{node_type.name}_{new_node_id}")
        
        # 添加新节点
        self.add_node(new_node)
        
        # 禁用原边
        edge['enabled'] = False
        
        # ============================================================
        
        # - source → new_node: weight=1.0 (恒等映射)
        # - new_node → target: weight=edge['weight'] (保持原信号强度)
        # 这样新节点初始时不改变信号传递，后续通过 mutate_weight 调节
        # ============================================================
        self.add_edge(edge['source_id'], new_node_id, weight=1.0, enabled=True)
        self.add_edge(new_node_id, edge['target_id'], weight=edge['weight'], enabled=True)
        
        return True
    
    def mutate_add_edge(self, max_attempts: int = 50) -> bool:
        """
        随机选择两个目前没有连接的节点，建立一条新边
        
        规则:
            - 不能连接 ACTUATOR → SENSOR (反馈连接)
            - 不能创建已存在的边
            - 不能创建会导致环的边
        
        参数:
            max_attempts: 最大尝试次数
        
        返回:
            是否成功执行突变
        """
        # 获取可作为源和目标的节点
        source_nodes = [
            n.node_id for n in self.nodes.values() 
            if n.node_type in (NodeType.SENSOR, NodeType.CONSTANT, NodeType.ADD, 
                               NodeType.MULTIPLY, NodeType.DELAY, NodeType.PREDICTOR)
        ]
        target_nodes = [
            n.node_id for n in self.nodes.values() 
            if n.node_type in (NodeType.ADD, NodeType.MULTIPLY, NodeType.ACTUATOR, 
                               NodeType.DELAY, NodeType.THRESHOLD, NodeType.PREDICTOR)
        ]
        
        for _ in range(max_attempts):
            source_id = source_nodes[np.random.randint(len(source_nodes))]
            target_id = target_nodes[np.random.randint(len(target_nodes))]
            
            # 跳过同一节点
            if source_id == target_id:
                continue
            
            # 检查边是否已存在
            existing = any(
                e['source_id'] == source_id and e['target_id'] == target_id 
                for e in self.edges
            )
            if existing:
                continue
            
            # 简单环检测: 检查 target 是否能回溯到 source
            if self._would_create_cycle(source_id, target_id):
                continue
            
            # 添加新边，随机权重
            weight = np.random.randn() * 0.5  # 高斯噪声初始化
            self.add_edge(source_id, target_id, weight=weight, enabled=True)
            return True
        
        return False
    
    def _would_create_cycle(self, source_id: int, target_id: int) -> bool:
        """
        检查添加边是否会导致"非法"环
        
        逻辑更新：允许包含 DELAY 节点的环路
        - 如果搜索路径上经过 DELAY 节点，则该路径不再是"非法"组合环路
        - 这类似于同步数字电路，DELAY 打破信号的即时循环
        
        返回:
            True 如果会创建非法环 (不含 DELAY)
            False 如果允许 (含 DELAY 或无环)
        """
        # 记录搜索路径上是否有 DELAY 节点
        path_has_delay = False
        
        # DFS 检测环，同时追踪是否经过 DELAY
        visited = set()
        path_stack = [(target_id, False)]  # (node_id, has_delay_in_path)
        
        while path_stack:
            node, has_delay = path_stack.pop()
            
            # 先检查当前节点是否是 DELAY (必须在判断环路之前!)
            current_has_delay = has_delay
            if self.nodes[node].node_type == NodeType.DELAY:
                current_has_delay = True
            
            if node == source_id:
                # 如果回到源节点，且路径上没有 DELAY，则是非法环
                if not current_has_delay:
                    return True
                # 有 DELAY，允许这个环路
                continue
            
            if node in visited:
                continue
            visited.add(node)
            
            # 沿边向前查找
            for edge in self.edges:
                if edge['enabled'] and edge['source_id'] == node:
                    path_stack.append((edge['target_id'], current_has_delay))
        
        return False
    
    def mutate_weight(self, sigma: float = 0.1, probability: float = 0.1) -> bool:
        """
        随机挑选几条边，对其权重进行微小的扰动 (高斯噪声)
        
        参数:
            sigma: 高斯噪声的标准差
            probability: 每条边被扰动的概率
        
        返回:
            是否有边被修改
        """
        modified = False
        
        for edge in self.edges:
            if edge['enabled'] and np.random.random() < probability:
                edge['weight'] += np.random.randn() * sigma
                modified = True
        
        return modified
    
    def copy(self) -> 'OperatorGenome':
        """
        深拷贝当前基因组 (用于复制智能体)
        
        注意: 复制时共享InnovationManager以保持ID一致性
        """
        new_genome = OperatorGenome(innovation_manager=self._innovation_mgr)
        
        # 复制节点
        for node in self.nodes.values():
            
            delay_steps = getattr(node, 'delay_steps', 1)
            new_node = Node(
                node_id=node.node_id,
                node_type=node.node_type,
                constant_value=node.constant_value,
                name=node.name,
                delay_steps=delay_steps
            )
            
            if node.angle_weights is not None:
                new_node.angle_weights = node.angle_weights.copy()
                new_node.distance_weights = node.distance_weights.copy()
                new_node.target_type_weights = node.target_type_weights.copy()
            new_genome.add_node(new_node)
        
        # 复制边 (保留创新ID)
        for edge in self.edges:
            new_genome.add_edge(
                source_id=edge['source_id'],
                target_id=edge['target_id'],
                weight=edge['weight'],
                enabled=edge['enabled'],
                innovation_id=edge.get('innovation_id')
            )
            # 复制学习率
            if edge.get('learning_rate', 0) > 0:
                new_genome.edges[-1]['learning_rate'] = edge['learning_rate']
        
        return new_genome
    
    def hebbian_update(self, node_activations: Dict[int, float], lr: float = 0.01):
        """
        赫布规则更新可塑性边
        
        Hebbian rule: Δw = lr * pre_activation * post_activation
        " neurons that fire together, wire together "
        
        参数:
            node_activations: 节点激活值字典 {node_id: activation}
            lr: 学习率系数
        """
        for edge in self.edges:
            if not edge['enabled']:
                continue
            
            # 检查是否可学习
            learning_rate = edge.get('learning_rate', 0)
            if learning_rate <= 0:
                continue
            
            pre_id = edge['source_id']
            post_id = edge['target_id']
            
            # 获取前后节点激活值
            pre_act = node_activations.get(pre_id, 0.0)
            post_act = node_activations.get(post_id, 0.0)
            
            # Hebbian update: w += lr * pre * post
            delta = learning_rate * lr * pre_act * post_act
            edge['weight'] = np.clip(edge['weight'] + delta, -1.0, 1.0)
    
    def mutate_enable_plasticity(self, probability: float = 0.1) -> bool:
        """
        突变: 随机启用某些边的可塑性
        
        参数:
            probability: 启用可塑性的概率
            
        返回:
            是否成功添加可塑性边
        """
        if not self.edges:
            return False
        
        # 随机选择边添加可塑性
        for edge in self.edges:
            if not edge['enabled']:
                continue
            if np.random.random() < probability:
                # 随机学习率 (0.001 ~ 0.1)
                edge['learning_rate'] = np.random.uniform(0.001, 0.1)
        
        return True
    
    def mutate_recursive(self, supernodes: List['SuperNode'] = None) -> bool:
        """
        v4.0: 递归突变 - 允许在"基础算子"和"超级节点"之间连线
        
        这允许大脑长出层级结构:
        - 基础层: SENSOR → DELAY → ACTUATOR
        - 中间层: SuperNode (冻结的功能器官)
        - 顶层: 连接到 SuperNode 的新算子
        
        参数:
            supernodes: 可用的超级节点列表
            
        返回:
            是否成功添加边
        """
        if supernodes is None:
            supernodes = []
        
        # 如果没有超级节点，降级为普通添加边
        if not supernodes:
            return self.mutate_add_edge()
        
        # 决定是添加新边还是重组现有边
        if np.random.random() < 0.5:
            return self._add_supernode_edge(supernodes)
        else:
            return self._restructure_with_supernode(supernodes)
    
    def _add_supernode_edge(self, supernodes: List['SuperNode']) -> bool:
        """添加与超级节点的连线"""
        if not self.nodes or not supernodes:
            return False
        
        # 随机选择: 从基础节点连到 SuperNode, 或从 SuperNode 连出
        direction = np.random.choice(['in', 'out'])
        
        if direction == 'in':
            # 基础节点 → SuperNode
            source_id = np.random.choice(list(self.nodes.keys()))
            target_supernode = np.random.choice(supernodes)
            
            # SuperNode 的输入ID作为目标
            target_id = np.random.choice(target_supernode.input_ids)
            
            # 检查边是否已存在
            if any(e['source_id'] == source_id and e['target_id'] == target_id for e in self.edges):
                return False
            
            weight = np.random.uniform(-1.0, 1.0)
            self.edges.append({
                'source_id': source_id,
                'target_id': target_id,
                'weight': weight,
                'enabled': True,
                'learning_rate': 0.01,
                'plastic': False
            })
            return True
        else:
            # SuperNode → 基础节点
            source_supernode = np.random.choice(supernodes)
            target_id = np.random.choice(list(self.nodes.keys()))
            
            # SuperNode 的输出ID作为源
            source_id = np.random.choice(source_supernode.output_ids)
            
            if any(e['source_id'] == source_id and e['target_id'] == target_id for e in self.edges):
                return False
            
            weight = np.random.uniform(-1.0, 1.0)
            self.edges.append({
                'source_id': source_id,
                'target_id': target_id,
                'weight': weight,
                'enabled': True,
                'learning_rate': 0.01,
                'plastic': False
            })
            return True
    
    def _restructuring_with_supernode(self, supernodes: List['SuperNode']) -> bool:
        """使用超级节点重构现有连接"""
        if len(self.edges) < 2 or not supernodes:
            return False
        
        # 随机选择一条边，尝试替换为超级节点路径
        edge_idx = np.random.randint(0, len(self.edges))
        old_edge = self.edges[edge_idx]
        
        # 选择一个超级节点插入
        supernode = np.random.choice(supernodes)
        
        # 创建两跳路径: source → supernode → target
        new_edges = [
            {
                'source_id': old_edge['source_id'],
                'target_id': supernode.input_ids[0] if supernode.input_ids else old_edge['target_id'],
                'weight': old_edge['weight'] * 0.5,
                'enabled': True,
                'learning_rate': 0.01,
                'plastic': False
            },
            {
                'source_id': supernode.output_ids[0] if supernode.output_ids else old_edge['source_id'],
                'target_id': old_edge['target_id'],
                'weight': old_edge['weight'] * 0.5,
                'enabled': True,
                'learning_rate': 0.01,
                'plastic': False
            }
        ]
        
        # 替换旧边
        self.edges[edge_idx] = new_edges[0]
        self.edges.append(new_edges[1])
        
        return True


# ============================================================
# 3. 智能体 (Agent)
# ============================================================


    # ============================================================
    # JSON 序列化支持
    # ============================================================
    def to_dict(self) -> Dict:
        """导出基因组为字典"""
        return {
            'nodes': [node_to_dict(node) for node in self.nodes.values()],
            'edges': [
                {
                    'source_id': e['source_id'],
                    'target_id': e['target_id'],
                    'weight': e['weight'],
                    'enabled': e['enabled'],
                    'learning_rate': e.get('learning_rate', 0.0),
                    'innovation_id': e.get('innovation_id')
                }
                for e in self.edges
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OperatorGenome':
        """从字典导入基因组"""
        genome = cls()
        
        # 恢复节点
        for node_data in data['nodes']:
            from .node import Node
            from .node import node_from_dict; node = node_from_dict(node_data, Node)
            genome.add_node(node)
        
        # 恢复边
        for edge_data in data['edges']:
            genome.add_edge(
                source_id=edge_data['source_id'],
                target_id=edge_data['target_id'],
                weight=edge_data['weight'],
                enabled=edge_data['enabled'],
                innovation_id=edge_data.get('innovation_id')
            )
            # 恢复学习率
            if len(genome.edges) > 0:
                genome.edges[-1]['learning_rate'] = edge_data.get('learning_rate', 0.0)
        
        return genome
    
    def save_json(self, path: str) -> None:
        """保存为JSON文件"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_json(cls, path: str) -> 'OperatorGenome':
        """从JSON文件加载"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

