from typing import List, Dict, Optional, Tuple, Set
import numpy as np

from .node import Node, NodeType, SuperNode, MacroOperator, SubgraphTracker
from .genome import OperatorGenome
from .agent import Agent
from .environment import Environment, ChunkManager

class SubgraphFreezer:
    """
    子图提取与冻结系统 (v4.1)
    
    机制:
    1. 每隔20代分析top 5%智能体
    2. 提取连接数>3且频繁出现的拓扑结构
    3. 封装为MacroOperator加入算子池
    4. 后续演化可复用这些"功能器官"
    """
    
    def __init__(
        self,
        freeze_interval: int = 20,      # 冻结间隔代数
        top_percent: float = 0.05,       # 分析top 5%
        min_connections: int = 3,        # 最少连接数
        min_frequency: int = 2,          # 最小出现频率
        max_macros: int = 10             # 最多保留的宏算子
    ):
        self.freeze_interval = freeze_interval
        self.top_percent = top_percent
        self.min_connections = min_connections
        self.min_frequency = min_frequency
        self.max_macros = max_macros
        
        # 算子池: 累积的MacroOperator
        self.operator_pool: List[MacroOperator] = []
        
        # 拓扑结构计数器: 边模式 → 出现次数
        self.topology_counts: Dict[Tuple, int] = {}
        
        # 历史记录
        self.freeze_history: List[Dict] = []
    
    def should_freeze(self, generation: int) -> bool:
        """检查是否应该执行冻结"""
        return generation > 0 and generation % self.freeze_interval == 0
    
    def extract_subgraphs(
        self, 
        elite_agents: List['Agent'],
        generation: int,
        avg_food: float = 0.0
    ) -> List[MacroOperator]:
        """
        从精英智能体中提取子图并冻结
        
        v5.1 宏算子准入制:
        - 只有Food_Eaten > avg_food的个体才有资格提取
        - 严禁从"零捕食"个体中提取宏算子
        
        算法:
        1. 过滤: 仅选择捕食成功的个体
        2. 收集拓扑模式
        3. 统计高频模式
        4. 提取连通子图
        5. 封装为MacroOperator
        """
        # v5.1: 宏算子准入制 - 过滤捕食者
        qualifying_agents = [a for a in elite_agents if a.food_eaten > avg_food]
        
        if len(qualifying_agents) < 1:
            return []
        
        # Step 1: 收集拓扑模式 (仅从合格个体)
        all_patterns = []
        for agent in qualifying_agents:
            patterns = self._extract_topology_pattern(agent.genome)
            all_patterns.extend(patterns)
        
        # Step 2: 统计频率
        topology_freq = {}
        for pattern in all_patterns:
            topology_freq[pattern] = topology_freq.get(pattern, 0) + 1
        
        # Step 3: 筛选高频模式
        frequent_patterns = {
            p: c for p, c in topology_freq.items() 
            if c >= self.min_frequency
        }
        
        # Step 4: 提取子图并封装
        new_macros = []
        
        for agent in elite_agents:
            # 提取边
            edges = agent.genome.edges
            nodes = agent.genome.nodes
            
            # 内部节点候选
            internal_types = {
                NodeType.ADD, NodeType.MULTIPLY, NodeType.DELAY,
                NodeType.THRESHOLD, NodeType.PREDICTOR, NodeType.CONSTANT
            }
            
            internal_nodes = [
                nid for nid, n in nodes.items()
                if n.node_type in internal_types
            ]
            
            # 从每个内部节点开始搜索
            for start_id in internal_nodes:
                subgraph_nodes = self._find_connected_nodes(
                    nodes, edges, start_id, depth=2
                )
                
                if len(subgraph_nodes) >= self.min_connections:
                    # 创建MacroOperator
                    macro = self._create_macro(
                        nodes, edges, subgraph_nodes,
                        generation
                    )
                    if macro:
                        new_macros.append(macro)
        
        # 去重
        unique_macros = self._deduplicate_macros(new_macros)
        
        # 添加到算子池
        for macro in unique_macros:
            if len(self.operator_pool) < self.max_macros:
                self.operator_pool.append(macro)
        
        # 记录历史
        self.freeze_history.append({
            'generation': generation,
            'num_macros': len(unique_macros),
            'pool_size': len(self.operator_pool)
        })
        
        return unique_macros
    
    def _extract_topology_pattern(self, genome: 'OperatorGenome') -> List[Tuple]:
        """
        提取拓扑模式 (增强版)
        
        原来只使用 (src_type, tgt_type, weight_sign)
        增强后包含:
        - 源/目标节点类型
        - 权重符号和量化区间 (pos_small/pos_large/neg_small/neg_large)
        - 节点度数信息 (in_degree, out_degree)
        - 局部结构 (是否有残差连接)
        """
        patterns = []
        
        # 预计算节点度数
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        for edge in genome.edges:
            if edge['enabled']:
                out_degree[edge['source_id']] += 1
                in_degree[edge['target_id']] += 1
        
        for edge in genome.edges:
            if not edge['enabled']:
                continue
            
            src_node = genome.nodes[edge['source_id']]
            tgt_node = genome.nodes[edge['target_id']]
            
            src_type = src_node.node_type
            tgt_type = tgt_node.node_type
            
            # 权重量化
            w = edge['weight']
            if w > 0:
                weight_quant = 'pos_small' if abs(w) < 1.0 else 'pos_large'
            else:
                weight_quant = 'neg_small' if abs(w) < 1.0 else 'neg_large'
            
            # 局部结构: 检查是否有残差连接 (目标->源)
            has_residual = any(
                e['enabled'] and e['source_id'] == edge['target_id'] and e['target_id'] == edge['source_id']
                for e in genome.edges
            )
            
            # 创建增强模式
            pattern = (
                src_type,
                tgt_type,
                weight_quant,
                (in_degree[edge['source_id']], out_degree[edge['source_id']]),  # 源度数
                (in_degree[edge['target_id']], out_degree[edge['target_id']]),  # 目标度数
                'residual' if has_residual else 'none'
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _find_connected_nodes(
        self, 
        nodes: Dict[int, Node], 
        edges: List[Dict],
        start_id: int,
        depth: int = 2
    ) -> Dict[int, Node]:
        """BFS查找连通节点 (排除SENSOR/ACTUATOR)"""
        if start_id not in nodes:
            return {}
        
        # 允许的内部节点类型
        internal_types = {
            NodeType.ADD, NodeType.MULTIPLY, NodeType.DELAY,
            NodeType.THRESHOLD, NodeType.PREDICTOR, NodeType.CONSTANT
        }
        
        connected = {start_id: nodes[start_id]}
        queue = [start_id]
        visited = {start_id}
        
        for _ in range(depth):
            new_queue = []
            for node_id in queue:
                # 找所有连接的边
                for edge in edges:
                    if not edge['enabled']:
                        continue
                    
                    # 检查目标节点
                    if edge['source_id'] == node_id and edge['target_id'] not in visited:
                        tgt = edge['target_id']
                        if tgt in nodes:
                            # 只添加内部节点类型
                            if nodes[tgt].node_type in internal_types:
                                connected[tgt] = nodes[tgt]
                                visited.add(tgt)
                                new_queue.append(tgt)
                            else:
                                # 记录边界但不继续扩展
                                visited.add(tgt)
                    
                    # 检查源节点
                    if edge['target_id'] == node_id and edge['source_id'] not in visited:
                        src = edge['source_id']
                        if src in nodes:
                            if nodes[src].node_type in internal_types:
                                connected[src] = nodes[src]
                                visited.add(src)
                                new_queue.append(src)
                            else:
                                visited.add(src)
            
            queue = new_queue
            if not queue:
                break
        
        return connected
    
    def _make_pattern(
        self, 
        nodes: Dict[int, Node], 
        edges: List[Dict],
        subgraph: Dict[int, Node]
    ) -> Tuple:
        """创建子图的模式签名"""
        node_types = tuple(
            nodes[nid].node_type.name for nid in sorted(subgraph.keys())
        )
        
        # 边模式
        edge_patterns = []
        for edge in edges:
            if edge['enabled'] and edge['source_id'] in subgraph and edge['target_id'] in subgraph:
                edge_patterns.append((
                    nodes[edge['source_id']].node_type.name,
                    nodes[edge['target_id']].node_type.name
                ))
        
        return (node_types, tuple(edge_patterns))
    
    def _create_macro(
        self,
        nodes: Dict[int, Node],
        edges: List[Dict],
        subgraph: Dict[int, Node],
        generation: int
    ) -> Optional[MacroOperator]:
        """创建MacroOperator"""
        if len(subgraph) < self.min_connections:
            return None
        
        # 确定输入输出
        input_ids = []
        output_ids = []
        
        subgraph_ids = set(subgraph.keys())
        
        for edge in edges:
            if not edge['enabled']:
                continue
            
            src_in = edge['source_id'] in subgraph_ids
            tgt_in = edge['target_id'] in subgraph_ids
            
            if src_in and not tgt_in:
                output_ids.append(edge['source_id'])
            elif tgt_in and not src_in:
                input_ids.append(edge['target_id'])
        
        if not input_ids or not output_ids:
            return None
        
        # 提取内部边
        internal_edges = [
            e for e in edges 
            if e['enabled'] and e['source_id'] in subgraph_ids and e['target_id'] in subgraph_ids
        ]
        
        # 创建MacroOperator
        node_list = [
            {'id': nid, 'type': n.node_type.name, 'name': n.name}
            for nid, n in subgraph.items()
        ]
        
        return MacroOperator(
            nodes=node_list,
            edges=internal_edges,
            input_ids=list(set(input_ids)),
            output_ids=list(set(output_ids))
        )
    
    def _deduplicate_macros(self, macros: List[MacroOperator]) -> List[MacroOperator]:
        """去重MacroOperator"""
        seen = set()
        unique = []
        
        for macro in macros:
            # 用输入输出作为签名
            sig = (tuple(sorted(macro.input_ids)), tuple(sorted(macro.output_ids)))
            
            if sig not in seen:
                seen.add(sig)
                unique.append(macro)
        
        return unique
    
    def get_pool(self) -> List[MacroOperator]:
        """获取算子池"""
        return self.operator_pool.copy()


# ============================================================
# 6. 内部循环噪声 (Spontaneous Activity)
# ============================================================

class SpontaneousActivity:
    """
    内部循环噪声 (v4.1)
    
    在没有外部输入时，允许算子网络产生自发振荡
    模拟大脑的"默认模式网络"活动
    
    机制:
    1. 当传感器输入低于阈值时触发
    2. 使用随机噪声注入激活网络
    3. 允许网络产生内部模式
    """
    
    def __init__(
        self,
        threshold: float = 0.1,       # 触发阈值
        noise_scale: float = 0.05,    # 噪声强度
        decay: float = 0.95,          # 衰减系数
        enable_prob: float = 0.3      # 启用概率
    ):
        self.threshold = threshold
        self.noise_scale = noise_scale
        self.decay = decay
        self.enable_prob = enable_prob
        
        # 内部活动状态
        self.activity_level: float = 0.0
        self.last_triggered: int = 0
    
    def should_activate(self, sensor_inputs: np.ndarray, current_step: int) -> bool:
        """检查是否应该激活内部活动"""
        # 检查传感器输入是否低于阈值
        mean_input = np.mean(np.abs(sensor_inputs)) if len(sensor_inputs) > 0 else 0.0
        
        if mean_input < self.threshold:
            # 检查是否已经激活过
            if current_step - self.last_triggered > 5:
                self.last_triggered = current_step
                return np.random.random() < self.enable_prob
        
        return False
    
    def generate_noise(self, n_nodes: int) -> Dict[int, float]:
        """生成内部噪声"""
        if self.activity_level < 0.01:
            self.activity_level = self.noise_scale
        
        # 生成稀疏噪声 (只激活部分节点)
        noise = {}
        active_count = max(1, n_nodes // 5)  # 约20%节点
        
        for _ in range(active_count):
            node_id = np.random.randint(0, max(10, n_nodes))
            noise[node_id] = np.random.normal(0, self.activity_level)
        
        # 衰减
        self.activity_level *= self.decay
        
        return noise
    
    def reset(self):
        """重置内部活动"""
        self.activity_level = 0.0


# ============================================================
# 7. 多级预测驱动 (Multi-level Prediction)
# ============================================================

class MultiLevelPredictor:
    """
    多级预测驱动 (v4.1)
    
    要求Agent不仅预测传感器，还要预测内部节点状态
    这强迫网络学习内部表示和自我模型
    
    机制:
    1. 选择若干"预测目标节点"
    2. 要求网络预测这些节点的下一时刻状态
    3. 预测误差加入适应度
    """
    
    def __init__(
        self,
        num_targets: int = 2,         # 预测目标数量
        target_types: List[NodeType] = None,  # 目标节点类型
        prediction_weight: float = 2.0  # 预测误差权重
    ):
        self.num_targets = num_targets
        self.target_types = target_types or [
            NodeType.ADD, 
            NodeType.MULTIPLY, 
            NodeType.DELAY,
            NodeType.THRESHOLD
        ]
        self.prediction_weight = prediction_weight
        
        # 预测目标节点
        self.target_nodes: Dict[int, List[int]] = {}  # agent_id -> [node_ids]
        
        # 预测历史
        self.predictions: Dict[int, Dict[int, float]] = {}  # agent_id -> {node_id: predicted}
    
    def select_targets(self, agent: 'Agent') -> List[int]:
        """为Agent选择预测目标节点"""
        # 筛选指定类型的节点
        candidates = [
            nid for nid, node in agent.genome.nodes.items()
            if node.node_type in self.target_types and node.node_type not in [
                NodeType.SENSOR, NodeType.ACTUATOR, NodeType.CONSTANT
            ]
        ]
        
        if not candidates:
            return []
        
        # 随机选择
        n_select = min(self.num_targets, len(candidates))
        selected = list(np.random.choice(candidates, n_select, replace=False))
        
        self.target_nodes[agent.id] = selected
        return selected
    
    def predict(
        self, 
        agent: 'Agent', 
        current_activations: Dict[int, float]
    ) -> Dict[int, float]:
        """
        预测下一时刻的节点状态
        
        使用简单的预测: y(t+1) = y(t)
        更复杂的实现可以使用网络输出
        """
        predictions = {}
        
        target_ids = self.target_nodes.get(agent.id, [])
        
        for nid in target_ids:
            if nid in current_activations:
                # 简化预测: 下一时刻等于当前
                predictions[nid] = current_activations[nid]
        
        self.predictions[agent.id] = predictions
        return predictions
    
    def calculate_error(
        self, 
        agent: 'Agent', 
        actual_activations: Dict[int, float]
    ) -> float:
        """计算预测误差"""
        predictions = self.predictions.get(agent.id, {})
        
        if not predictions:
            return 0.0
        
        errors = []
        for nid, pred in predictions.items():
            if nid in actual_activations:
                error = (actual_activations[nid] - pred) ** 2
                errors.append(error)
        
        return np.mean(errors) if errors else 0.0
    
    def get_prediction_loss(self, agent: 'Agent') -> float:
        """获取预测损失用于适应度计算"""
        return self.calculate_error(agent, {})


# ============================================================

class Population:
    """
    智能体种群管理器
    
    属性:
        population_size: 种群大小
        agents: 智能体列表
        environment: 生存环境
        generation: 当前代数
        elite_ratio: 精英选择比例
        mutation_rates: 各突变类型的概率
    
    方法:
        epoch(): 运行一个生命周期
        reproduce(): 基于适应度选择和突变生成下一代
    """
    
    def __init__(
        self,
        population_size: int = 100,
        elite_ratio: float = 0.2,
        env_width: float = 100.0,
        env_height: float = 100.0,
        target_pos: Optional[Tuple[float, float]] = None,
        metabolic_alpha: float = 0.05,  # v9.0: 降低代谢
        metabolic_beta: float = 0.05,
        lifespan: int = 500,
        n_food: int = 5,              # 食物数量
        food_energy: float = 30.0,    # v9.0: 减半能量 (30→15)
        respawn_food: bool = True,    # 食物是否重生
        n_walls: int = 0,             # 障碍物数量 (v3.0)
        day_night_cycle: bool = True, # 昼夜循环 (v3.0)
        use_champion: bool = False,   # v4.1: 使用冠军结构初始化
        pure_survival_mode: bool = False,  # v0.74: 纯生存适应度
        # v0.78: 季节系统
        seasonal_cycle: bool = False,       # 启用季节循环
        season_length: int = 50,            # 每季多少帧
        winter_food_multiplier: float = 0.0,  # 冬天食物生成倍率(0=无食物)
        winter_metabolic_multiplier: float = 2.0,  # 冬天代谢倍率
        # v0.80: 红皇后假说
        red_queen: bool = False,            # 启用红皇后(敌对竞争)
        n_rivals: int = 3,                  # 敌对Agent数量
        rival_refresh_interval: int = 40    # 敌对刷新间隔(代)
    ):
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.lifespan = lifespan
        self.generation = 0
        self.use_champion = use_champion
        self.pure_survival_mode = pure_survival_mode  # v0.74
        
        # 突变概率配置
        self.mutation_rates = {
            'add_node': 0.1,      # 添加节点概率
            'add_edge': 0.15,     # 添加边概率
            'mutate_weight': 0.3  # 权重扰动概率
        }
        
        # 创建环境 (EOE v3.0)
        # v0.78: 季节系统参数
        self.seasonal_cycle = seasonal_cycle
        self.season_length = season_length
        self.winter_food_multiplier = winter_food_multiplier
        self.winter_metabolic_multiplier = winter_metabolic_multiplier
        self.current_season = "summer"  # 开始于夏天
        self.season_frame = 0
        
        # v0.80: 红皇后假说
        self.red_queen = red_queen
        self.n_rivals = n_rivals
        self.rival_refresh_interval = rival_refresh_interval
        self.rivals: List['Agent'] = []  # 敌对Agent列表
        self.rival_timer = 0
        
        self.environment = Environment(
            width=env_width,
            height=env_height,
            target_pos=target_pos,
            metabolic_alpha=metabolic_alpha,
            metabolic_beta=metabolic_beta,
            n_food=n_food,
            food_energy=food_energy,
            respawn_food=respawn_food,
            n_walls=n_walls,
            day_night_cycle=day_night_cycle,
            pure_survival_mode=pure_survival_mode,  # v0.74
            # v0.78: 季节参数
            seasonal_cycle=seasonal_cycle,
            season_length=season_length,
            winter_food_multiplier=winter_food_multiplier,
            winter_metabolic_multiplier=winter_metabolic_multiplier
        )
        
        # ============================================================
        # v4.1: 新系统集成
        # ============================================================
        
        # 子图冻结系统
        self.subgraph_freezer = SubgraphFreezer(
            freeze_interval=20,
            top_percent=0.05,
            min_connections=3,
            max_macros=10
        )
        
        # v5.2: 创建深度冷冻的Basic_Navigator宏算子
        self._create_basic_navigator()
    
    def _create_basic_navigator(self) -> None:
        """
        v5.2: 创建深度冷冻的Basic_Navigator宏算子
        
        稳定拓扑:
        - SENSOR(0,1) → ACTUATOR(2,3) 差分驱动
        - ACTUATOR → DELAY → ACTUATOR 记忆回路
        
        这个宏算子将被深度冷冻,禁止突变内部结构
        """
        # 定义Basic_Navigator内部节点
        nodes = [
            {'id': 0, 'type': 'SENSOR', 'name': 'S_L'},
            {'id': 1, 'type': 'SENSOR', 'name': 'S_R'},
            {'id': 2, 'type': 'ACTUATOR', 'name': 'A_L'},
            {'id': 3, 'type': 'ACTUATOR', 'name': 'A_R'},
            {'id': 10, 'type': 'DELAY', 'name': 'DELAY', 'delay_steps': 4},
        ]
        
        edges = [
            {'source_id': 0, 'target_id': 2, 'weight': 1.5},
            {'source_id': 1, 'target_id': 3, 'weight': 1.5},
            {'source_id': 0, 'target_id': 3, 'weight': -1.2},
            {'source_id': 1, 'target_id': 2, 'weight': -1.2},
            {'source_id': 2, 'target_id': 10, 'weight': 1.5},
            {'source_id': 10, 'target_id': 2, 'weight': 0.8},
            {'source_id': 10, 'target_id': 3, 'weight': 0.5},
        ]
        
        # 创建宏算子
        basic_nav = MacroOperator(
            nodes=nodes,
            edges=edges,
            input_ids=[0, 1],  # 传感器输入
            output_ids=[2, 3]  # 执行器输出
        )
        
        # v5.2: 深度冷冻
        basic_nav.is_frozen = True
        basic_nav.name = "Basic_Navigator"
        basic_nav.fitness_contribution = 500.0  # 预设高分
        basic_nav.generation_created = 0
        
        # 加入算子池
        self.subgraph_freezer.operator_pool.append(basic_nav)
        
        # 内部循环噪声
        self.spontaneous_activity = SpontaneousActivity(
            threshold=0.1,
            noise_scale=0.05,
            enable_prob=0.3
        )
        
        # 多级预测驱动
        self.multi_predictor = MultiLevelPredictor(
            num_targets=2,
            prediction_weight=2.0
        )
        
        # 初始化种群
        self.agents: List[Agent] = []
        self._init_population()
    
    def _init_population(self) -> None:
        """初始化第一代种群"""
        # 随机生成目标位置
        self.environment.target_pos = (
            np.random.uniform(10, self.environment.width - 10),
            np.random.uniform(10, self.environment.height - 10)
        )
        
        for i in range(self.population_size):
            # 随机初始位置和朝向
            x = np.random.uniform(10, self.environment.width - 10)
            y = np.random.uniform(10, self.environment.height - 10)
            theta = np.random.uniform(0, 2 * np.pi)
            
            agent = Agent(agent_id=i, x=x, y=y, theta=theta)
            
            # v4.1: 选择初始化方式
            if self.use_champion:
                self._init_champion_brain(agent)
            else:
                self._init_agent_brain(agent)
            
            self.agents.append(agent)
            self.environment.add_agent(agent)
    
    def _init_agent_brain(self, agent: Agent) -> None:
        """
        初始化智能体的脑结构 (修正版趋化性连接)
        
        差分驱动逻辑:
        - 左轮 = k*左传感 - k*右传感 (食物在左时加速左轮)
        - 右轮 = k*右传感 - k*左传感 (食物在右时加速右轮)
        
        这样当食物在左侧时，左轮更快 → 右转朝向食物
        当食物在右侧时，右轮更快 → 左转朝向食物
        """
        # 右侧传感器 → 右轮 (正相关)
        agent.genome.add_edge(source_id=1, target_id=3, weight=1.2)
        # 左侧传感器 → 右轮 (负相关，差分)
        agent.genome.add_edge(source_id=0, target_id=3, weight=-0.8)
        
        # 左侧传感器 → 左轮 (正相关)
        agent.genome.add_edge(source_id=0, target_id=2, weight=1.2)
        # 右侧传感器 → 左轮 (负相关，差分)
        agent.genome.add_edge(source_id=1, target_id=2, weight=-0.8)
    
    def _init_champion_brain(self, agent: Agent) -> None:
        """
        使用v4.1冠军结构初始化脑 (最佳记忆回路)
        
        包含:
        - 差分驱动连接
        - DELAY记忆回路
        - 预测回路
        """
        # 先添加额外节点
        # DELAY节点 (id=10)
        if 10 not in agent.genome.nodes:
            delay_node = Node(10, NodeType.DELAY, delay_steps=4, name='DELAY')
            agent.genome.add_node(delay_node)
        
        # PREDICTOR节点 (id=4,5,9)
        for nid, name in [(4, 'P1'), (5, 'P2'), (9, 'P3')]:
            if nid not in agent.genome.nodes:
                pred_node = Node(nid, NodeType.PREDICTOR, name=name)
                agent.genome.add_node(pred_node)
        
        # THRESHOLD节点 (id=7,8)
        for nid, name in [(7, 'T1'), (8, 'T2')]:
            if nid not in agent.genome.nodes:
                thresh_node = Node(nid, NodeType.THRESHOLD, name=name)
                agent.genome.add_node(thresh_node)
        
        # 差分驱动 (使用现有节点: 0,1=传感器, 2=ADD, 3,4=执行器)
        # v0.74修复: 使用正确的趋化性连接（正相关）
        # 原理：食物在右 → 右传感器高 → 右执行器强 → 右轮快 → 左转朝向食物
        agent.genome.add_edge(source_id=0, target_id=2, weight=1.5)   # S_L -> A_L (正)
        agent.genome.add_edge(source_id=1, target_id=3, weight=1.5)   # S_R -> A_R (正)
        agent.genome.add_edge(source_id=0, target_id=3, weight=-0.5)  # S_L -> A_R (差分)
        agent.genome.add_edge(source_id=1, target_id=2, weight=-0.5)  # S_R -> A_L (差分)
        
        # DELAY记忆回路 (关键!)
        agent.genome.add_edge(source_id=2, target_id=10, weight=1.5)
        agent.genome.add_edge(source_id=10, target_id=2, weight=0.8)
        agent.genome.add_edge(source_id=10, target_id=3, weight=0.8)  # DELAY辅助A_L
        agent.genome.add_edge(source_id=10, target_id=4, weight=0.5)  # DELAY辅助A_R
        
        # 预测回路
        agent.genome.add_edge(source_id=4, target_id=5, weight=0.3)
        agent.genome.add_edge(source_id=5, target_id=7, weight=-0.4)
        agent.genome.add_edge(source_id=7, target_id=4, weight=1.0)
    
    def _compute_genomic_distance(self, g1: 'OperatorGenome', g2: 'OperatorGenome') -> float:
        """
        v5.5 NEAT风格: 计算两个基因组的遗传距离 δ
        
        δ = (c1 * E + c2 * D) / N + c3 * W̄ + c4 * T̄
        
        E: Excess (多余基因) - 在创新ID范围外的基因
        D: Disjoint (离散基因) - 在范围内但只在一个中存在的基因
        N: 较大基因组的基因总数 (归一化)
        W̄: 同源连接的权重平均偏差
        T̄: 同源节点的算子类型偏差
        
        权重系数: c1=1.0, c2=1.0, c3=0.4, c4=1.0
        """
        c1, c2, c3, c4 = 1.0, 1.0, 0.4, 1.0
        
        # 获取节点创新ID集合
        nodes1 = {n.innovation_id: n for n in g1.nodes.values()}
        nodes2 = {n.innovation_id: n for n in g2.nodes.values()}
        
        innovation_ids1 = set(nodes1.keys())
        innovation_ids2 = set(nodes2.keys())
        
        max_innovation1 = max(innovation_ids1) if innovation_ids1 else 0
        max_innovation2 = max(innovation_ids2) if innovation_ids2 else 0
        max_genome = max(max_innovation1, max_innovation2)
        
        # Excess: 在较大ID范围内的多余基因
        excess1 = [i for i in innovation_ids1 if i > max_innovation2]
        excess2 = [i for i in innovation_ids2 if i > max_innovation1]
        E = len(excess1) + len(excess2)
        
        # Disjoint: 在两个ID范围内的离散基因
        min_geo = min(max_innovation1, max_innovation2)
        disjoint1 = [i for i in innovation_ids1 if i <= min_geo and i not in innovation_ids2]
        disjoint2 = [i for i in innovation_ids2 if i <= min_geo and i not in innovation_ids1]
        D = len(disjoint1) + len(disjoint2)
        
        # N: 较大基因组的基因数
        N = max(len(nodes1), len(nodes2))
        
        # 同源节点 (Matching): 在两个基因组中都存在
        matching_ids = innovation_ids1.intersection(innovation_ids2)
        
        # 权重差异 W̄: 同源连接的权重偏差
        edges1 = {e['innovation_id']: e for e in g1.edges if e.get('innovation_id')}
        edges2 = {e['innovation_id']: e for e in g2.edges if e.get('innovation_id')}
        
        matching_edges = set(edges1.keys()).intersection(set(edges2.keys()))
        
        weight_diff_sum = 0.0
        type_diff_sum = 0.0
        
        if matching_ids or matching_edges:
            # 节点类型偏差 T̄
            for inn_id in matching_ids:
                n1 = nodes1[inn_id]
                n2 = nodes2[inn_id]
                if n1.node_type != n2.node_type:
                    type_diff_sum += 1.0
            
            # 边权重差异
            for inn_id in matching_edges:
                e1 = edges1[inn_id]
                e2 = edges2[inn_id]
                weight_diff_sum += abs(e1['weight'] - e2['weight'])
        
        # 归一化
        n_match = max(len(matching_ids), 1)
        n_edge_match = max(len(matching_edges), 1)
        
        W_bar = weight_diff_sum / n_edge_match if matching_edges else 0.0
        T_bar = type_diff_sum / n_match if matching_ids else 0.0
        
        # 最终距离计算
        if N == 0:
            return 0.0
        
        delta = (c1 * E + c2 * D) / N + c3 * W_bar + c4 * T_bar
        
        return min(delta, 10.0)  # 限制上限
    
    def _assign_species(self) -> None:
        """
        v5.4 物种形成: 根据基因组距离划分物种
        
        - 计算每个个体与已有物种原型的距离
        - 如果距离 > threshold, 创建新物种
        - 分配生态位类型 (草食/肉食/拾荒/通用)
        """
        species_threshold = 1.5  # v5.5 差异阈值 (更宽松,允许更多物种)
        species_prototypes: List[Agent] = []  # 物种原型
        
        for agent in self.agents:
            if not agent.is_alive:
                continue
            
            # 寻找最相似的物种
            best_species = 0
            best_distance = float('inf')
            
            for i, prototype in enumerate(species_prototypes):
                dist = self._compute_genomic_distance(agent.genome, prototype.genome)
                if dist < best_distance:
                    best_distance = dist
                    best_species = i
            
            if best_distance < species_threshold and species_prototypes:
                # 加入现有物种
                agent.species_id = best_species
                agent.genome_distance = best_distance
            else:
                # 创建新物种
                agent.species_id = len(species_prototypes)
                agent.genome_distance = 0.0
                species_prototypes.append(agent)
            
            # 根据结构特征确定生态位
            self._assign_niche(agent)
    
    def _assign_niche(self, agent: Agent) -> None:
        """
        v5.4 生态位分配: 根据节点类型确定生态位
        
        草食 (herbivore): 高PREDICTOR, 低战斗节点
        肉食 (carnivore): NODE_BITE/BOOST 活跃
        拾荒 (scavenger): 极简结构, 低代谢
        通用 (general): 混合结构
        """
        # v6.0 GAIA: 生态位由端口偏好决定 (已在 _compute_niche_preference 中计算)
        # 这里保留兼容逻辑
        genome = agent.genome
        
        n_predictor = sum(1 for n in genome.nodes.values() if n.node_type == NodeType.PREDICTOR)
        n_delay = sum(1 for n in genome.nodes.values() if n.node_type == NodeType.DELAY)
        
        # 端口偏好已在 _compute_niche_preference 中计算
        # 保留简单fallback逻辑
        if agent.niche_type != "general":
            pass  # 已在端口计算中确定
        elif n_predictor >= 3:
            agent.niche_type = "herbivore"
        elif len(genome.nodes) <= 8 and n_delay >= 1:
            agent.niche_type = "scavenger"
    
    def epoch(self, verbose: bool = False) -> Dict:
        """
        运行一个生命周期 (一代智能体的一生)
        
        参数:
            verbose: 是否打印详细信息
        
        返回:
            统计信息字典
        """
        # v0.80: 红皇后 - 第一代生成敌对
        if self.red_queen and self.generation == 0:
            self.spawn_rivals()
        
        for step in range(self.lifespan):
            # v0.80: 红皇后 - 敌对与正常Agent一起竞争
            if self.red_queen:
                self.environment.agents = self.agents + self.rivals
            else:
                self.environment.agents = self.agents
            
            self.environment.step()
            
            # ============================================================
            # v4.1: 内部循环噪声 (当传感器输入低时)
            # ============================================================
            for agent in self.agents:
                if not agent.is_alive:
                    continue
                
                # 获取传感器输入
                sensor_inputs = agent.last_sensor_inputs if agent.last_sensor_inputs is not None else np.array([0.0])
                
                if self.spontaneous_activity.should_activate(sensor_inputs, step):
                    # 获取内部节点数量
                    n_internal = len([
                        n for n in agent.genome.nodes.values()
                        if n.node_type not in [NodeType.SENSOR, NodeType.ACTUATOR]
                    ])
                    
                    # 生成噪声并注入
                    noise = self.spontaneous_activity.generate_noise(n_internal)
                    
                    # 将噪声添加到内部节点
                    for node_id, noise_val in noise.items():
                        if node_id in agent.genome.nodes:
                            agent.genome.nodes[node_id].activation += noise_val
            
            # ============================================================
            # v4.1: 多级预测驱动 (选择目标节点)
            # ============================================================
            # 在第一步选择预测目标
            if step == 0:
                for agent in self.agents:
                    if agent.is_alive:
                        self.multi_predictor.select_targets(agent)
            
            # 计算内部节点预测误差
            for agent in self.agents:
                if not agent.is_alive:
                    continue
                
                # 获取当前激活值
                current_activations = {
                    nid: n.activation 
                    for nid, n in agent.genome.nodes.items()
                    if n.node_type not in [NodeType.SENSOR, NodeType.ACTUATOR]
                }
                
                # 计算预测误差
                if step > 0:
                    pred_error = self.multi_predictor.calculate_error(
                        agent, 
                        current_activations
                    )
                    
                    # 预测误差加入适应度 (负向惩罚)
                    agent.fitness -= pred_error * self.multi_predictor.prediction_weight
            
            if verbose and step % 100 == 0:
                best_agent = max(self.agents, key=lambda a: a.fitness)
                print(f"  Generation {self.generation}, Step {step}: "
                      f"best_fitness={best_agent.fitness:.2f}, "
                      f"nodes={best_agent.genome.get_info()['total_nodes']}, "
                      f"edges={best_agent.genome.get_info()['enabled_edges']}")
        
        # 统计信息
        fitnesses = [a.fitness for a in self.agents]
        best_idx = np.argmax(fitnesses)
        
        # ============================================================
        # Novelty Search: 计算独特度并加入适应度
        # ============================================================
        novelty_weight = 0.5       # 独特度权重
        exploration_weight = 0.3   # 探索奖励权重 (提高以防止极简陷阱)
        
        # 计算每个 agent 的 novelty
        for agent in self.agents:
            novelty = agent.compute_novelty(self.agents, k=3)
            # 探索奖励: 到访更多独特区域
            exploration = agent.get_exploration_score()
            
            # v5.1: 适应度硬化 - 饥饿开关
            # 只有吃到食物的个体才能获得探索/新颖度加分
            if agent.food_eaten > 0:
                agent.fitness = agent.fitness + novelty * novelty_weight + exploration * exploration_weight
            else:
                # 饥饿模式: 无探索奖励,只扣代谢
                pass  # 保持原有fitness (只有代谢惩罚)
        
        # 重新计算统计
        fitnesses = [a.fitness for a in self.agents]
        best_idx = np.argmax(fitnesses)
        
        return {
            'generation': self.generation,
            'best_fitness': fitnesses[best_idx],
            'avg_fitness': np.mean(fitnesses),
            'worst_fitness': min(fitnesses),
            'best_agent': self.agents[best_idx],
            'avg_nodes': np.mean([a.genome.get_info()['total_nodes'] for a in self.agents]),
            'avg_edges': np.mean([a.genome.get_info()['enabled_edges'] for a in self.agents]),
            'avg_novelty': np.mean([a.novelty_score for a in self.agents]),
            'avg_exploration': np.mean([a.get_exploration_score() for a in self.agents])
        }
    
    # ============================================================
    # v0.80: 红皇后假说 - 敌对Agent竞争
    # ============================================================
    
    def spawn_rivals(self) -> None:
        """从精英Agent生成敌对"""
        if not self.red_queen:
            return
        
        self.rivals = []
        
        # 按适应度排序，选择精英
        sorted_agents = sorted(self.agents, key=lambda a: a.fitness, reverse=True)
        elites = sorted_agents[:max(1, self.n_rivals)]
        
        for i in range(self.n_rivals):
            # 使用精英脑结构
            elite = elites[i % len(elites)]
            import copy
            rival = copy.deepcopy(elite)
            
            # 重新设置敌对属性
            rival.id = f"rival_{self.generation}_{i}"
            rival.x = np.random.uniform(10, self.environment.width - 10)
            rival.y = np.random.uniform(10, self.environment.height - 10)
            rival.theta = np.random.uniform(0, 2 * np.pi)
            rival.internal_energy = rival.max_energy
            rival.is_alive = True
            rival.food_eaten = 0
            rival.food_carried = 0
            rival.food_stored = 0
            rival.steps_alive = 0
            
            self.rivals.append(rival)
    
    def _update_rivals(self) -> None:
        """更新敌对Agent (与正常Agent竞争)"""
        if not self.red_queen:
            return
        
        # 敌对也参与环境交互
        self.environment.agents = self.agents + self.rivals
        
        for rival in self.rivals:
            if not rival.is_alive:
                continue
            
            # 敌对执行一步
            self.environment.step()
    
    def refresh_rivals_if_needed(self) -> None:
        """定时刷新敌对"""
        if not self.red_queen:
            return
        
        self.rival_timer += 1
        if self.rival_timer >= self.rival_refresh_interval:
            self.rival_timer = 0
            self.spawn_rivals()
    
    def reproduce(self, verbose: bool = False) -> None:
        """
        基于适应度和食物获取选择生成下一代 (马尔萨斯竞速)
        
        过程:
            1. 按适应度排序
            2. 选择 top 3% 作为精英持久化 (零突变 - Elitism Persistence)
            3. 下一个 7% 作为微突变精英 (仅权重扰动)
            4. 其余 90% 正常突变
            5. 精英复制并应用突变
            6. 补充到原始种群大小
            7. 吃到食物的精英有额外概率复制 (能量奖励)
        
        v5.0 精英持久化:
            - Top 1-3% 个体完全保留,零突变
            - 核心拓扑作为"底座",后续突变仅在外围增加
        """
        # ============================================================
        # v5.3 Battle Royale: 多目标适应度计算 (Pareto)
        # Fitness = w1 * Predation_Success + w2 * Survival_Time - w3 * Metabolic_Waste
        # ============================================================
        w1, w2, w3 = 2.0, 1.0, 0.5  # 权重
        
        for agent in self.agents:
            # 计算多目标适应度
            predation = agent.food_eaten * 10.0  # 捕食成功
            survival = agent.steps_alive * 0.5   # 生存时间
            metabolic = agent.metabolic_waste    # 代谢浪费
            
            # 战斗奖励
            battle_bonus = agent.battle_wins * 5.0 - agent.battle_losses * 2.0
            
            # Epic Motif 保护
            epic_bonus = 20.0 if agent.is_epic_motif else 0.0
            
            # 组合适应度
            agent.fitness = w1 * predation + w2 * survival - w3 * metabolic + battle_bonus + epic_bonus
        
        # 按适应度排序 (降序)
        sorted_agents = sorted(self.agents, key=lambda a: a.fitness, reverse=True)
        
        # ============================================================
        # v5.3 Battle Royale: 冷酷选择 - 末位50%淘汰
        # ============================================================
        n_survivors = max(2, self.population_size // 2)  # 只保留50%
        sorted_agents = sorted_agents[:n_survivors]
        
        # 存活且吃到食物的 agent 有更高优先级
        survivors_with_food = [a for a in sorted_agents if a.is_alive and a.food_eaten > 0]
        
        # v6.0 GAIA: 精英选择时检查繁衍条件
        # 条件: Energy > 200% AND Age in [20%, 60%]
        def can_reproduce(agent: Agent) -> bool:
            age_pct = (agent.age / agent.max_age) * 100 if agent.max_age > 0 else 0
            energy_pct = (agent.internal_energy / agent.initial_energy) * 100 if agent.initial_energy > 0 else 0
            return (energy_pct > 200 and 20 <= age_pct <= 60)
        
        # 筛选可繁衍的精英
        reproducibles = [a for a in survivors_with_food if can_reproduce(a)]
        
        # 精英选择: 优先选择可繁衍的,其次是有食物的
        n_elites = int(self.population_size * self.elite_ratio)
        
        if reproducibles:
            elite_pool = reproducibles[:min(n_elites * 2, len(reproducibles))]
        elif survivors_with_food:
            elite_pool = survivors_with_food[:min(n_elites * 2, len(survivors_with_food))]
        else:
            elite_pool = sorted_agents[:n_elites]
        
        elites = elite_pool[:n_elites]
        
        # ============================================================
        # v5.0: 精英持久化分层
        # ============================================================
        n_total = self.population_size
        n_elite_persistent = max(1, int(n_total * 0.03))  # Top 3% - 零突变
        n_micro_mutation = max(1, int(n_total * 0.07))    # Next 7% - 微突变
        
        # 保存精英持久化个体的基因组 (深拷贝)
        persistent_genomes = []
        for i in range(min(n_elite_persistent, len(elites))):
            persistent_genomes.append(elites[i].genome.copy())
        
        # 保存微突变个体的基因组
        micro_mutation_genomes = []
        for i in range(n_elite_persistent, min(n_elite_persistent + n_micro_mutation, len(elites))):
            micro_mutation_genomes.append(elites[i].genome.copy())
        
        if verbose:
            food_stats = f", Food eaten: {sum(a.food_eaten for a in self.agents)}"
            print(f"  Elites: {n_elites}, Best fitness: {elites[0].fitness:.2f}{food_stats}")
        
        # 生成新一代
        new_agents: List[Agent] = []
        
        for i in range(self.population_size):
            # ============================================================
            # v5.0: 精英持久化分层突变
            # ============================================================
            if i < n_elite_persistent and i < len(persistent_genomes):
                # Tier 1: 精英持久化 - 零突变 (直接保留)
                child_genome = persistent_genomes[i].copy()
                parent_food = elites[i].food_eaten if i < len(elites) else 0
            elif i < n_elite_persistent + n_micro_mutation and len(micro_mutation_genomes) > 0 and (i - n_elite_persistent) < len(micro_mutation_genomes):
                # Tier 2: 微突变 - 仅权重小扰动
                child_genome = micro_mutation_genomes[i - n_elite_persistent].copy()
                parent_food = elites[min(i, len(elites)-1)].food_eaten
                # 仅应用微权重突变
                self._apply_micro_mutations(child_genome)
            else:
                # Tier 3: 正常突变 - 轮盘赌选择
                # v修复6: 适应度归一化 - 避免负权重导致选择崩溃
                fitnesses = np.array([e.fitness for e in elite_pool])
                
                # 将负适应度映射到正区间 (min-max归一化)
                min_fit = fitnesses.min()
                max_fit = fitnesses.max()
                
                if max_fit > min_fit:
                    # 归一化到 [0.1, 1.0] 避免零权重
                    weights = 0.1 + 0.9 * (fitnesses - min_fit) / (max_fit - min_fit)
                else:
                    # 所有适应度相同，使用均匀分布
                    weights = np.ones(len(fitnesses))
                
                # 轮盘赌选择
                parent = elite_pool[np.random.choice(len(elite_pool), p=weights/weights.sum())]
                child_genome = parent.genome.copy()
                parent_food = parent.food_eaten
                
                # 应用完整突变
                self._apply_mutations(child_genome)
                
                # 10% 概率启用可塑性边 (赫布学习)
                if np.random.random() < 0.1:
                    child_genome.mutate_enable_plasticity(probability=0.3)
            
            # v0.74: 元节点压缩 - 如果父节点共同激活>=10，压缩为META_NODE
            # 尝试获取parent (在Tier 3中定义)
            try:
                parent_coact = getattr(parent, 'node_coactivation', {})
                if parent_coact:
                    max_coact = max(parent_coact.values())
                    if max_coact >= 10:
                        # 找到共同激活最多的节点对
                        best_pair = max(parent_coact.items(), key=lambda x: x[1])[0]
                        
                        # 创建META_NODE
                        from eoe.node import Node, NodeType
                        new_node_id = max(child_genome.nodes.keys()) + 1
                        meta_node = Node(new_node_id, NodeType.META_NODE, name='META')
                        child_genome.add_node(meta_node)
                        child_genome.add_edge(best_pair[0], new_node_id, weight=1.0)
                        child_genome.add_edge(best_pair[1], new_node_id, weight=1.0)
            except NameError:
                pass  # Tier 1或2中没有parent变量
            
            # 创建子代智能体
            x = np.random.uniform(10, self.environment.width - 10)
            y = np.random.uniform(10, self.environment.height - 10)
            theta = np.random.uniform(0, 2 * np.pi)
            
            child = Agent(agent_id=i, x=x, y=y, theta=theta)
            child.genome = child_genome
            
            # 能量继承: 父母吃到的食物转化为子代初始能量
            # v5.7 物理规律: 初始能量 = 基础50 + 食物转化
            child.initial_energy = 50.0 + parent_food * self.environment.food_energy
            child.internal_energy = min(child.initial_energy, child.max_energy)
            # 注意: initial_energy 已经包含食物转化,无需重复计入 energy_gained
            
            new_agents.append(child)
        
        # 随机化目标位置
        self.environment.target_pos = (
            np.random.uniform(10, self.environment.width - 10),
            np.random.uniform(10, self.environment.height - 10)
        )
        
        # 更新种群
        self.agents = new_agents
        self.environment.agents = new_agents
        self.generation += 1
        
        # v0.80: 红皇后 - 定时刷新敌对
        if self.red_queen:
            self.refresh_rivals_if_needed()
        
        # 应用动态环境压力 (Malthusian Trap)
        self.environment.apply_dynamic_pressure(self.generation)
    
    # ============================================================
    # v0.76: 红皇后假说 - 动态敌对刷新
    # ============================================================
    def add_rivals(self, n_rivals: int = 3, use_buff: bool = True) -> None:
        """添加敌对Agent (使用精英脑结构 + 可选增强)"""
        # v0.76: 初始化红皇后参数
        if not hasattr(self, 'rivals'):
            self.rivals = []
        if not hasattr(self, 'rival_refresh_interval'):
            self.rival_refresh_interval = 20  # 每20代刷新
        if not hasattr(self, 'rival_buff_level'):
            self.rival_buff_level = 1.0  # 敌对buff等级
        
        # 找到当前最优秀的脑
        if self.agents:
            best_genome = max(self.agents, key=lambda a: a.fitness).genome
            
            for i in range(n_rivals):
                from eoe.agent import Agent
                rival = Agent(
                    agent_id=10000 + len(self.rivals) + i,
                    x=np.random.uniform(10, self.environment.width - 10),
                    y=np.random.uniform(10, self.environment.height - 10),
                    theta=np.random.uniform(0, 2 * np.pi)
                )
                rival.genome = best_genome.copy()
                rival.is_rival = True
                rival.fitness = 0
                
                # v0.76: 红皇后buff - 随等级增强
                if use_buff:
                    rival.rival_buff = self.rival_buff_level
                else:
                    rival.rival_buff = 1.0
                
                self.rivals.append(rival)
            
            # 混合到环境中
            all_agents = self.agents + self.rivals
            self.environment.agents = all_agents
    
    def refresh_rivals(self, n_rivals: int = 3) -> None:
        """刷新敌对Agent (从优秀敌对中选择)"""
        self.remove_rivals()
        
        # v0.77: 红皇后协同进化 - 敌对也在进化!
        # 找到最优秀的敌对，保留其脑结构
        if hasattr(self, 'rivals') and self.rivals:
            # 按适应度排序，保留最强的
            best_rivals = sorted(self.rivals, key=lambda a: a.fitness, reverse=True)[:n_rivals]
            
            # 使用最强敌对的脑结构
            best_rival_genome = best_rivals[0].genome if best_rivals else None
        else:
            best_rival_genome = None
        
        # 如果没有好敌对，用正常精英
        if best_rival_genome is None and self.agents:
            best_rival_genome = max(self.agents, key=lambda a: a.fitness).genome
        
        # 重新生成敌对 (使用进化后的脑)
        if best_rival_genome:
            self.rivals = []
            for i in range(n_rivals):
                from eoe.agent import Agent
                rival = Agent(
                    agent_id=10000 + i,
                    x=np.random.uniform(10, self.environment.width - 10),
                    y=np.random.uniform(10, self.environment.height - 10),
                    theta=np.random.uniform(0, 2 * np.pi)
                )
                rival.genome = best_rival_genome.copy()
                # 添加少量突变让敌对也进化
                self._apply_mutations(rival.genome)
                rival.is_rival = True
                rival.fitness = 0
                rival.rival_buff = 1.0  # 不使用buff
                self.rivals.append(rival)
        
        # 混合到环境中
        all_agents = self.agents + self.rivals
        self.environment.agents = all_agents
        
        print(f'  [RedQueen] 敌对协同进化!')
    
    def remove_rivals(self) -> None:
        """移除敌对Agent"""
        if hasattr(self, 'rivals'):
            self.rivals = []
            self.environment.agents = self.agents
    
    def _apply_mutations(self, agent_or_genome) -> None:
        """对智能体应用随机突变 (支持Agent或Genome)
        
        v5.1: 模拟退火突变 (Annealed Mutation)
        - 随着节点数增加,突变概率指数下降
        - μ_local = μ_0 * e^(-k * N_nodes)
        - 大脑越大,核心越稳定,变异仅在边缘
        """
        # 获取genome对象
        genome = getattr(agent_or_genome, 'genome', agent_or_genome)
        
        # 计算模拟退火因子
        n_nodes = len(genome.nodes)
        k = 0.05  # 衰减系数
        anneal_factor = np.exp(-k * n_nodes)  # 节点越多,因子越小
        
        # 基础突变率
        base_add_node = self.mutation_rates['add_node']
        base_add_edge = self.mutation_rates['add_edge']
        
        # 模拟退火后的实际突变率
        annealed_add_node = base_add_node * anneal_factor
        annealed_add_edge = base_add_edge * anneal_factor
        
        # 添加节点突变 (边缘概率较高)
        if n_nodes < 8 and np.random.random() < annealed_add_node:
            genome.mutate_add_node()
        elif n_nodes >= 8 and np.random.random() < annealed_add_node * 0.5:
            # 大脑较大时,节点添加更保守
            genome.mutate_add_node()
        
        # 添加边突变
        if np.random.random() < annealed_add_edge:
            genome.mutate_add_edge()
        
        # 权重扰动 (相对稳定,稍微衰减)
        weight_rate = self.mutation_rates['mutate_weight'] * (0.7 + 0.3 * anneal_factor)
        if np.random.random() < weight_rate:
            genome.mutate_weight(sigma=0.1 * anneal_factor, probability=0.2)
        
        # v0.81: 传感器感知权重突变
        from .node import NodeType
        for node in genome.nodes.values():
            if node.node_type == NodeType.SENSOR and np.random.random() < 0.2:
                node.mutate_sensor_weights(sigma=0.1 * anneal_factor)
    
    def _apply_micro_mutations(self, genome: 'OperatorGenome') -> None:
        """
        v5.0: 微突变 - 仅权重小扰动,不改变拓扑
        
        用于精英持久化的第二层,允许微小调适但保持核心结构
        """
        # 仅权重微扰动 (sigma=0.05, 仅为正常的50%)
        if np.random.random() < 0.3:  # 30%概率
            genome.mutate_weight(sigma=0.05, probability=0.1)
        
        # 极低概率的节点添加 (仅当基因很小时)
        if len(genome.nodes) < 4 and np.random.random() < 0.02:
            genome.mutate_add_node()
    
    def _check_operator_extinction(self, verbose: bool = False) -> None:
        """
        v5.0: 算子灭绝机制
        
        检查并移除低效 MacroOperator:
        - 连续50代未使用
        - 连续低使用(usage < 5)超过50代
        - 平均适应度下降超过50%
        """
        if not hasattr(self, 'subgraph_freezer'):
            return
        
        pool = self.subgraph_freezer.operator_pool
        if not pool:
            return
        
        extinct_count = 0
        surviving = []
        
        for macro in pool:
            if macro.should_extinct(self.generation):
                extinct_count += 1
                if verbose:
                    print(f"  [Extinction] 移除 MacroOperator #{macro.id} "
                          f"(usage={macro.usage_count}, created={macro.generation_created})")
            else:
                surviving.append(macro)
        
        if extinct_count > 0:
            self.subgraph_freezer.operator_pool = surviving
            if verbose:
                print(f"  [Extinction] 算子池: {len(pool)} -> {len(surviving)} (移除{extinct_count}个)")
    
    def run(
        self, 
        n_generations: int = 10, 
        verbose: bool = True
    ) -> List[Dict]:
        """
        运行多代演化
        
        参数:
            n_generations: 演化代数
            verbose: 是否打印详细信息
        
        返回:
            每代的统计信息列表
        """
        history = []
        
        for gen in range(n_generations):
            if verbose:
                print(f"\n=== Generation {self.generation} ===")
            
            # ============================================================
            # v4.1: 课程学习 - 更新闪烁难度
            # ============================================================
            self.environment.update_blink_difficulty(self.generation)
            
            # 运行一代
            stats = self.epoch(verbose=verbose)
            history.append(stats)
            
            if verbose:
                print(f"  Best Fitness: {stats['best_fitness']:.2f}")
                print(f"  Avg Fitness:  {stats['avg_fitness']:.2f}")
                print(f"  Avg Nodes:    {stats['avg_nodes']:.1f}")
                print(f"  Avg Edges:    {stats['avg_edges']:.1f}")
            
            # ============================================================
            # v4.1: 子图冻结 (每20代)
            # ============================================================
            if self.subgraph_freezer.should_freeze(self.generation + 1):
                # 获取top 5%精英
                sorted_agents = sorted(
                    self.agents, 
                    key=lambda a: a.fitness, 
                    reverse=True
                )
                n_elite = max(2, int(len(sorted_agents) * 0.05))
                elite_agents = sorted_agents[:n_elite]
                
                # v5.1: 计算平均食物获取量
                avg_food = np.mean([a.food_eaten for a in self.agents])
                
                # 提取并冻结子图 (仅从捕食成功的个体)
                new_macros = self.subgraph_freezer.extract_subgraphs(
                    elite_agents, 
                    self.generation + 1,
                    avg_food=avg_food
                )
                
                # v5.0: 为新宏算子设置创建代
                for macro in new_macros:
                    macro.generation_created = self.generation + 1
                    macro.fitness_contribution = elite_agents[0].fitness if elite_agents else 0
                
                if new_macros and verbose:
                    print(f"  [Freezer] 新增 {len(new_macros)} 个MacroOperator")
                    print(f"  [Freezer] 算子池大小: {len(self.subgraph_freezer.operator_pool)}")
                
                # v5.0: 算子灭绝检查
                self._check_operator_extinction(verbose=verbose)
            
            # 演化新一代
            if gen < n_generations - 1:
                self.reproduce(verbose=verbose)
        
        return history


# ============================================================
# 便捷函数
# ============================================================

def create_simple_agent(
    agent_id: int = 0,
    x: float = 50.0,
    y: float = 50.0,
    theta: float = 0.0,
    connection_type: str = "chemotaxis"
) -> Agent:
    """
    创建一个带有默认连接的智能体
    
    连接类型:
        - "direct": 左→左，右→右 (简单直连)
        - "chemotaxis": 交叉连接实现趋化性
          - 左传感器强驱动右推进器 (右转)
          - 右传感器强驱动左推进器 (左转)
          - 这实现了经典的"朝向目标"行为
    
    chemotaxis 模式下:
        - 当右侧传感器值高(闻到什么) → 左推进器强 → 向左转
        - 当左侧传感器值高 → 右推进器强 → 向右转
        结果: 智能体会朝向气味源移动
    """
    agent = Agent(agent_id=agent_id, x=x, y=y, theta=theta)
    
    if connection_type == "direct":
        # 直连模式：左传感器驱动左推进器，右传感器驱动右推进器
        agent.genome.add_edge(source_id=0, target_id=2, weight=1.0)
        agent.genome.add_edge(source_id=1, target_id=3, weight=1.0)
    
    elif connection_type == "chemotaxis":
        # 趋化性连接：交叉驱动
        # 左侧传感器信号 → 右推进器 (左转)
        agent.genome.add_edge(source_id=0, target_id=3, weight=1.5)
        # 右侧传感器信号 → 左推进器 (右转)
        agent.genome.add_edge(source_id=1, target_id=2, weight=1.5)
        
        # 添加基础推力 (无论传感器如何都有推进)
        agent.genome.add_edge(source_id=0, target_id=2, weight=0.5)
        agent.genome.add_edge(source_id=1, target_id=3, weight=0.5)
    
    return agent


def demo():
    """
    演示函数 - 测试整个系统
    """
    print("=" * 60)
    print("EOE MVP 演示 - 具身算子演化")
    print("=" * 60)
    
    # 创建环境
    env = Environment(width=100, height=100, target_pos=(80.0, 80.0))
    print(f"\n环境创建: {env.width}x{env.height}")
    print(f"目标位置: {env.target_pos}")
    
    # 创建智能体 (使用趋化性连接)
    agent = create_simple_agent(
        agent_id=0, 
        x=10.0, 
        y=80.0,  
        theta=0.0,  # 朝向右 (目标在80,80)
        connection_type="chemotaxis"
    )
    env.add_agent(agent)
    print(f"\n智能体创建: {agent}")
    print(f"基因组信息: {agent.genome.get_info()}")
    print("\n连接: 左传感器→右推进器, 右传感器→左推进器 (趋化性)")
    print("预期行为: 智能体会转向并朝向目标(80,80)移动")
    
    # 模拟多步
    print("\n--- 模拟开始 ---")
    for step in range(20):
        env.step()
        sensor_values = env._compute_sensor_values(agent)
        outputs = agent.genome.forward(sensor_values)
        
        print(
            f"Step {step+1:2d}: "
            f"pos=({agent.x:5.1f}, {agent.y:5.1f}) "
            f"theta={np.degrees(agent.theta):6.1f}° "
            f"sensors=[{sensor_values[0]:.3f}, {sensor_values[1]:.3f}] "
            f"actuators=[{outputs[0]:.3f}, {outputs[1]:.3f}] "
            f"fitness={agent.fitness:.1f}"
        )
    
    print("\n--- 模拟结束 ---")
    print(f"最终位置: ({agent.x:.2f}, {agent.y:.2f})")
    print(f"最终适应度: {agent.fitness:.2f}")
    print(f"距目标距离: {np.sqrt((env.target_pos[0]-agent.x)**2 + (env.target_pos[1]-agent.y)**2):.2f}")
    
    return env, agent


# ============================================================
# 6. 可视化模块 (Visualization)
# ============================================================

# 尝试导入 matplotlib，失败时提供优雅降级
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not available. Install with 'pip install matplotlib' for visual brain plots.")


def print_brain_text(agent: Agent) -> str:
    """
    文本方式打印大脑结构 (无需 matplotlib)
    
    参数:
        agent: 智能体
    
    返回:
        结构描述字符串
    """
    genome = agent.genome
    info = genome.get_info()
    
    lines = []
    lines.append("=" * 40)
    lines.append("AGENT BRAIN STRUCTURE")
    lines.append("=" * 40)
    lines.append(f"Nodes: {info['total_nodes']}, Edges: {info['enabled_edges']}")
    lines.append(f"Fitness: {agent.fitness:.4f}")
    lines.append("")
    
    # 节点类型统计
    type_counts = info.get('node_counts', {})
    lines.append("Node Types:")
    for ntype, count in type_counts.items():
        lines.append(f"  {ntype}: {count}")
    
    lines.append("")
    lines.append("Node Details:")
    for node in sorted(genome.nodes.values(), key=lambda n: n.node_id):
        if node.node_type == NodeType.DELAY:
            lines.append(f"  #{node.node_id}: {node.node_type.name:10s} (act={node.activation:7.3f}, delay_state={node.delay_state:7.3f})")
        else:
            lines.append(f"  #{node.node_id}: {node.node_type.name:10s} (act={node.activation:7.3f})")
    
    lines.append("")
    lines.append("Edges (connections):")
    for edge in genome.edges:
        if not edge['enabled']:
            continue
        src = genome.nodes.get(edge['source_id'])
        tgt = genome.nodes.get(edge['target_id'])
        if src and tgt:
            lines.append(f"  #{edge['source_id']}({src.node_type.name:10s}) --> #{edge['target_id']}({tgt.node_type.name:10s})  [w={edge['weight']:+.3f}]")
    
    lines.append("=" * 40)
    return "\n".join(lines)


def visualize_brain(
    agent: Agent,
    ax=None,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None
):
    """
    可视化智能体的大脑拓扑图
    
    参数:
        agent: 智能体
        ax: matplotlib axes (可选)
        figsize: 图形大小
        title: 标题 (可选)
    
    显示:
    - SENSOR: 绿色圆圈
    - ACTUATOR: 红色圆圈
    - ADD: 蓝色方形 (+)
    - MULTIPLY: 橙色方形 (x)
    - DELAY: 紫色菱形
    - CONSTANT: 灰色三角形
    - 边: 箭头表示数据流向，线宽表示权重
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    genome = agent.genome
    nodes = genome.nodes
    edges = genome.edges
    
    if not nodes:
        print("Warning: Empty genome!")
        return
    
    # 创建图形
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 使用层次布局 (按节点类型分层)
    # 层: SENSOR -> ADD/MULTIPLY/DELAY -> ACTUATOR
    layers = {
        NodeType.SENSOR: [],
        NodeType.CONSTANT: [],
        NodeType.ADD: [],
        NodeType.MULTIPLY: [],
        NodeType.DELAY: [],
        NodeType.ACTUATOR: []
    }
    
    for node in nodes.values():
        layers[node.node_type].append(node.node_id)
    
    # 计算节点位置
    positions = {}
    y_positions = {
        NodeType.SENSOR: 0,
        NodeType.CONSTANT: 1,
        NodeType.ADD: 2,
        NodeType.MULTIPLY: 2,
        NodeType.DELAY: 2,
        NodeType.ACTUATOR: 3
    }
    
    for node_type, node_ids in layers.items():
        n = len(node_ids)
        if n == 0:
            continue
        y = y_positions.get(node_type, 1)
        for i, node_id in enumerate(sorted(node_ids)):
            x = (i - (n - 1) / 2) * 2  # 水平分布
            positions[node_id] = (x, y)
    
    # 绘制边 (先画边，这样节点在上层)
    for edge in edges:
        if not edge['enabled']:
            continue
        
        start = positions.get(edge['source_id'])
        end = positions.get(edge['target_id'])
        
        if start is None or end is None:
            continue
        
        # 线宽基于权重
        weight = abs(edge['weight'])
        linewidth = max(0.5, weight * 2)
        
        # 颜色基于权重正负
        color = 'blue' if edge['weight'] > 0 else 'red'
        
        # 绘制箭头
        ax.annotate(
            '', xy=end, xytext=start,
            arrowprops=dict(arrowstyle='->', color=color, lw=linewidth, 
                           connectionstyle='arc3,rad=0.1')
        )
    
    # 定义节点样式
    node_styles = {
        NodeType.SENSOR: ('green', 'o', 'SENSOR'),
        NodeType.ACTUATOR: ('red', 'o', 'ACTUATOR'),
        NodeType.ADD: ('blue', 's', '+'),
        NodeType.MULTIPLY: ('orange', 's', '×'),
        NodeType.DELAY: ('purple', 'D', '⏱'),
        NodeType.CONSTANT: ('gray', '^', 'C')
    }
    
    # 绘制节点
    for node_id, (x, y) in positions.items():
        node = nodes[node_id]
        color, marker, label = node_styles.get(node.node_type, ('black', 'o', '?'))
        
        # 节点大小基于激活值
        size = 500 + abs(node.activation) * 200
        size = min(size, 1500)
        
        ax.scatter(x, y, s=size, c=color, marker=marker, 
                   edgecolors='black', linewidths=2, zorder=10)
        
        # 添加标签
        ax.text(x, y, label, ha='center', va='center', 
                fontsize=10, fontweight='bold', zorder=20)
        
        # 显示节点 ID
        ax.text(x, y - 0.3, f'#{node_id}', ha='center', va='top', 
                fontsize=8, color='gray')
    
    # 设置坐标轴
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 标题
    info = genome.get_info()
    if title is None:
        title = f"Agent Brain (Nodes: {info['total_nodes']}, Edges: {info['enabled_edges']}, Fitness: {agent.fitness:.2f})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor='green', edgecolor='black', label='SENSOR'),
        mpatches.Patch(facecolor='red', edgecolor='black', label='ACTUATOR'),
        mpatches.Patch(facecolor='blue', edgecolor='black', label='ADD (+)'),
        mpatches.Patch(facecolor='orange', edgecolor='black', label='MULTIPLY (×)'),
        mpatches.Patch(facecolor='purple', edgecolor='black', label='DELAY (⏱)'),
        mpatches.Patch(facecolor='gray', edgecolor='black', label='CONSTANT')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    return ax


def visualize_population_best(population: Population, generation: int = None):
    """
    可视化当前种群中最佳智能体的大脑
    
    参数:
        population: 种群对象
        generation: 当前代数 (用于标题)
    """
    import matplotlib.pyplot as plt
    
    # 找到最佳智能体
    best_agent = max(population.agents, key=lambda a: a.fitness)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    gen_num = generation if generation is not None else population.generation
    visualize_brain(best_agent, ax=ax, 
                    title=f"Generation {gen_num} - Best Agent Brain")
    
    plt.tight_layout()
    return fig, best_agent


def demo_evolution():
    """
    演示函数 - 测试演化系统
    """
    print("=" * 60)
    print("EOE 演化系统演示 - 具身算子演化")
    print("=" * 60)
    
    # 创建种群
    pop = Population(
        population_size=50,   # 50 个智能体
        elite_ratio=0.2,      # top 20% 精英
        lifespan=200,         # 200 步生命周期
        metabolic_alpha=0.1,  # 节点代谢惩罚
        metabolic_beta=0.05   # 边代谢惩罚
    )
    
    print(f"\n种群初始化: {pop.population_size} 个智能体")
    print(f"代谢惩罚: α={pop.environment.metabolic_alpha}, β={pop.environment.metabolic_beta}")
    
    # 运行 5 代演化
    history = pop.run(n_generations=5, verbose=True)
    
    # 输出最终结果
    print("\n" + "=" * 60)
    print("演化结果汇总")
    print("=" * 60)
    
    for i, stats in enumerate(history):
        print(f"Generation {i}: best={stats['best_fitness']:.2f}, "
              f"avg={stats['avg_fitness']:.2f}, "
              f"nodes={stats['avg_nodes']:.1f}, "
              f"edges={stats['avg_edges']:.1f}")
    
    # 展示最佳智能体
    best = history[-1]['best_agent']
    print(f"\n最佳智能体 (Gen {len(history)-1}):")
    print(f"  位置: ({best.x:.1f}, {best.y:.1f})")
    print(f"  基因组: {best.genome.get_info()}")
    
    return pop, history

