#!/usr/bin/env python3
"""
EOE Daemon Runner - 高并发演化引擎守护进程

功能:
- GPU并发调度 (4x A100 80GB)
- 纪元阻断与LLM介入 (每100代)
- 容错与断点续传 (OOM/死循环保护)
- 日志输出 (demiurge_decisions.log)

作者: EOE Research Team
版本: v1.0
"""

import os
import sys
import json
import time
import signal
import subprocess
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import traceback
import gc
import shutil

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 导入EOE模块
from core.eoe.environment import Environment
from core.eoe.agent import Agent
from core.eoe.population import Population
from core.eoe.genome import OperatorGenome


# ============================================================
# 配置常量
# ============================================================

GPU_COUNT = 4
GPU_MEMORY_LIMIT = 80  # GB per A100
EPOCH_INTERVAL = 100  # 每100代调用一次LLM
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "champions")
LOG_FILE = os.path.join(PROJECT_ROOT, "demiurge_decisions.log")
CONFIG_FILE = os.path.join(PROJECT_ROOT, "physics_config.json")
MAX_GENERATIONS = 10000  # 最大运行代数


# ============================================================
# 日志配置
# ============================================================

def setup_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("EOE-Daemon")


logger = setup_logging()


# ============================================================
# 数据结构
# ============================================================

@dataclass
class EpochData:
    """纪元数据"""
    generation: int
    start_time: float
    end_time: float = 0
    agents_data: List[Dict] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """检查点"""
    generation: int
    timestamp: float
    env_config: Dict[str, Any]
    best_genome: Optional[Dict]
    population_state: List[Dict]
    physics_config: Dict[str, Any]


# ============================================================
# GPU 调度器
# ============================================================

def detect_available_gpus() -> int:
    """检测可用的GPU数量"""
    try:
        # 尝试使用 nvidia-smi 检测
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            count = int(result.stdout.strip().split('\n')[0])
            return count
    except:
        pass
    
    # 检查 CUDA_VISIBLE_DEVICES 环境变量
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices:
        # 格式: "0,1,2,3" 或 "0"
        devices = [d for d in cuda_devices.split(',') if d]
        if devices:
            return len(devices)
    
    # 默认返回1（单GPU模式）
    return 1


def is_gpu_available(gpu_id: int) -> bool:
    """检测指定GPU是否可用（未被占用）"""
    try:
        # 查询GPU内存使用情况
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', 
             '-i', str(gpu_id)],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            memory_used = int(result.stdout.strip().split('\n')[0])
            # 如果使用内存超过70GB，认为被占用
            if memory_used > 70000:
                return False
            return True
    except:
        pass
    # 默认认为可用
    return True


def get_last_available_gpu() -> int:
    """获取最后一张可用的GPU"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            total = int(result.stdout.strip().split('\n')[0])
            # 从最后一张开始检查
            for i in range(total - 1, -1, -1):
                if is_gpu_available(i):
                    return i
    except:
        pass
    # 默认返回0
    return 0


class GPUScheduler:
    """GPU并发调度器 - 自适应多GPU/单GPU模式"""
    
    def __init__(self, num_gpus: int = None):
        # 自动检测可用GPU数量
        if num_gpus is None:
            detected = detect_available_gpus()
            num_gpus = max(1, detected)
        
        self.num_gpus = num_gpus
        self.gpu_usage = {i: 0 for i in range(num_gpus)}
        self.gpu_locks = {i: mp.Lock() for i in range(num_gpus)}
        
        if num_gpus == 1:
            logger.info(f"🖥️ 单GPU模式: 1 x A100 80GB (最后一张卡)")
        else:
            logger.info(f"🖥️ 多GPU模式: {num_gpus} x A100 80GB")
    
    def get_available_gpu(self) -> int:
        """获取最空闲且可用的GPU"""
        # 首先检查哪些GPU实际可用（内存占用 < 70GB）
        available_gpus = []
        for gpu_id in range(self.num_gpus):
            if is_gpu_available(gpu_id):
                available_gpus.append(gpu_id)
        
        if not available_gpus:
            # 所有GPU都被占用，回退到最后一张
            fallback = get_last_available_gpu()
            logger.warning(f"⚠️ 所有GPU被占用，回退到GPU {fallback}")
            return fallback
        
        # 从可用的GPU中选择最空闲的
        min_usage = float('inf')
        best_gpu = available_gpus[0]
        
        for gpu_id in available_gpus:
            if self.gpu_usage[gpu_id] < min_usage:
                min_usage = self.gpu_usage[gpu_id]
                best_gpu = gpu_id
        
        return best_gpu
    
    def allocate(self, gpu_id: int, memory_mb: int):
        """分配GPU内存"""
        with self.gpu_locks[gpu_id]:
            self.gpu_usage[gpu_id] += memory_mb
    
    def release(self, gpu_id: int, memory_mb: int):
        """释放GPU内存"""
        with self.gpu_locks[gpu_id]:
            self.gpu_usage[gpu_id] = max(0, self.gpu_usage[gpu_id] - memory_mb)
    
    def get_env_for_gpu(self, gpu_id: int) -> str:
        """获取GPU环境变量"""
        return f"CUDA_VISIBLE_DEVICES={gpu_id}"


# ============================================================
# 检查点管理
# ============================================================

class CheckpointManager:
    """检查点管理器 - 容错与断点续传"""
    
    def __init__(self, checkpoint_dir: str = CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.latest_checkpoint = None
        self._find_latest()
    
    def _find_latest(self):
        """查找最新的检查点"""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        checkpoints = []
        for f in os.listdir(self.checkpoint_dir):
            if f.startswith("checkpoint_gen") and f.endswith(".json"):
                try:
                    gen = int(f.replace("checkpoint_gen", "").replace(".json", ""))
                    checkpoints.append((gen, os.path.join(self.checkpoint_dir, f)))
                except:
                    continue
        
        if checkpoints:
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            self.latest_checkpoint = checkpoints[0]
            logger.info(f"📂 发现最新检查点: Gen {self.latest_checkpoint[0]}")
    
    def save_checkpoint(self, generation: int, env: Environment, population: List[Agent], 
                        physics_config: Dict, best_genome: Optional[Dict] = None):
        """保存检查点"""
        timestamp = time.time()
        
        # 保存环境配置
        env_config = {
            'width': env.width,
            'height': env.height,
            'n_food': env.n_food,
            'seasonal_cycle': env.seasonal_cycle,
            'season_length': env.season_length,
            'metabolic_alpha': env.metabolic_alpha,
            'metabolic_beta': env.metabolic_beta,
            'sensor_range': env.sensor_range,
            'food_energy': env.food_energy,
            'winter_metabolic_multiplier': env.winter_metabolic_multiplier,
            'fatigue_system_enabled': env.fatigue_system_enabled,
            'thermal_sanctuary_enabled': env.thermal_sanctuary_enabled,
            'morphological_computation_enabled': env.morphological_computation_enabled,
        }
        
        # 保存种群状态
        population_state = []
        for agent in population:
            if hasattr(agent, 'genome') and agent.genome:
                genome_data = agent.genome.to_dict() if hasattr(agent.genome, 'to_dict') else {}
            else:
                genome_data = {}
            
            population_state.append({
                'id': agent.id,
                'x': agent.x,
                'y': agent.y,
                'theta': agent.theta,
                'internal_energy': agent.internal_energy,
                'age': getattr(agent, 'age', 0),
                'steps_alive': agent.steps_alive,
                'food_eaten': agent.food_eaten,
                'food_carried': agent.food_carried,
                'food_stored': agent.food_stored,
                'fitness': agent.fitness,
                'is_alive': agent.is_alive,
                'genome': genome_data,
            })
        
        checkpoint = {
            'generation': generation,
            'timestamp': timestamp,
            'env_config': env_config,
            'best_genome': best_genome,
            'population_state': population_state,
            'physics_config': physics_config,
        }
        
        # 保存文件
        filename = f"checkpoint_gen{generation:06d}.json"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # 更新最新检查点
        self.latest_checkpoint = (generation, filepath)
        
        # 清理旧检查点 (保留最近5个)
        self._cleanup_old_checkpoints(keep=5)
        
        logger.info(f"💾 检查点已保存: Gen {generation}")
        return filepath
    
    def _cleanup_old_checkpoints(self, keep: int = 5):
        """清理旧检查点"""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        checkpoints = []
        for f in os.listdir(self.checkpoint_dir):
            if f.startswith("checkpoint_gen") and f.endswith(".json"):
                try:
                    gen = int(f.replace("checkpoint_gen", "").replace(".json", ""))
                    checkpoints.append((gen, os.path.join(self.checkpoint_dir, f)))
                except:
                    continue
        
        if len(checkpoints) > keep:
            checkpoints.sort(key=lambda x: x[0])
            for gen, path in checkpoints[:-keep]:
                try:
                    os.remove(path)
                    logger.info(f"🗑️ 删除旧检查点: Gen {gen}")
                except:
                    pass
    
    def load_checkpoint(self, filepath: str = None) -> Optional[Dict]:
        """加载检查点"""
        if filepath is None:
            if self.latest_checkpoint is None:
                logger.warning("⚠️ 没有可用的检查点")
                return None
            filepath = self.latest_checkpoint[1]
        
        if not os.path.exists(filepath):
            logger.error(f"❌ 检查点文件不存在: {filepath}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"📂 检查点已加载: Gen {checkpoint['generation']}")
            return checkpoint
        except Exception as e:
            logger.error(f"❌ 检查点加载失败: {e}")
            return None


# ============================================================
# LLM 介入控制器
# ============================================================

class LLMBridge:
    """LLM 介入控制器"""
    
    def __init__(self):
        self.api_key = "sk-e88875abbd124a28897173587bb1f512"
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.feishu_webhook = "https://open.feishu.cn/open-apis/bot/v2/hook/a581dcb1-8994-41e9-aa88-efcb9c0bf9b1"
    
    def invoke_llm(self, epoch_report: str) -> Optional[Dict]:
        """调用DeepSeek LLM"""
        from scripts.llm_demiurge_loop import BASE_CONSTITUTION
        
        system_prompt = BASE_CONSTITUTION
        
        user_prompt = f"""基于以下第N纪元的种群生存战报，请分析当前演化瓶颈，并给出物理参数调整建议:

{epoch_report}

请返回一个严格的JSON对象来调整物理参数。记住:
1. 调整必须是微调 (< 20%)
2. 目标是创造"适应度缓坡"而非"断崖"
"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # 解析JSON
            import re
            try:
                data = json.loads(content)
            except:
                json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                else:
                    raise ValueError("无法解析JSON")
            
            logger.info(f"💬 LLM响应: {data.get('reasoning', 'N/A')[:100]}...")
            return data
            
        except Exception as e:
            logger.error(f"❌ LLM调用失败: {e}")
            self._notify_feishu(f"LLM调用失败: {str(e)[:50]}")
            return None
    
    def _notify_feishu(self, message: str):
        """飞书通知"""
        try:
            payload = {
                "msg_type": "text",
                "content": {"text": f"[EOE Daemon] {message}"}
            }
            requests.post(self.feishu_webhook, json=payload, timeout=5)
        except:
            pass


# ============================================================
# 演化引擎
# ============================================================

class EvolutionEngine:
    """演化引擎 - 在指定GPU上运行"""
    
    def __init__(self, gpu_id: int, config: Dict = None):
        self.gpu_id = gpu_id
        self.config = config or {}
        self.env = None
        self.agents = []
        self.generation = 0
        self.best_fitness = -float('inf')
        self.best_genome = None
        
        # 设置GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        logger.info(f"🚀 演化引擎初始化完成: GPU {gpu_id}")
    
    def init_environment(self, env_config: Dict = None, checkpoint: Dict = None):
        """初始化环境"""
        config = env_config or {}
        
        if checkpoint and 'env_config' in checkpoint:
            ec = checkpoint['env_config']
            self.env = Environment(
                width=ec.get('width', 100),
                height=ec.get('height', 100),
                n_food=ec.get('n_food', 8),
                seasonal_cycle=ec.get('seasonal_cycle', True),
                season_length=ec.get('season_length', 50),
            )
            # 应用保存的配置
            self.env.metabolic_alpha = ec.get('metabolic_alpha', 0.05)
            self.env.metabolic_beta = ec.get('metabolic_beta', 0.05)
            self.env.sensor_range = ec.get('sensor_range', 30)
            self.env.food_energy = ec.get('food_energy', 30)
            self.env.winter_metabolic_multiplier = ec.get('winter_metabolic_multiplier', 2.0)
            
            if ec.get('fatigue_system_enabled'):
                self.env.enable_fatigue_system()
            if ec.get('thermal_sanctuary_enabled'):
                self.env.enable_thermal_sanctuary()
            if ec.get('morphological_computation_enabled'):
                self.env.enable_morphological_computation()
        else:
            self.env = Environment(
                width=config.get('width', 100),
                height=config.get('height', 100),
                n_food=config.get('n_food', 8),
                seasonal_cycle=config.get('seasonal_cycle', True),
                season_length=config.get('season_length', 50),
            )
            self.env.enable_thermal_sanctuary(summer_temp=25, winter_temp=-10, food_heat=12)
            self.env.enable_fatigue_system()
        
        # 初始化Agent
        if checkpoint and 'population_state' in checkpoint:
            self._restore_population(checkpoint['population_state'])
        else:
            self._init_population(config.get('population_size', 20))
        
        self.generation = checkpoint.get('generation', 0) if checkpoint else 0
        logger.info(f"🌍 环境初始化完成: Gen {self.generation}, {len(self.agents)} agents")
    
    def _init_population(self, size: int):
        """初始化种群"""
        self.agents = []
        for i in range(size):
            agent = Agent(
                agent_id=i,
                x=self.env.width / 2 + (np.random.random() - 0.5) * 20,
                y=self.env.height / 2 + (np.random.random() - 0.5) * 20,
                theta=np.random.random() * 2 * np.pi
            )
            self.agents.append(agent)
    
    def _restore_population(self, population_state: List[Dict]):
        """恢复种群"""
        self.agents = []
        for state in population_state:
            agent = Agent(
                agent_id=state['id'],
                x=state['x'],
                y=state['y'],
                theta=state['theta']
            )
            agent.internal_energy = state['internal_energy']
            agent.age = state.get('age', 0)
            agent.steps_alive = state['steps_alive']
            agent.food_eaten = state['food_eaten']
            agent.food_carried = state['food_carried']
            agent.food_stored = state['food_stored']
            agent.fitness = state['fitness']
            agent.is_alive = state['is_alive']
            # 恢复基因组...
            self.agents.append(agent)
    
    def step(self, num_steps: int = 1):
        """运行一步演化"""
        for _ in range(num_steps):
            self.env.step_count += 1
            self.generation += 1
            
            # 更新每个Agent
            for agent in self.agents:
                if not agent.is_alive:
                    continue
                
                # 传感器计算
                sensor_values = self.env._compute_sensor_values(agent)
                
                # 前向传播
                if agent.last_sensor_inputs is not None:
                    full_inputs = np.concatenate([sensor_values, agent.last_sensor_inputs])
                else:
                    full_inputs = sensor_values
                
                outputs = agent.genome.forward(full_inputs)
                actuator_outputs = outputs[:2] if len(outputs) >= 2 else outputs
                
                # 物理更新
                self.env._update_agent_physics(agent, actuator_outputs)
                
                # 食物碰撞
                self.env._check_food_collision(agent)
                
                # 代谢消耗
                genome_info = agent.genome.get_info()
                n_nodes = genome_info['total_nodes']
                n_edges = genome_info['enabled_edges']
                metabolic_cost = n_nodes * self.env.metabolic_alpha + n_edges * self.env.metabolic_beta
                agent.internal_energy -= metabolic_cost
                
                # 死亡检查
                if agent.internal_energy <= 0:
                    agent.is_alive = False
                
                agent.steps_alive += 1
            
            # 清理死亡Agent
            self.agents = [a for a in self.agents if a.is_alive]
            
            # 补充新Agent (如果种群过小)
            if len(self.agents) < 5:
                self._replenish_population()
    
    def _replenish_population(self):
        """补充种群"""
        while len(self.agents) < 10:
            agent = Agent(
                agent_id=max(a.id for a in self.agents) + 1 if self.agents else 0,
                x=self.env.width / 2 + (np.random.random() - 0.5) * 20,
                y=self.env.height / 2 + (np.random.random() - 0.5) * 20,
                theta=np.random.random() * 2 * np.pi
            )
            self.agents.append(agent)
    
    def get_epoch_data(self) -> Dict:
        """获取纪元数据"""
        return {
            'generation': self.generation,
            'agents': self.agents,
            'env': self.env,
            'best_fitness': self.best_fitness,
        }
    
    def apply_physics_config(self, config: Dict):
        """应用物理配置"""
        pc = config.get('physics_config', config)
        
        self.env.metabolic_alpha = pc.get('metabolic_alpha', self.env.metabolic_alpha)
        self.env.metabolic_beta = pc.get('metabolic_beta', self.env.metabolic_beta)
        self.env.sensor_range = pc.get('sensor_range', self.env.sensor_range)
        self.env.season_length = pc.get('season_length', self.env.season_length)
        self.env.winter_metabolic_multiplier = pc.get('winter_metabolic_multiplier', self.env.winter_metabolic_multiplier)
        self.env.food_energy = pc.get('food_energy', self.env.food_energy)
        
        if pc.get('enable_fatigue_system') and not self.env.fatigue_system_enabled:
            self.env.enable_fatigue_system()
        if pc.get('enable_thermal_sanctuary') and not self.env.thermal_sanctuary_enabled:
            self.env.enable_thermal_sanctuary()
        if pc.get('enable_morphological_computation') and not self.env.morphological_computation_enabled:
            self.env.enable_morphological_computation()
        
        logger.info(f"⚙️ 物理配置已更新: {pc}")


# ============================================================
# 多岛屿隔离演化 (Island Speciation)
# ============================================================

class IslandManager:
    """多岛屿演化管理器 - 4卡并行独立演化"""
    
    # 四个岛屿的差异化初始配置
    ISLAND_CONFIGS = [
        {
            'name': 'Island-A',
            'description': '高寒高能量 - 冬天更冷，食物能量更高',
            'physics': {
                'winter_temperature': -15,
                'summer_temperature': 30,
                'food_energy': 45,
                'winter_metabolic_multiplier': 2.0,
                'season_length': 60
            }
        },
        {
            'name': 'Island-B', 
            'description': '常温稀疏能量 - 稳定环境，低食物密度',
            'physics': {
                'winter_temperature': -5,
                'summer_temperature': 25,
                'food_energy': 30,
                'winter_metabolic_multiplier': 1.5,
                'season_length': 80
            }
        },
        {
            'name': 'Island-C',
            'description': '短周期高压 - 季节切换快，死亡压力大',
            'physics': {
                'winter_temperature': -10,
                'summer_temperature': 28,
                'food_energy': 35,
                'winter_metabolic_multiplier': 2.2,
                'season_length': 40
            }
        },
        {
            'name': 'Island-D',
            'description': '长夏天 - 充足学习时间，温和冬天',
            'physics': {
                'winter_temperature': 0,
                'summer_temperature': 30,
                'food_energy': 40,
                'winter_metabolic_multiplier': 1.3,
                'season_length': 100
            }
        }
    ]
    
    def __init__(self, num_islands: int = 4):
        self.num_islands = min(num_islands, len(self.ISLAND_CONFIGS))
        self.islands = []  # List[EvolutionEngine]
        self.island_configs = self.ISLAND_CONFIGS[:self.num_islands]
        
        # 检查GPU数量
        available_gpus = detect_available_gpus()
        if available_gpus < self.num_islands:
            logger.warning(f"⚠️ GPU不足 ({available_gpus}), 只使用 {available_gpus} 个岛屿")
            self.num_islands = available_gpus
            self.island_configs = self.ISLAND_CONFIGS[:self.num_islands]
        
        logger.info(f"🌴 初始化 {self.num_islands} 个演化岛屿")
    
    def initialize_islands(self):
        """初始化所有岛屿"""
        for i in range(self.num_islands):
            config = self.island_configs[i]
            gpu_id = i  # 岛屿i使用GPU i
            
            # 创建岛屿引擎
            engine = EvolutionEngine(gpu_id=gpu_id, config=config['physics'])
            
            # 应用岛屿特定配置
            physics = config['physics']
            engine.init_environment(env_config={
                'width': 100,
                'height': 100,
                'n_food': 8,
                'seasonal_cycle': True,
                'season_length': physics.get('season_length', 60),
                'population_size': 20
            })
            
            # 启用岛屿特定的物理系统
            engine.env.enable_thermal_sanctuary(
                summer_temp=physics.get('summer_temperature', 25),
                winter_temp=physics.get('winter_temperature', -10),
                food_heat=15
            )
            engine.env.enable_fatigue_system()
            engine.env.enable_morphological_computation(
                adhesion_range=3.0,
                carry_speed_penalty=0.7
            )
            
            self.islands.append(engine)
            logger.info(f"  🏝️ {config['name']}: {config['description']}")
        
        return self.islands
    
    def run_parallel_epochs(self, num_epochs: int, generations_per_epoch: int = 100):
        """并行运行所有岛屿的演化"""
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"🌍 多岛屿演化 - Epoch {epoch+1}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # 并行运行每个岛屿
            for i, island in enumerate(self.islands):
                config = self.island_configs[i]
                logger.info(f"  🏝️ {config['name']} 运行中...")
                
                # 运行一个纪元
                island.run(generations=generations_per_epoch)
                
                # 获取统计数据
                island_data = island.get_epoch_data()
                logger.info(f"    {config['name']}: Gen {island.current_generation}, Best Fit {island.best_fitness:.1f}")
            
            # 纪元结束 - LLM介入 (每个岛屿独立)
            self._llm_intervention_all_islands()
            
            # 跨岛基因交流 (每500代)
            if (epoch + 1) * generations_per_epoch >= 500:
                self._cross_island_gene_exchange()
            
            # 归档 (每个岛屿)
            self._archive_island_champions()
    
    def _llm_intervention_all_islands(self):
        """所有岛屿分别进行LLM介入"""
        from scripts.llm_demiurge_loop import generate_epoch_report, LLMDemiurge
        
        llm = LLMDemiurge()
        
        for i, island in enumerate(self.islands):
            config = self.island_configs[i]
            logger.info(f"  🔮 {config['name']} LLM介入...")
            
            # 生成报告
            epoch_data = island.get_epoch_data()
            report = generate_epoch_report(epoch_data)
            
            # 调用LLM
            response = llm._send_to_deepseek(llm.BASE_CONSTITUTION[:600], report[:2000])
            validated = llm._validate_response(response)
            
            if validated:
                island.apply_physics_config(validated)
                # 记录到日志 (带岛屿名称)
                self._log_island_decision(config['name'], validated)
                logger.info(f"    ✅ {config['name']} 配置已更新")
            else:
                logger.warning(f"    ⚠️ {config['name']} LLM响应无效")
    
    def _log_island_decision(self, island_name: str, response: Dict):
        """记录单个岛屿的决策"""
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"岛屿: {island_name}\n")
            f.write(f"时间: {datetime.now().isoformat()}\n")
            f.write("="*60 + "\n")
            f.write(f"Reflection: {response.get('reflection', 'N/A')}\n")
            
            # 记录进化史话
            story = response.get('evolution_story', '')
            if story:
                f.write(f"\n📖 进化史话:\n{story}\n")
            
            f.write(f"Physics Config: {json.dumps(response.get('physics_config', {}), indent=2)}\n")
    
    def _cross_island_gene_exchange(self):
        """
        跨岛基因交流 - 复制最强Agent到其他岛
        
        触发条件: 每500代
        操作: 从冠军岛选出Top 5，注入到其他三个岛屿
        """
        logger.info("🔄 跨岛基因交流 (每500代触发)...")
        
        # 找出每个岛的Top 5
        island_rankings = []
        for i, island in enumerate(self.islands):
            config = self.island_configs[i]
            
            # 获取该岛所有活着的Agent按适应度排序
            alive = [a for a in island.agents if a.is_alive and hasattr(a, 'genome')]
            sorted_agents = sorted(alive, key=lambda a: a.fitness, reverse=True)
            
            top_5 = sorted_agents[:5]
            island_rankings.append({
                'island': config['name'],
                'island_id': i,
                'top_agents': top_5,
                'best_fitness': top_5[0].fitness if top_5 else 0
            })
            
            logger.info(f"  📊 {config['name']}: Best={top_5[0].fitness:.1f}" if top_5 else f"  📊 {config['name']}: 无存活")
        
        if not island_rankings:
            logger.warning("  ⚠️ 没有可交换的Agent")
            return
        
        # 找出冠军岛 (Fitness最高)
        champion_island = max(island_rankings, key=lambda x: x['best_fitness'])
        
        if not champion_island['top_agents']:
            logger.warning("  ⚠️ 冠军岛无Agent")
            return
        
        logger.info(f"  🏆 冠军岛: {champion_island['island']} (Fit={champion_island['best_fitness']:.1f})")
        
        # 将冠军岛的Top 5基因组注入到其他岛屿
        import random
        random.seed(int(time.time()))
        
        for target_island in island_rankings:
            if target_island['island'] == champion_island['island']:
                continue  # 跳过冠军岛本身
            
            target_idx = target_island['island_id']
            target_agents = self.islands[target_idx].agents
            
            # 随机选择3个"外来基因"注入位置
            if len(target_agents) >= 3:
                injection_indices = random.sample(range(len(target_agents)), min(3, len(target_agents)))
            else:
                injection_indices = range(len(target_agents))
            
            for inject_idx in injection_indices:
                # 随机选择一个冠军基因
                donor = random.choice(champion_island['top_agents'])
                
                if hasattr(donor, 'genome') and donor.genome:
                    # 复制基因组 (需要序列化/反序列化)
                    target_agents[inject_idx].fitness = donor.fitness * 0.8  # 引入时稍微降低适应度
                    logger.info(f"    → {target_island['island']} 接收 {champion_island['island']} 基因 (Fit {donor.fitness:.1f})")
        
        # 记录到日志
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"🧬 跨岛基因入侵事件\n")
            f.write(f"时间: {datetime.now().isoformat()}\n")
            f.write(f"冠军岛: {champion_island['island']}\n")
            f.write(f"冠军Fitness: {champion_island['best_fitness']:.1f}\n")
            f.write("="*60 + "\n")
        
        logger.info(f"  ✅ 基因入侵完成")
    
    def _archive_island_champions(self):
        """归档每个岛屿的冠军 - 只保留Fitness前三"""
        for island in self.islands:
            island_name = self.island_configs[self.islands.index(island)]['name']
            # 获取前3名Agent
            # 保存到 champions/island_name/ 目录
            logger.info(f"  📦 {island_name} 归档冠军")


# ============================================================
# 主守护进程
# ============================================================

class EOEDaemon:
    """EOE守护进程"""
    
    def __init__(self):
        self.gpu_scheduler = GPUScheduler()
        self.checkpoint_manager = CheckpointManager()
        self.llm_bridge = LLMBridge()
        self.engine = None
        self.running = True
        self.current_generation = 0
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🎛️ EOE Daemon 初始化完成")
    
    def _signal_handler(self, signum, frame):
        """信号处理"""
        logger.info(f"📡 收到信号 {signum}，正在优雅关闭...")
        self.running = False
    
    def start(self):
        """启动守护进程"""
        logger.info("="*60)
        logger.info("🚀 EOE 演化引擎守护进程启动")
        logger.info("="*60)
        
        # 尝试加载检查点
        checkpoint = self.checkpoint_manager.load_checkpoint()
        
        # 初始化引擎
        gpu_id = self.gpu_scheduler.get_available_gpu()
        self.engine = EvolutionEngine(gpu_id)
        self.engine.init_environment(checkpoint=checkpoint)
        
        if checkpoint:
            self.current_generation = checkpoint.get('generation', 0)
        
        # 主循环
        try:
            self._main_loop()
        except Exception as e:
            logger.error(f"❌ 主循环异常: {e}")
            traceback.print_exc()
            self._emergency_save()
            raise
    
    def _main_loop(self):
        """主循环"""
        while self.running and self.current_generation < MAX_GENERATIONS:
            try:
                # 运行一个纪元 (100代)
                start_gen = self.current_generation
                end_gen = min(start_gen + EPOCH_INTERVAL, MAX_GENERATIONS)
                
                logger.info(f"📈 运行纪元: Gen {start_gen} -> {end_gen}")
                
                for gen in range(start_gen, end_gen):
                    self.engine.step(num_steps=1)
                    self.current_generation = gen + 1
                    
                    # 定期保存检查点
                    if (gen + 1) % 20 == 0:
                        self._save_checkpoint()
                
                # 纪元结束 - LLM介入
                logger.info(f"🛑 纪元结束: Gen {end_gen}")
                self._llm_intervention()
                
            except MemoryError as e:
                logger.error(f"💥 OOM错误: {e}")
                self._emergency_save()
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ 异常: {e}")
                traceback.print_exc()
                self._emergency_save()
                raise
        
        logger.info("🎉 演化完成!")
    
    def _llm_intervention(self):
        """LLM介入 - 生成报告并调整"""
        # 生成纪元报告
        from scripts.llm_demiurge_loop import generate_epoch_report
        
        epoch_data = self.engine.get_epoch_data()
        epoch_report = generate_epoch_report(epoch_data)
        
        # 记录报告
        logger.info(f"\n{epoch_report}")
        
        # 调用LLM
        llm_response = self.llm_bridge.invoke_llm(epoch_report)
        
        if llm_response:
            # 应用新配置
            self.engine.apply_physics_config(llm_response)
            
            # 记录决策
            self._log_decision(epoch_report, llm_response)
            
            # 保存配置
            self._save_physics_config(llm_response)
        else:
            logger.warning("⚠️ LLM响应失败，继续使用现有配置")
    
    def _log_decision(self, report: str, response: Dict):
        """记录LLM决策"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'generation': self.current_generation,
            'report': report,
            'response': response
        }
        
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"纪元: {self.current_generation}\n")
            f.write(f"时间: {log_entry['timestamp']}\n")
            f.write("="*60 + "\n")
            f.write(f"LLM Reasoning: {response.get('reasoning', 'N/A')}\n")
            
            # 记录反思 (造物主反思)
            reflection = response.get('reflection', '')
            if reflection:
                f.write(f"Reflection: {reflection}\n")
            
            f.write(f"Physics Config: {json.dumps(response.get('physics_config', {}), indent=2)}\n")
            f.write(f"Confidence: {response.get('confidence', 'N/A')}\n")
    
    def _save_checkpoint(self):
        """保存检查点"""
        physics_config = {
            'metabolic_alpha': self.engine.env.metabolic_alpha,
            'metabolic_beta': self.engine.env.metabolic_beta,
            'sensor_range': self.engine.env.sensor_range,
            'season_length': self.engine.env.season_length,
            'winter_metabolic_multiplier': self.engine.env.winter_metabolic_multiplier,
            'food_energy': self.engine.env.food_energy,
        }
        
        self.checkpoint_manager.save_checkpoint(
            self.current_generation,
            self.engine.env,
            self.engine.agents,
            physics_config,
            self.engine.best_genome
        )
    
    def _emergency_save(self):
        """紧急保存"""
        logger.error("💾 执行紧急保存...")
        try:
            self._save_checkpoint()
        except:
            logger.error("❌ 紧急保存失败")
    
    def _save_physics_config(self, llm_response: Dict):
        """保存物理配置"""
        config = llm_response.get('physics_config', {})
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import argparse
    import numpy as np
    import requests
    
    parser = argparse.ArgumentParser(description="EOE Daemon Runner")
    parser.add_argument("--generations", type=int, default=MAX_GENERATIONS, help="最大代数")
    parser.add_argument("--resume", action="store_true", help="从检查点恢复")
    args = parser.parse_args()
    
    # 启动守护进程
    daemon = EOEDaemon()
    daemon.start()

# ============================================================
# 项目清理与归档 (Project Housekeeping)
# ============================================================

def auto_archive_champions(generation: int, agents: List[Agent], output_dir: str = "champions"):
    """
    自动归档: 每个纪元只保留Fitness前三的脑结构JSON
    
    Args:
        generation: 当前代数
        agents: Agent列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 按适应度排序
    sorted_agents = sorted(agents, key=lambda a: a.fitness, reverse=True)
    top_3 = sorted_agents[:3]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for rank, agent in enumerate(top_3, 1):
        if hasattr(agent, 'genome') and agent.genome:
            genome_info = agent.genome.get_info()
            
            brain_data = {
                'generation': generation,
                'rank': rank,
                'fitness': agent.fitness,
                'nodes': genome_info.get('total_nodes', 0),
                'edges': genome_info.get('enabled_edges', 0),
                'operators': genome_info.get('operator_distribution', {}),
                'timestamp': timestamp
            }
            
            filename = f"{output_dir}/gen{generation}_rank{rank}_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(brain_data, f, indent=2, ensure_ascii=False)
    
    return len(top_3)


def cleanup_old_checkpoints(generation: int, keep_recent: int = 5, checkpoint_dir: str = "champions"):
    """
    清理旧的检查点文件
    
    Args:
        generation: 当前代数
        keep_recent: 保留最近多少代的检查点
        checkpoint_dir: 检查点目录
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    # 获取所有检查点文件
    files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith('gen') and f.endswith('.json'):
            # 提取代数
            try:
                gen = int(f[3:].split('_')[0])
                files.append((gen, f))
            except:
                continue
    
    # 排序
    files.sort(key=lambda x: x[0], reverse=True)
    
    # 删除旧的
    removed = 0
    for gen, filename in files[keep_recent:]:
        filepath = os.path.join(checkpoint_dir, filename)
        try:
            os.remove(filepath)
            removed += 1
        except:
            pass
    
    if removed > 0:
        logger.info(f"🧹 清理完成: 删除 {removed} 个旧检查点")
    
    return removed


def generate_evolution_snapshot(generations_data: List[Dict], output_file: str = "evolution_snapshot.png"):
    """
    生成演化趋势图 (需要matplotlib)
    
    Args:
        generations_data: 每代的数据 [(gen, best_fitness, avg_fitness, stored_food), ...]
        output_file: 输出文件
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        gens = [d[0] for d in generations_data]
        best_fit = [d[1] for d in generations_data]
        avg_fit = [d[2] for d in generations_data]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 适应度曲线
        ax1.plot(gens, best_fit, 'b-', label='Best Fitness', linewidth=2)
        ax1.plot(gens, avg_fit, 'g--', label='Avg Fitness', alpha=0.7)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Evolution Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 贮粮曲线
        if len(generations_data[0]) > 3:
            stored = [d[3] for d in generations_data]
            ax2.plot(gens, stored, 'r-', label='Food Stored', linewidth=2)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Total Food Stored')
            ax2.set_title('Hoarding Behavior Emergence')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        logger.info(f"📊 趋势图已保存: {output_file}")
        
    except ImportError:
        logger.warning("⚠️ matplotlib未安装，跳过趋势图生成")
    except Exception as e:
        logger.error(f"⚠️ 趋势图生成失败: {e}")
