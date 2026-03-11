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

class GPUScheduler:
    """GPU并发调度器 - 4x A100 80GB"""
    
    def __init__(self, num_gpus: int = GPU_COUNT):
        self.num_gpus = num_gpus
        self.gpu_usage = {i: 0 for i in range(num_gpus)}
        self.gpu_locks = {i: mp.Lock() for i in range(num_gpus)}
        logger.info(f"🖥️ 初始化GPU调度器: {num_gpus} x A100 80GB")
    
    def get_available_gpu(self) -> int:
        """获取最空闲的GPU"""
        min_usage = min(self.gpu_usage.values())
        for gpu_id, usage in self.gpu_usage.items():
            if usage == min_usage:
                return gpu_id
        return 0
    
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