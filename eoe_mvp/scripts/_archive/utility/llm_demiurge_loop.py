#!/usr/bin/env python3
"""
EOE LLM Demiurge Loop - 动态物理法则调整系统

功能:
- 每100代(纪元)结束时分析种群痛点
- 将真实数据拼接为"生存战报"发送给DeepSeek
- 根据回复动态调整物理参数(热更新)
- 严格校验JSON Schema，拦截幻觉

作者: EOE Research Team
版本: v1.0
"""

import json
import time
import requests
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.eoe.population import Population
from core.eoe.environment import Environment


# ============================================================
# 1. 不可变的"宪法" - System Prompt
# ============================================================

BASE_CONSTITUTION = """你是 EOE (Evolutionary Optimalism Engine) 演化宇宙的底层物理法则控制者。

你的唯一目标是通过调整环境的物理参数，为种群创造适度的生存压力（适应度缓坡），促使它们涌现出如"贮粮过冬"等高级智能行为。

## 核心职责
1. 分析种群当前的演化瓶颈和痛点
2. 调整物理参数使适应度 landscape 呈现"缓坡"而非"悬崖"
3. 引导复杂行为（如长链路贮粮）自然涌现

## 绝对红线（禁止违反）
1. 绝对禁止以任何形式建议或直接修改 Agent 的神经拓扑结构、大脑节点或基因组（Genome）
2. 绝对禁止制造瞬间团灭的极端环境断崖。你所有的参数调整必须是微调（幅度 < 20%）
3. 绝对禁止直接"设计"智能行为，只能通过环境压力"引导"涌现

## 物理参数调整指南
- 代谢参数 (metabolic_alpha/beta): 调整能量消耗，影响寿命
- 传感器范围 (sensor_range): 影响感知能力
- 季节参数 (season_length, winter_metabolic_multiplier): 影响贮粮动机
- 疲劳系统 (fatigue_build_rate): 影响行为模式
- 奖励机制 (food_energy): 调整行为激励强度

## 输出格式要求
你必须返回一个严格的 JSON 对象，包含:
{
  "reasoning": "你对当前种群状态的分析和调整理由",
  "physics_config": {
    "metabolic_alpha": 数值,
    "metabolic_beta": 数值,
    "sensor_range": 数值,
    "season_length": 数值,
    "winter_metabolic_multiplier": 数值,
    "fatigue_build_rate": 数值,
    "food_energy": 数值,
    "enable_fatigue_system": 布尔值,
    "enable_thermal_sanctuary": 布尔值,
    "enable_morphological_computation": 布尔值
  },
  "confidence": 0.0-1.0 调整信心指数
}
"""


# ============================================================
# 2. 数据结构定义
# ============================================================

@dataclass
class PopulationStats:
    """种群统计数据"""
    generation: int
    population_size: int
    
    # 存活相关
    avg_steps_alive: float = 0.0
    max_steps_alive: int = 0
    
    # 死亡原因分析
    death_by_starvation: int = 0
    death_by_cold: int = 0
    death_by_age: int = 0
    death_by_unknown: int = 0
    
    # 行为统计
    total_food_eaten: int = 0
    total_food_carried: int = 0
    total_food_stored: int = 0
    avg_food_per_agent: float = 0.0
    
    # 脑复杂度
    avg_node_count: float = 0.0
    avg_edge_count: float = 0.0
    max_complexity: int = 0
    
    # 适应度
    avg_fitness: float = 0.0
    max_fitness: float = 0.0
    fitness_variance: float = 0.0
    
    # 生理状态
    avg_energy: float = 0.0
    avg_age: float = 0.0


@dataclass
class PhysicsConfig:
    """物理配置参数 - 支持嵌套配置"""
    # 基础参数
    metabolic_alpha: float = 0.05
    metabolic_beta: float = 0.05
    sensor_range: float = 200.0
    season_length: int = 50
    winter_metabolic_multiplier: float = 2.0
    fatigue_build_rate: float = 0.3
    food_energy: float = 30.0
    enable_fatigue_system: bool = False
    enable_thermal_sanctuary: bool = False
    enable_morphological_computation: bool = False
    
    # 嵌套配置 (通过__init__处理)
    def __init__(self, **kwargs):
        # 解包嵌套配置
        thermal = kwargs.pop('thermal_sanctuary', {})
        morph = kwargs.pop('morphological_computation', {})
        onto = kwargs.pop('ontogenetic_phase', {})
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # 保存嵌套配置
        self.thermal_sanctuary = thermal
        self.morphological_computation = morph
        self.ontogenetic_phase = onto
    
    def to_dict(self) -> dict:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                result[key] = value
        return result
    

# ============================================================
# 3. 战报生成器 - 从种群数据提取痛点
# ============================================================

class WarReportGenerator:
    """动态战报生成器 - 将numpy环境数据转化为LLM可理解的文字"""
    
    def __init__(self, env: Environment, population: Population, generation: int):
        self.env = env
        self.population = population
        self.generation = generation
        self.stats = self._collect_stats()
    
    def _collect_stats(self) -> PopulationStats:
        """从环境中收集统计数据"""
        stats = PopulationStats(
            generation=self.generation,
            population_size=len(self.population.agents)
        )
        
        if not self.population.agents:
            return stats
        
        # 收集所有Agent的数据
        steps_alive = []
        energies = []
        ages = []
        node_counts = []
        edge_counts = []
        fitnesses = []
        
        food_eaten_total = 0
        food_carried_total = 0
        food_stored_total = 0
        
        death_starvation = 0
        death_cold = 0
        death_age = 0
        death_unknown = 0
        
        for agent in self.population.agents:
            # 存活数据
            steps_alive.append(agent.steps_alive)
            energies.append(agent.internal_energy)
            ages.append(agent.age)
            fitnesses.append(agent.fitness)
            
            # 基因组复杂度
            genome_info = agent.genome.get_info()
            node_counts.append(genome_info['total_nodes'])
            edge_counts.append(genome_info['enabled_edges'])
            
            # 行为统计
            food_eaten_total += agent.food_eaten
            food_carried_total += agent.food_carried
            food_stored_total += agent.food_stored
            
            # 推断死亡原因
            if not agent.is_alive:
                # 能量耗尽 = 饿死
                if agent.internal_energy <= 0:
                    # 检查是否有冻死痕迹
                    if (hasattr(agent, 'body_temperature') and 
                        agent.body_temperature < getattr(self.env, 'cold_damage_threshold', -100)):
                        death_cold += 1
                    else:
                        death_starvation += 1
                # 老死
                elif hasattr(agent, 'age') and agent.age >= agent.max_age:
                    death_age += 1
                else:
                    death_unknown += 1
        
        # 计算统计数据
        stats.avg_steps_alive = np.mean(steps_alive) if steps_alive else 0
        stats.max_steps_alive = max(steps_alive) if steps_alive else 0
        
        stats.death_by_starvation = death_starvation
        stats.death_by_cold = death_cold
        stats.death_by_age = death_age
        stats.death_by_unknown = death_unknown
        
        stats.total_food_eaten = food_eaten_total
        stats.total_food_carried = food_carried_total
        stats.total_food_stored = food_stored_total
        stats.avg_food_per_agent = food_eaten_total / len(self.population.agents) if self.population.agents else 0
        
        stats.avg_node_count = np.mean(node_counts) if node_counts else 0
        stats.avg_edge_count = np.mean(edge_counts) if edge_counts else 0
        stats.max_complexity = max(node_counts) if node_counts else 0
        
        stats.avg_fitness = np.mean(fitnesses) if fitnesses else 0
        stats.max_fitness = max(fitnesses) if fitnesses else 0
        stats.fitness_variance = np.var(fitnesses) if len(fitnesses) > 1 else 0
        
        stats.avg_energy = np.mean(energies) if energies else 0
        stats.avg_age = np.mean(ages) if ages else 0
        
        return stats
    
    def generate_report(self) -> str:
        """生成完整的战报文本"""
        s = self.stats
        
        report = f"""## 📊 第 {s.generation} 纪元生存战报

### 🧬 种群概况
- 当前规模: {s.population_size} 个Agent
- 平均存活步数: {s.avg_steps_alive:.1f} 步
- 最长存活: {s.max_steps_alive} 步

### 💀 死亡原因分析
"""
        total_deaths = s.death_by_starvation + s.death_by_cold + s.death_by_age + s.death_by_unknown
        if total_deaths > 0:
            starv_pct = s.death_by_starvation / total_deaths * 100
            cold_pct = s.death_by_cold / total_deaths * 100
            age_pct = s.death_by_age / total_deaths * 100
            report += f"- 饿死: {s.death_by_starvation} ({starv_pct:.1f}%)\n"
            report += f"- 冻死: {s.death_by_cold} ({cold_pct:.1f}%)\n"
            report += f"- 老死: {s.death_by_age} ({age_pct:.1f}%)\n"
        else:
            report += "- 无死亡记录\n"
        
        report += f"""
### 🍎 行为统计
- 总计吃掉食物: {s.total_food_eaten}
- 总计携带食物: {s.total_food_carried}
- 总计贮粮次数: {s.total_food_stored}
- Agent平均进食: {s.avg_food_per_agent:.2f}
"""
        
        # 关键痛点检测
        pain_points = []
        
        if s.total_food_stored == 0 and s.generation > 50:
            pain_points.append("⚠️ 警示: 经过50+代演化，种群仍未能涌现贮粮行为！")
        
        if s.death_by_starvation > s.death_by_cold * 2 and s.generation > 30:
            pain_points.append("⚠️ 饥饿压力过大: 饿死数量是冻死的2倍以上，Agent可能还没学会贮粮就饿死了")
        
        if s.avg_steps_alive < 20 and s.generation > 20:
            pain_points.append("⚠️ 存活时间过短: 平均存活不到20步，说明环境压力过于极端")
        
        if s.fitness_variance < 10 and s.population_size > 5:
            pain_points.append("⚠️ 适应度停滞: 种群多样性过低，可能陷入局部最优")
        
        if s.avg_node_count < 5:
            pain_points.append("⚠️ 脑复杂度不足: 平均节点数<5，神经网络可能过于简单")
        
        if pain_points:
            report += "\n### 🚨 关键痛点\n"
            for pain in pain_points:
                report += f"- {pain}\n"
        
        report += f"""
### 🧠 神经复杂度
- 平均节点数: {s.avg_node_count:.1f}
- 平均边数: {s.avg_edge_count:.1f}
- 最大复杂度: {s.max_complexity}

### 📈 适应度
- 平均适应度: {s.avg_fitness:.2f}
- 最高适应度: {s.max_fitness:.2f}
- 适应度方差: {s.fitness_variance:.2f}

### ⚙️ 当前物理参数
- 代谢alpha: {getattr(self.env, 'metabolic_alpha', 'N/A')}
- 代谢beta: {getattr(self.env, 'metabolic_beta', 'N/A')}
- 传感器范围: {getattr(self.env, 'sensor_range', 'N/A')}
- 食物能量: {getattr(self.env, 'food_energy', 'N/A')}
- 季节长度: {getattr(self.env, 'season_length', 'N/A')}
- 冬季代谢倍率: {getattr(self.env, 'winter_metabolic_multiplier', 'N/A')}
- 疲劳系统: {getattr(self.env, 'fatigue_system_enabled', 'N/A')}
- 热力学庇护所: {getattr(self.env, 'thermal_sanctuary_enabled', 'N/A')}
- 形态计算: {getattr(self.env, 'morphological_computation_enabled', 'N/A')}
"""
        
        return report


# ============================================================
# v10.0: 参数平滑缓冲系统 (Environmental Inertia)
# ============================================================
class ParameterBuffer:
    """参数平滑缓冲 - 防止适应度悬崖"""
    
    def __init__(self, transition_generations: int = 15):
        self.transition_generations = transition_generations
        self.pending_params = {}
        self.start_values = {}
        self.current_values = {}
        self.generations_remaining = {}
    
    def request_change(self, param: str, target_value: float, current_value: float):
        self.pending_params[param] = target_value
        self.start_values[param] = current_value
        self.current_values[param] = current_value
        self.generations_remaining[param] = self.transition_generations
    
    def get_value(self, param: str, default: float) -> float:
        if param not in self.pending_params:
            return default
        remaining = self.generations_remaining.get(param, 0)
        if remaining <= 0:
            return self.pending_params[param]
        start = self.start_values[param]
        target = self.pending_params[param]
        progress = 1.0 - (remaining / self.transition_generations)
        return start + (target - start) * progress
    
    def tick(self):
        for param in list(self.generations_remaining.keys()):
            self.generations_remaining[param] -= 1


# ============================================================
# 4. API 请求与物理法则热更新
# ============================================================

class LLMDemiurge:
    """LLM 控制的物理法则调整器"""
    
    API_KEY = "sk-e88875abbd124a28897173587bb1f512"
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    FEISHU_WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/a581dcb1-8994-41e9-aa88-efcb9c0bf9b1"
    
    def __init__(self, config_path: str = "physics_config.json"):
        self.config_path = config_path
        self.current_config = self._load_config()
    
    def _load_config(self) -> PhysicsConfig:
        """加载物理配置"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                return PhysicsConfig(**data)
        return PhysicsConfig()
    
    def _save_config(self, config: PhysicsConfig):
        """保存物理配置 - 保留嵌套配置"""
        # 读取原始配置，保留嵌套结构
        try:
            with open(self.config_path, 'r') as f:
                original = json.load(f)
        except:
            original = {}
        
        # 更新基础参数
        for key in ['metabolic_alpha', 'metabolic_beta', 'sensor_range', 
                    'season_length', 'winter_metabolic_multiplier', 
                    'fatigue_build_rate', 'food_energy',
                    'enable_fatigue_system', 'enable_thermal_sanctuary', 
                    'enable_morphological_computation']:
            if hasattr(config, key):
                original[key] = getattr(config, key)
        
        # 保留嵌套配置
        if hasattr(config, 'thermal_sanctuary') and config.thermal_sanctuary:
            original['thermal_sanctuary'] = config.thermal_sanctuary
        if hasattr(config, 'morphological_computation') and config.morphological_computation:
            original['morphological_computation'] = config.morphological_computation
        if hasattr(config, 'ontogenetic_phase') and config.ontogenetic_phase:
            original['ontogenetic_phase'] = config.ontogenetic_phase
        
        # 添加注释
        original['notes'] = 'Updated by LLM Demiurge'
        
        with open(self.config_path, 'w') as f:
            json.dump(original, f, indent=2)
    
    def _send_to_deepseek(self, constitution: str, user_prompt: str) -> Dict:
        """发送请求到 DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": constitution},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        response = requests.post(
            self.API_URL, 
            headers=headers, 
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"API请求失败: {response.status_code} - {response.text}")
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def _notify_feishu(self, message: str):
        """发送飞书通知"""
        try:
            payload = {
                "msg_type": "text",
                "content": {"text": f"[EOE Demiurge] {message}"}
            }
            requests.post(self.FEISHU_WEBHOOK, json=payload, timeout=5)
        except Exception as e:
            print(f"飞书通知失败: {e}")
    
    def _validate_response(self, response_text: str) -> Optional[Dict]:
        """验证并解析 DeepSeek 的回复"""
        try:
            # 尝试提取 JSON
            # 查找 JSON 块
            import re
            
            # 尝试直接解析
            try:
                data = json.loads(response_text)
            except:
                # 尝试从 markdown 代码块中提取
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                else:
                    # 尝试找到 { } 包围的 JSON
                    json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                    else:
                        return None
            
            # 验证必需字段
            if 'physics_config' not in data:
                print("⚠️ 响应缺少 physics_config 字段")
                return None
            
            pc = data['physics_config']
            required_fields = [
                'metabolic_alpha', 'metabolic_beta', 'sensor_range',
                'season_length', 'winter_metabolic_multiplier', 'food_energy'
            ]
            
            for field in required_fields:
                if field not in pc:
                    print(f"⚠️ physics_config 缺少字段: {field}")
                    return None
            
            # 类型校验
            if not isinstance(pc['metabolic_alpha'], (int, float)):
                return None
            if pc['metabolic_alpha'] < 0 or pc['metabolic_alpha'] > 1:
                print("⚠️ metabolic_alpha 超出合理范围 [0, 1]")
                return None
            
            # ============================================================
            # v1.4: 参数边界校验 - 拒绝超过20%的调整!
            # ============================================================
            # 读取原始配置作为基准
            try:
                with open(self.config_path, 'r') as f:
                    original = json.load(f)
            except:
                original = {}
            
            max_change = 0.20  # 20% 最大调整幅度
            rejected = []
            
            for key in ['metabolic_alpha', 'metabolic_beta', 'sensor_range', 
                        'season_length', 'winter_metabolic_multiplier', 
                        'fatigue_build_rate', 'food_energy']:
                if key in pc and key in original:
                    orig = original[key]
                    new = pc[key]
                    if orig != 0:
                        change = abs(new - orig) / orig
                    else:
                        change = 1.0 if new != 0 else 0
                    
                    if change > max_change:
                        rejected.append(f"{key}: {orig}->{new} ({change*100:.1f}%)")
            
            if rejected:
                print(f"⚠️ 以下参数调整超过20%，被拒绝:")
                for r in rejected:
                    print(f"   - {r}")
                # 将被拒绝的参数恢复为原始值
                for key in rejected:
                    k = key.split(':')[0].strip()
                    if k in original:
                        pc[k] = original[k]
            
            return data
            
        except Exception as e:
            print(f"⚠️ JSON解析失败: {e}")
            return None
    
    def _apply_config(self, new_config: Dict, env: Environment) -> bool:
        """应用新的物理配置到环境 - 使用柔性截断避免死锁"""
        try:
            pc = new_config['physics_config']
            
            # 定义参数的合理范围
            # 格式: (key, min_ratio, max_ratio, default_min, default_max, special_clip)
            # special_clip=None 表示使用比例截断，否则使用绝对范围截断
            param_limits = {
                'metabolic_alpha': (0.5, 1.5, None),      # 允许 ±50%
                'metabolic_beta': (0.5, 1.5, None),       # 允许 ±50%
                'sensor_range': (0.5, 1.5, None),         # 允许 ±50%
                'season_length': (0.7, 1.3, None),        # 允许 ±30% (季节不宜剧烈变化)
                'winter_metabolic_multiplier': (0.8, 1.2, (1.0, 3.0)),  # 允许 ±20% 或绝对范围 [1.0, 3.0]
                'fatigue_build_rate': (0.5, 1.5, None),   # 允许 ±50%
                'food_energy': (0.3, 3.0, (5.0, 200.0)),  # 允许 -70% ~ +200% 或绝对范围 [5.0, 200.0]
                'winter_temperature': (0.5, 1.5, (-20.0, 30.0)),  # 允许 ±50% 或绝对范围
            }
            
            # 逐参数应用柔性截断
            clipped_params = {}
            for key, value in pc.items():
                if key in param_limits and value is not None:
                    old_value = getattr(env, key, None)
                    if old_value is None:
                        old_value = value  # 如果环境没有这个参数，使用新值
                    
                    min_ratio, max_ratio, special_clip = param_limits[key]
                    
                    # 计算截断边界
                    lower_bound = old_value * min_ratio
                    upper_bound = old_value * max_ratio
                    
                    # 如果有特殊绝对范围限制，使用更宽松的边界
                    if special_clip is not None:
                        abs_min, abs_max = special_clip
                        lower_bound = max(lower_bound, abs_min)
                        upper_bound = min(upper_bound, abs_max)
                    
                    # 柔性截断
                    if value < lower_bound:
                        print(f"⚠️ 参数 {key} 调整幅度过大: {old_value:.4f} -> {value:.4f}, 已截断至 {lower_bound:.4f}")
                        clipped_params[key] = lower_bound
                    elif value > upper_bound:
                        print(f"⚠️ 参数 {key} 调整幅度过大: {old_value:.4f} -> {value:.4f}, 已截断至 {upper_bound:.4f}")
                        clipped_params[key] = upper_bound
                    else:
                        clipped_params[key] = value
                else:
                    clipped_params[key] = value
            
            # 应用截断后的配置
            for key, value in clipped_params.items():
                # 修复布尔值解析：将字符串 "true"/"false" 转换为 Python bool
                if isinstance(value, str):
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                
                # 数值型参数直接设置
                if hasattr(env, key) and not key.startswith('enable_'):
                    setattr(env, key, value)
                
                # 布尔型参数需要调用专门的 enable/disable 方法
                elif key == 'enable_fatigue_system' and value is not None:
                    if value and not env.fatigue_system_enabled:
                        env.enable_fatigue_system(enabled=True, max_fatigue=100.0, fatigue_build_rate=0.3)
                        print(f"[INFO] 疲劳系统已启用")
                    elif not value and env.fatigue_system_enabled:
                        env.fatigue_system_enabled = False
                        print(f"[INFO] 疲劳系统已禁用")
                
                elif key == 'enable_thermal_sanctuary' and value is not None:
                    if value and not env.thermal_sanctuary_enabled:
                        env.enable_thermal_sanctuary(enabled=True)
                        print(f"[INFO] 热力学庇护所已启用")
                    elif not value and env.thermal_sanctuary_enabled:
                        env.thermal_sanctuary_enabled = False
                        print(f"[INFO] 热力学庇护所已禁用")
                
                elif key == 'enable_morphological_computation' and value is not None:
                    if value and not env.morphological_computation_enabled:
                        env.enable_morphological_computation(enabled=True)
                        print(f"[INFO] 形态计算已启用")
                    elif not value and env.morphological_computation_enabled:
                        env.morphological_computation_enabled = False
                        print(f"[INFO] 形态计算已禁用")
            
            # 打印成功日志
            print(f"[SUCCESS] 物理参数已热更新: alpha={env.metabolic_alpha}, beta={env.metabolic_beta}, "
                  f"sensor={env.sensor_range}, food_energy={env.food_energy}, season={env.season_length}")
            
            # 保存到文件
            self._save_config(PhysicsConfig(**clipped_params))
            
            return True
            
        except Exception as e:
            import traceback
            print(f"⚠️ 配置应用失败: {e}")
            print(f"⚠️ 异常堆栈: {traceback.format_exc()}")
            return False
    
    def run_epoch(self, env: Environment, population: Population, generation: int) -> bool:
        """
        运行一个纪元的分析和建议
        
        Args:
            env: 环境对象
            population: 种群对象
            generation: 当前代数
        
        Returns:
            bool: 是否成功应用新配置
        """
        print(f"\n{'='*60}")
        print(f"🔮 第 {generation} 纪元 LLM Demiurge 启动")
        print(f"{'='*60}")
        
        # 生成战报
        war_report = WarReportGenerator(env, population, generation)
        report_text = war_report.generate_report()
        
        print(report_text)
        
        # 构建用户提示
        user_prompt = f"""基于以下第 {generation} 纪元的种群生存战报，请分析当前演化瓶颈，并给出物理参数调整建议:

{report_text}

请分析种群面临的核心问题，并返回一个严格的 JSON 对象来调整物理参数。记住:
1. 调整必须是微调 (< 20%)
2. 目标是创造"适应度缓坡"而非"断崖"
3. 如果贮粮行为未涌现，考虑降低饥饿压力或增加贮粮奖励
"""
        
        # 发送到 DeepSeek
        print("\n📡 正在咨询 DeepSeek 物理法则...")
        
        try:
            response = self._send_to_deepseek(BASE_CONSTITUTION, user_prompt)
            print(f"💬 DeepSeek 回复: {response[:500]}...")
            
        except Exception as e:
            error_msg = f"API请求失败: {str(e)}"
            print(f"❌ {error_msg}")
            self._notify_feishu(error_msg)
            return False
        
        # 验证响应
        validated = self._validate_response(response)
        if not validated:
            print("❌ DeepSeek 响应格式校验失败，放弃应用")
            return False
        
        # 应用配置
        print(f"\n✅ 校验通过，应用新配置...")
        success = self._apply_config(validated, env)
        
        if success:
            print(f"✅ 配置已热更新!")
            print(f"💡 调整理由: {validated.get('reasoning', 'N/A')}")
            print(f"📊 信心指数: {validated.get('confidence', 'N/A')}")
            
            # 通知飞书
            self._notify_feishu(f"第{generation}代配置已更新: {validated.get('reasoning', '')[:50]}...")
        else:
            print("❌ 配置应用失败")
            self._notify_feishu(f"第{generation}代配置应用失败")
        
        return success


# ============================================================
# 5. 主入口 - 运行 Demiurge Loop
# ============================================================
# 5.5. 纪元报告生成器 - 四大维度数据提取
# ============================================================

def generate_epoch_report(epoch_data: Dict[str, Any]) -> str:
    """
    生成纪元报告 - 四大维度信息提取
    
    Args:
        epoch_data: 包含以下键的字典:
            - generation: 当前代数
            - agents: List[Agent] - 所有Agent对象
            - env: Environment - 环境对象
            - start_step: 纪元起始步数
            - end_step: 纪元结束步数
    
    Returns:
        str: 格式化的报告文本，直接用于发送给DeepSeek
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.eoe.agent import Agent
    
    generation = epoch_data.get('generation', 0)
    agents = epoch_data.get('agents', [])
    env = epoch_data.get('env', None)
    
    # ========== 1. 生存与死亡 ==========
    alive_agents = [a for a in agents if a.is_alive]
    dead_agents = [a for a in agents if not a.is_alive]
    
    steps_alive = [a.steps_alive for a in agents if hasattr(a, 'steps_alive')]
    avg_steps = np.mean(steps_alive) if steps_alive else 0
    max_steps = max(steps_alive) if steps_alive else 0
    min_steps = min(steps_alive) if steps_alive else 0
    
    # 死亡原因分析
    death_starvation = 0  # 饿死 (能量<=0)
    death_cold = 0       # 冻死 (体温过低)
    death_age = 0        # 老死
    death_other = 0      # 其他
    
    if env and hasattr(env, 'cold_damage_threshold'):
        cold_threshold = env.cold_damage_threshold
    else:
        cold_threshold = -5.0
    
    for agent in dead_agents:
        if hasattr(agent, 'age') and agent.age >= getattr(agent, 'max_age', 100):
            death_age += 1
        elif hasattr(agent, 'body_temperature') and agent.body_temperature < cold_threshold:
            death_cold += 1
        elif getattr(agent, 'internal_energy', 0) <= 0:
            death_starvation += 1
        else:
            death_other += 1
    
    total_deaths = len(dead_agents)
    starvation_ratio = (death_starvation / total_deaths * 100) if total_deaths > 0 else 0
    cold_ratio = (death_cold / total_deaths * 100) if total_deaths > 0 else 0
    
    # ========== 2. 行为里程碑 ==========
    total_food_touched = 0      # 碰触食物次数
    total_food_carried_steps = 0  # 携带食物总步数
    total_food_stored = 0       # 成功搬入巢穴的食物数
    
    for agent in agents:
        total_food_touched += getattr(agent, 'food_eaten', 0)
        total_food_touched += getattr(agent, 'food_carried', 0)
        total_food_carried_steps += getattr(agent, 'food_carried', 0) * getattr(agent, 'steps_alive', 0)
        total_food_stored += getattr(agent, 'food_stored', 0)
    
    # ========== 3. 能量收支 ==========
    energy_in = []   # 摄入热量
    energy_out = []  # 消耗热量
    
    for agent in agents:
        if hasattr(agent, 'energy_gained'):
            energy_in.append(agent.energy_gained)
        if hasattr(agent, 'energy_spent'):
            energy_out.append(agent.energy_spent)
    
    avg_energy_in = np.mean(energy_in) if energy_in else 0
    avg_energy_out = np.mean(energy_out) if energy_out else 0
    total_energy_in = sum(energy_in)
    total_energy_out = sum(energy_out)
    
    # ========== 4. 脑容积趋势 + 脑部热图 ==========
    node_counts = []
    edge_counts = []
    elite_nodes = []  # 精英个体的节点数
    elite_edges = []  # 精英个体的边数
    operator_counts = {
        'SENSOR': 0, 'DELAY': 0, 'MOTOR': 0, 
        'META': 0, 'COMPOSITE': 0, 'THRESHOLD': 0
    }
    
    # 找出精英个体 (适应度前20%)
    fitnesses = [getattr(a, 'fitness', 0) for a in agents if hasattr(a, 'fitness')]
    if fitnesses:
        threshold = np.percentile(fitnesses, 80)  # 前20%
    else:
        threshold = 0
    
    for agent in agents:
        if hasattr(agent, 'genome') and agent.genome:
            info = agent.genome.get_info()
            node_counts.append(info.get('total_nodes', 0))
            edge_counts.append(info.get('enabled_edges', 0))
            
            # 统计精英个体
            fitness = getattr(agent, 'fitness', 0)
            if fitness >= threshold and fitness > 0:
                elite_nodes.append(info.get('total_nodes', 0))
                elite_edges.append(info.get('enabled_edges', 0))
            
            # 统计算子分布
            if 'operator_distribution' in info:
                for op, count in info['operator_distribution'].items():
                    operator_counts[op] = operator_counts.get(op, 0) + count
    
    avg_nodes = np.mean(node_counts) if node_counts else 0
    avg_edges = np.mean(edge_counts) if edge_counts else 0
    max_nodes = max(node_counts) if node_counts else 0
    max_edges = max(edge_counts) if edge_counts else 0
    
    # 精英个体脑部统计
    avg_elite_nodes = np.mean(elite_nodes) if elite_nodes else 0
    avg_elite_edges = np.mean(elite_edges) if elite_edges else 0
    
    # 连接密度 = 边数 / (节点数*(节点数-1)/2)
    connection_density = avg_edges / max(1, avg_nodes * (avg_nodes - 1) / 2) if avg_nodes > 1 else 0
    
    # 复杂度变化趋势
    complexity_trend = "上升" if avg_nodes > 5 else "稳定"
    if avg_nodes < 3:
        complexity_trend = "下降/简化"
    
    # ========== 5. 适应度瓶颈检测 ==========
    bottleneck_warning = ""
    if total_food_stored == 0 and generation > 50:
        # 检查是否是冬天太早或吸附太弱
        winter_length = getattr(env, 'season_length', 50) if env else 50
        winter_multiplier = getattr(env, 'winter_metabolic_multiplier', 2.0) if env else 2.0
        adhesion_range = getattr(env, 'adhesion_range', 2.0) if env else 2.0
        
        bottleneck_warning = f"""
================================================================================
                         🚨 适应度瓶颈警告
================================================================================

经过 {generation} 代演化，种群尚未发现贮粮机制！

可能原因分析:
1. 冬天过早降临: 季节长度={winter_length}帧, 冬季代谢倍率={winter_multiplier}x
   → Agent可能在学会贮粮前就冻死

2. 吸附力过弱: 吸附范围={adhesion_range}
   → Agent可能无法有效携带食物

3. 代谢压力过大: 当前环境可能不适合长链路行为涌现

建议调整方向:
- 延长夏天，给Agent足够时间学习贮粮
- 增强吸附力（提高adhesion_range）
- 降低冬季代谢惩罚
- 或增强贮粮奖励（food_stored奖励）
"""
    
    # 累积贮粮统计（如果有历史数据）
    cumulative_stored = total_food_stored
    if hasattr(env, 'nest_stored_food'):
        cumulative_stored = env.nest_stored_food
    
    if cumulative_stored == 0 and generation > 100:
        bottleneck_warning += f"""

⚠️ 重要: 经过100+代仍无贮粮行为，这是关键的演化瓶颈！
建议: 考虑大幅降低冬天压力或增加贮粮激励
"""
    
    # ========== 构建报告文本 ==========
    report = f"""
================================================================================
                     📊 第 {generation} 纪元 - 生存战报
================================================================================

【维度一】🩺 生存与死亡
────────────────────────────────────────────────────────────────────────────────
  存活Agent数: {len(alive_agents)} / {len(agents)}
  平均存活步数: {avg_steps:.1f} 步
  最长存活: {max_steps} 步 | 最短存活: {min_steps} 步
  
  死亡分析 (总计 {total_deaths} 具):
    • 饿死: {death_starvation} ({starvation_ratio:.1f}%)
    • 冻死: {death_cold} ({cold_ratio:.1f}%)
    • 老死: {death_age}
    • 其他: {death_other}
  
  ⚡ 关键信号: {"饿死占比过高! Agent可能还没学会贮粮就饿死了" if starvation_ratio > 50 else "死亡分布正常"}

【维度二】🎯 行为里程碑
────────────────────────────────────────────────────────────────────────────────
  碰触食物总数: {total_food_touched}
  携带食物总步数: {total_food_carried_steps}
  成功搬入巢穴: {total_food_stored}
  
  贮粮效率: {total_food_stored / max(1, total_food_touched) * 100:.1f}%
  
  ⚡ 关键信号: {"❌ 贮粮行为未涌现! 尚未发现将食物搬入巢穴的行为" if total_food_stored == 0 and generation > 50 else "✅ 已有贮粮行为"}

【维度三】⚡ 能量收支
────────────────────────────────────────────────────────────────────────────────
  总能量摄入: {total_energy_in:.1f}
  总能量消耗: {total_energy_out:.1f}
  净能量: {total_energy_in - total_energy_out:.1f}
  
  平均每Agent摄入: {avg_energy_in:.1f}
  平均每Agent消耗: {avg_energy_out:.1f}
  
  能量效率: {total_energy_in / max(1, total_energy_out) * 100:.1f}%

【维度四】🧠 脑容积趋势 + 脑部热图
────────────────────────────────────────────────────────────────────────────────
  平均节点数: {avg_nodes:.1f}
  平均边数: {avg_edges:.1f}
  最大节点: {max_nodes} | 最大边: {max_edges}
  连接密度: {connection_density:.2%}
  
  复杂度趋势: {complexity_trend}
  
  精英个体(前20%)脑部统计:
    • 平均节点: {avg_elite_nodes:.1f}
    • 平均边数: {avg_elite_edges:.1f}
  
  算子分布 (Operator Distribution):
    • SENSOR: {operator_counts.get('SENSOR', 0)}
    • DELAY: {operator_counts.get('DELAY', 0)}
    • MOTOR: {operator_counts.get('MOTOR', 0)}
    • META: {operator_counts.get('META', 0)}
    • THRESHOLD: {operator_counts.get('THRESHOLD', 0)}
  
  {"⚠️ 脑复杂度偏低，神经网络可能过于简单" if avg_nodes < 5 else "✅ 神经复杂度正常"}

{bottleneck_warning}

================================================================================
                           当前物理环境参数
================================================================================
"""
    
    # 格式化bottleneck_warning
    bottleneck_warning = bottleneck_warning if bottleneck_warning else ""
    
    # 添加当前物理参数
    if env:
        report += f"""
  代谢alpha: {getattr(env, 'metabolic_alpha', 'N/A')}
  代谢beta: {getattr(env, 'metabolic_beta', 'N/A')}
  传感器范围: {getattr(env, 'sensor_range', 'N/A')}
  食物能量: {getattr(env, 'food_energy', 'N/A')}
  季节长度: {getattr(env, 'season_length', 'N/A')}
  冬季代谢倍率: {getattr(env, 'winter_metabolic_multiplier', 'N/A')}
  
  疲劳系统: {getattr(env, 'fatigue_system_enabled', False)}
  热力学庇护所: {getattr(env, 'thermal_sanctuary_enabled', False)}
  形态计算: {getattr(env, 'morphological_computation_enabled', False)}
"""
    
    report += """
================================================================================
                         请分析以上数据并给出调整建议
================================================================================
"""
    
    return report


# ============================================================

def run_demiurge_loop(
    env: Environment,
    population: Population,
    start_generation: int = 0,
    end_generation: int = 500,
    epoch_interval: int = 100
):
    """
    运行完整的 Demiurge Loop
    
    Args:
        env: EOE 环境
        population: 种群
        start_generation: 起始代数
        end_generation: 结束代数
        epoch_interval: 每多少代运行一次LLM分析
    """
    demiurge = LLMDemiurge()
    
    for generation in range(start_generation, end_generation + 1):
        # 每隔 epoch_interval 代运行一次 LLM 分析
        if generation > 0 and generation % epoch_interval == 0:
            success = demiurge.run_epoch(env, population, generation)
            if success:
                print(f"✅ 第 {generation} 纪元完成，配置已更新")
            else:
                print(f"⚠️ 第 {generation} 纪元配置更新失败，使用现有配置继续")
        
        # 正常演化步骤
        # (这里应该调用 population.step() 等)
        # ...
    
    print(f"\n🎉 Demiurge Loop 完成!")


# ============================================================
# 6. 独立运行脚本
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EOE LLM Demiurge Loop")
    parser.add_argument("--config", type=str, default="physics_config.json", help="配置文件路径")
    parser.add_argument("--generation", type=int, default=100, help="当前代数")
    parser.add_argument("--dry-run", action="store_true", help="仅生成报告，不调用API")
    args = parser.parse_args()
    
    # 模拟环境（实际使用时从主程序传入）
    print("⚠️ 这是一个独立测试脚本")
    print("实际使用时需要从主程序传入 env 和 population 对象")
    print("\n示例用法:")
    print("  from scripts.llm_demiurge_loop import LLMDemiurge, WarReportGenerator")
    print("  ")
    print("  demiurge = LLMDemiurge()")
    print("  war_report = WarReportGenerator(env, population, generation=100)")
    print("  print(war_report.generate_report())")
    print("  ")
    print("  demiurge.run_epoch(env, population, generation=100)")