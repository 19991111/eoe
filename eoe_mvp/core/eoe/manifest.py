#!/usr/bin/env python3
"""
EOE v12.5: 物理法则注册系统 (Physics Manifest)

核心概念:
- PhysicsManifest: 带类型检查的配置Schema
- MechanismRegistry: 物理法则注册器
- PhysicalLaw: 独立的"法则"类,可通过配置开关

不再使用散落的 enable_* 开关,而是通过注册表统一管理
"""
from __future__ import annotations
from typing import Dict, Any, List, Type, Optional, Callable
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PhysicsManifest:
    """
    统一的物理法则配置清单
    所有参数必须从这里获取唯一真值
    """
    # ==================== 基础物理常数 ====================
    metabolic_alpha: float = 0.003
    metabolic_beta: float = 0.003
    initial_energy: float = 150.0
    sensor_range: float = 40.0
    
    # ==================== 食物与资源 ====================
    food_energy: float = 80.0
    food_escape_enabled: bool = False
    food_escape_speed: float = 0.6
    
    # ==================== v11.0 三大机制 ====================
    energy_decay_k: float = 0.00005      # 代谢熵增
    port_interference_gamma: float = 1.5 # 端口干涉
    season_jitter: float = 0.05          # 参数波动
    nest_tax: float = 0.08               # 入库税
    
    # ==================== 季节系统 ====================
    seasonal_cycle: bool = True
    season_length: int = 35
    winter_metabolic_multiplier: float = 1.2
    winter_food_multiplier: float = 0.0
    
    # ==================== 热力学庇护所 ====================
    thermal_sanctuary_enabled: bool = True
    summer_temperature: float = 28.0
    winter_temperature: float = -10.0
    food_heat_output: float = 15.0
    food_heat_radius: float = 15.0
    nest_insulation: float = 0.02
    
    # ==================== 疲劳系统 ====================
    fatigue_system_enabled: bool = True
    max_fatigue: float = 50.0
    fatigue_build_rate: float = 0.15
    sleep_danger_prob: float = 0.95
    enable_wakeup_hunger: bool = True
    enable_sleep_drop: bool = True
    
    # ==================== 形态计算 ====================
    morphological_computation_enabled: bool = False
    adhesion_range: float = 3.5
    carry_speed_penalty: float = 0.65
    discharge_threshold: float = 0.7
    
    # ==================== 压痕系统 ====================
    stigmergic_friction_enabled: bool = False
    stigmergy_decay_rate: float = 0.001
    stigmergy_max_impressions: int = 1000
    
    # ==================== 压痕系统 ====================
    stigmergic_friction_enabled: bool = False
    
    # ==================== 发育相变 ====================
    ontogenetic_phase_enabled: bool = False
    juvenile_duration: int = 60
    
    # ==================== 红皇后竞争 ====================
    red_queen_enabled: bool = False
    n_rivals: int = 3
    rival_refresh_interval: int = 40
    
    # ==================== 压力梯度熔炉 ====================
    crucible_enabled: bool = True
    complexity_premium: float = 1.5
    
    # ==================== v13.0 机制开关 (与 config/mechanisms.yaml 同步) ====================
    # 感知系统
    sensor_epf: bool = True
    sensor_kif: bool = True
    sensor_isf: bool = True
    sensor_energy: bool = True
    
    # 运动系统
    actuator_thrust: bool = True
    actuator_permeability: bool = True
    actuator_defense: bool = True
    
    # 信号系统
    signal_deposit: bool = True
    signal_receive: bool = True
    
    # 能量系统
    energy_extraction: bool = True
    energy_depletable: bool = True
    energy_infinite: bool = False
    energy_metabolic: bool = True
    energy_death: bool = True
    
    # 进化系统
    evolution_selection: bool = True
    evolution_mutation: bool = True
    evolution_crossover: bool = True
    evolution_isf_decay: bool = True
    evolution_enabled: bool = True
    
    # 环境系统
    env_epf: bool = True
    env_kif: bool = True
    env_isf: bool = True
    env_world_bounds: bool = True
    env_source_respawn: bool = True
    env_diffusion: bool = True
    env_gradient: bool = True
    
    # 物理系统
    physics_collision: bool = True
    physics_boundary_wrap: bool = False
    physics_velocity_decay: bool = True
    physics_friction: bool = True
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> PhysicsManifest:
        """从字典创建Manifest,带类型检查"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        
        # 过滤未知字段
        filtered = {k: v for k, v in config.items() if k in valid_fields}
        
        return cls(**filtered)
    
    @classmethod
    def from_yaml(cls, preset: str = "full") -> "PhysicsManifest":
        """从 YAML 配置文件加载 (config/mechanisms.yaml)
        
        Args:
            preset: 预设名称 (full/simple/no_signal/infinite_energy/no_evolution/wrap_world)
        """
        import os
        import yaml
        
        # 查找 YAML 文件 (eoe_mvp/config/mechanisms.yaml)
        # manifest.py 在 eoe_mvp/core/eoe/ 下
        manifest_dir = os.path.dirname(os.path.dirname(__file__))  # eoe_mvp/core/
        config_dir = os.path.dirname(manifest_dir)  # eoe_mvp/
        config_dir = os.path.join(config_dir, "config")  # eoe_mvp/config/
        yaml_path = os.path.join(config_dir, "mechanisms.yaml")
        
        if not os.path.exists(yaml_path):
            print(f"[PhysicsManifest] YAML not found: {yaml_path}, using defaults")
            return cls()
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # 获取预设或默认配置
        if preset and preset in data.get('presets', {}):
            config = data['presets'][preset]
        else:
            config = {k: v for k, v in data.items() if k != 'presets'}
        
        # 展平嵌套字典为 manifest 字段
        result = {}
        
        # 映射表: YAML路径 -> manifest字段
        mapping = {
            # 感知
            'sensor.epf': 'sensor_epf',
            'sensor.kif': 'sensor_kif',
            'sensor.isf': 'sensor_isf',
            'sensor.energy': 'sensor_energy',
            # 运动
            'actuator.thrust': 'actuator_thrust',
            'actuator.permeability': 'actuator_permeability',
            'actuator.defense': 'actuator_defense',
            # 信号
            'signal.deposit': 'signal_deposit',
            'signal.receive': 'signal_receive',
            # 能量
            'energy.extraction': 'energy_extraction',
            'energy.depletable': 'energy_depletable',
            'energy.infinite': 'energy_infinite',
            'energy.metabolic': 'energy_metabolic',
            'energy.death': 'energy_death',
            # 进化
            'evolution.selection': 'evolution_selection',
            'evolution.mutation': 'evolution_mutation',
            'evolution.crossover': 'evolution_crossover',
            'evolution.isf_decay': 'evolution_isf_decay',
            'evolution.enabled': 'evolution_enabled',
            # 环境
            'environment.epf': 'env_epf',
            'environment.kif': 'env_kif',
            'environment.isf': 'env_isf',
            'environment.world_bounds': 'env_world_bounds',
            'environment.source_respawn': 'env_source_respawn',
            'environment.diffusion': 'env_diffusion',
            'environment.gradient': 'env_gradient',
            # 物理
            'physics.collision': 'physics_collision',
            'physics.boundary_wrap': 'physics_boundary_wrap',
            'physics.velocity_decay': 'physics_velocity_decay',
            'physics.friction': 'physics_friction',
        }
        
        # 从嵌套配置中提取值
        for yaml_path, manifest_field in mapping.items():
            parts = yaml_path.split('.')
            value = config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
            if value is not None:
                result[manifest_field] = bool(value)
        
        print(f"[PhysicsManifest] Loaded preset: {preset}")
        return cls(**result)
    
    @classmethod
    def from_json(cls, filepath: str) -> PhysicsManifest:
        """从JSON文件加载"""
        import json
        with open(filepath) as f:
            config = json.load(f)
        return cls.from_dict(config)
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            f.name: getattr(self, f.name)
            for f in self.__dataclass_fields__.values()
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """安全获取参数"""
        return getattr(self, key, default)


class PhysicalLaw:
    """
    物理法则基类
    每个法则负责特定的物理计算,可以独立开启/关闭
    """
    name: str = "base_law"
    
    def __init__(self, manifest: PhysicsManifest):
        self.manifest = manifest
        self.enabled = True
    
    def apply(self, agents: List, world: Dict) -> None:
        """应用法则到所有Agent和世界状态"""
        raise NotImplementedError
    
    def reset(self) -> None:
        """重置法则状态(每代开始时调用)"""
        pass


class MechanismRegistry:
    """
    机制注册器
    负责管理所有物理法则的注册、启用、禁用
    """
    def __init__(self, manifest: PhysicsManifest):
        self.manifest = manifest
        self._laws: Dict[str, PhysicalLaw] = {}
        self._enabled_laws: List[str] = []
    
    def register(self, name: str, law_class: Type[PhysicalLaw], 
                 enabled: bool = True) -> PhysicalLaw:
        """注册一个新的物理法则"""
        law = law_class(self.manifest)
        law.name = name
        law.enabled = enabled and self._is_law_enabled(law_class)
        
        self._laws[name] = law
        
        if law.enabled:
            self._enabled_laws.append(name)
        
        return law
    
    def _is_law_enabled(self, law_class: Type[PhysicalLaw]) -> bool:
        """检查法则是否应该启用"""
        # 根据法则类名自动匹配manifest中的开关
        law_name = law_class.__name__.replace('Law', '').lower()
        
        # 常见映射: 类名(去掉Law后) → manifest中的开关字段
        law_base = law_class.__name__
        if law_base.endswith('Law'):
            law_base = law_base[:-3]  # 去掉Law后缀
        
        mapping = {
            'Metabolism': 'metabolic_alpha',  # 代谢始终启用,alpha>0
            'SeasonalCycle': 'seasonal_cycle',
            'ThermalSanctuary': 'thermal_sanctuary_enabled',
            'FatigueSystem': 'fatigue_system_enabled',
            'MorphologicalComputation': 'morphological_computation_enabled',
            'StigmergicFriction': 'stigmergic_friction_enabled',
            'OntogeneticPhase': 'ontogenetic_phase_enabled',
            'RedQueen': 'red_queen_enabled',
            'Crucible': 'crucible_enabled',
            'PortInterference': 'port_interference_gamma',  # gamma>0启用
        }
        
        key = mapping.get(law_base, f"{law_base.lower()}_enabled")
        value = getattr(self.manifest, key, None)
        
        # 特殊处理: metabolic_alpha > 0 表示启用
        if key == 'metabolic_alpha':
            return value is not None and value > 0
        
        # 特殊处理: port_interference_gamma > 0 表示启用
        if key == 'port_interference_gamma':
            return value is not None and value > 0
        
        # 布尔值检查
        return bool(value)
    
    def get_law(self, name: str) -> Optional[PhysicalLaw]:
        """获取指定法则"""
        return self._laws.get(name)
    
    def apply_all(self, agents: List, world: Dict) -> None:
        """应用所有启用的法则"""
        for name in self._enabled_laws:
            law = self._laws.get(name)
            if law and law.enabled:
                law.apply(agents, world)
    
    def list_laws(self) -> List[Dict[str, Any]]:
        """列出所有已注册的法则"""
        return [
            {
                'name': name,
                'enabled': law.enabled,
                'class': law.__class__.__name__
            }
            for name, law in self._laws.items()
        ]
    
    def enable(self, name: str) -> None:
        """启用指定法则"""
        if name in self._laws:
            self._laws[name].enabled = True
            if name not in self._enabled_laws:
                self._enabled_laws.append(name)
    
    def disable(self, name: str) -> None:
        """禁用指定法则"""
        if name in self._laws:
            self._laws[name].enabled = False
            if name in self._enabled_laws:
                self._enabled_laws.remove(name)


# ============================================================
# 预定义物理法则 (示例)
# ============================================================

class MetabolismLaw(PhysicalLaw):
    """代谢法则"""
    name = "metabolism"
    
    def apply(self, agents: List, world: Dict) -> None:
        for agent in agents:
            if not agent.is_alive:
                continue
            
            # 基础代谢消耗
            nodes = len(agent.genome.nodes)
            base_cost = self.manifest.metabolic_alpha * nodes + self.manifest.metabolic_beta
            
            # v11.0: 代谢熵增 (能量越高,挥发越快)
            if self.manifest.energy_decay_k > 0:
                decay = self.manifest.energy_decay_k * (agent.internal_energy ** 2)
                base_cost += decay
            
            agent.internal_energy -= base_cost * world.get('dt', 1.0)


class SeasonalCycleLaw(PhysicalLaw):
    """
    季节循环法则
    
    周期性执行 (每season_length帧更新一次)
    属于"可选/周期性法则"类别
    """
    name = "seasonal_cycle"
    
    def __init__(self, manifest):
        super().__init__(manifest)
        self.current_season = "summer"
        self.season_frame = 0
        self._last_update_frame = -1  # 缓存优化
    
    def reset(self) -> None:
        """重置季节状态"""
        self.current_season = "summer"
        self.season_frame = 0
        self._last_update_frame = -1
    
    def apply(self, agents: List, world: Dict) -> None:
        """应用季节法则"""
        if not self.manifest.seasonal_cycle:
            return
        
        self.season_frame += 1
        season_len = self.manifest.season_length
        
        # 季节切换
        if self.season_frame >= season_len:
            self.season_frame = 0
            self.current_season = "winter" if self.current_season == "summer" else "summer"
            self._last_update_frame = self.season_frame
        
        # 更新世界状态
        world['season'] = self.current_season
        world['temperature'] = (
            self.manifest.summer_temperature 
            if self.current_season == "summer" 
            else self.manifest.winter_temperature
        )
        
        # 冬天代谢惩罚 (只在新季节开始时计算一次)
        if self.current_season == "winter" and self._last_update_frame == self.season_frame:
            penalty = (self.manifest.winter_metabolic_multiplier - 1.0) * 0.01
            for agent in agents:
                if agent.is_alive:
                    agent.internal_energy *= (1.0 - penalty)
    
    def get_current_season(self) -> str:
        """获取当前季节 (供外部调用)"""
        return self.current_season
    
    def is_winter(self) -> bool:
        """快速检查是否冬天"""
        return self.current_season == "winter"


class FatigueSystemLaw(PhysicalLaw):
    """
    疲劳系统法则
    
    每步执行,但逻辑简单
    属于"可选法则"类别,可迁移到PhysicalLaw
    """
    name = "fatigue"
    
    def apply(self, agents: List, world: Dict) -> None:
        if not self.manifest.fatigue_system_enabled:
            return
        
        # 缓存参数以减少属性访问
        max_fatigue = self.manifest.max_fatigue
        build_rate = self.manifest.fatigue_build_rate
        
        for agent in agents:
            if not agent.is_alive:
                continue
            
            # 移动时累积疲劳
            motion = agent.port_motion
            if motion != 0:
                agent.fatigue += build_rate * abs(motion)
            
            # 疲劳惩罚: 速度降低
            if agent.fatigue > 0:
                speed_penalty = min(agent.fatigue / max_fatigue, 1.0)
                agent.port_motion *= (1.0 - speed_penalty * 0.5)
            
            # 超过最大值截断
            if agent.fatigue > max_fatigue:
                agent.fatigue = max_fatigue
    
    def is_fatigued(self, agent) -> bool:
        """检查Agent是否疲劳"""
        return agent.fatigue > self.manifest.max_fatigue * 0.8


class PortInterferenceLaw(PhysicalLaw):
    """
    端口干涉法则 (v11.0)
    
    多端口同时激活时的额外代谢成本
    属于"可选法则"
    """
    name = "port_interference"
    
    def apply(self, agents: List, world: Dict) -> None:
        if self.manifest.port_interference_gamma <= 0:
            return
        
        gamma = self.manifest.port_interference_gamma
        
        for agent in agents:
            if not agent.is_alive:
                continue
            
            # 计算端口激活程度
            ports = [
                abs(agent.port_motion),
                abs(agent.port_offense),
                abs(agent.port_defense),
                abs(agent.port_repair)
            ]
            total_port = sum(ports)
            
            if total_port > 0:
                # 干涉成本 = (端口总和)^gamma * 0.001
                interference_cost = (total_port ** gamma) * 0.001
                agent.internal_energy -= interference_cost


class MorphologicalComputationLaw(PhysicalLaw):
    """
    形态计算法则
    
    核心理念: "身体分担大脑的工作"
    - 吸附能力: 物理接触自动携带
    - 碰撞反馈: 身体感知环境
    - 能量卸载: 物理交互中的能量转移
    
    属于"高频可选"法则,需保持NumPy向量化以维持性能
    """
    name = "morphological_computation"
    
    def __init__(self, manifest):
        super().__init__(manifest)
        # 吸附检测预计算
        self._adhesion_range_sq = None
    
    def _get_adhesion_range_sq(self):
        """缓存吸附范围平方"""
        if self._adhesion_range_sq is None:
            r = self.manifest.adhesion_range
            self._adhesion_range_sq = r * r
        return self._adhesion_range_sq
    
    def apply(self, agents: List, world: Dict) -> None:
        """应用形态计算法则"""
        if not self.manifest.morphological_computation_enabled:
            return
        
        # 缓存参数
        adhesion_range_sq = self._get_adhesion_range_sq()
        carry_penalty = self.manifest.carry_speed_penalty
        discharge_thresh = self.manifest.discharge_threshold
        
        # 向量化: 批量检测吸附
        n_agents = len(agents)
        alive_agents = [a for a in agents if a.is_alive]
        
        for agent in alive_agents:
            # 检查是否携带食物
            if agent.food_carried > 0:
                # 速度惩罚
                if agent.port_motion > 0:
                    agent.port_motion *= carry_penalty
            
            # 能量卸载检测 (当能量过低时自动卸载食物)
            if agent.internal_energy < agent.internal_energy * discharge_thresh:
                if agent.food_carried > 0:
                    # 卸载食物到当前位置
                    self._drop_food_nearby(agent)
    
    def _drop_food_nearby(self, agent) -> None:
        """在Agent附近掉落食物"""
        # 减少携带量
        dropped = min(agent.food_carried, 1)
        agent.food_carried -= dropped
        
        # 留下环境印记 (供压痕系统使用)
        if hasattr(agent, 'last_drop_pos'):
            agent.last_drop_pos = (agent.x, agent.y)
    
    def compute_adhesion(self, agent_a, agent_b) -> float:
        """计算两Agent间的吸附力"""
        if not self.manifest.morphological_computation_enabled:
            return 0.0
        
        dx = agent_a.x - agent_b.x
        dy = agent_a.y - agent_b.y
        dist_sq = dx * dx + dy * dy
        
        if dist_sq < self._get_adhesion_range_sq():
            # 距离越近,吸附越强
            dist = dist_sq ** 0.5
            return 1.0 - (dist / self.manifest.adhesion_range)
        
        return 0.0
    
    def get_morphology_factor(self, agent) -> float:
        """获取Agent的形态因子 (用于适应度计算)"""
        factor = 1.0
        
        if agent.food_carried > 0:
            factor *= self.manifest.carry_speed_penalty
        
        return factor


class OntogeneticPhaseLaw(PhysicalLaw):
    """
    发育相变法则 (P0)
    
    引入"生命周期"概念:
    - 幼年期 (Juvenile): 低代谢但脆弱
    - 成熟期 (Mature): 正常代谢,具备繁殖能力
    - 衰老期 (Elderly): 高代谢,能力衰退
    
    为演化引入"时间维度",筛选更具远见的基因组
    """
    name = "ontogenetic_phase"
    
    def __init__(self, manifest):
        super().__init__(manifest)
        self.juvenile_duration = manifest.juvenile_duration
        self.juvenile_metabolic_rate = 0.25  # 幼年期25%代谢
    
    def reset(self) -> None:
        """重置发育状态"""
        pass
    
    def apply(self, agents: List, world: Dict) -> None:
        """应用发育相变"""
        if not self.manifest.ontogenetic_phase_enabled:
            return
        
        for agent in agents:
            if not agent.is_alive:
                continue
            
            # 初始化发育阶段(如果未设置)
            if not hasattr(agent, 'age'):
                agent.age = 0
                agent.phase = 'juvenile'
            
            # 年龄增长
            agent.age += 1
            
            # 阶段转换
            old_phase = agent.phase
            
            if agent.age < self.juvenile_duration:
                agent.phase = 'juvenile'
            elif agent.age < self.juvenile_duration * 3:
                agent.phase = 'mature'
            else:
                agent.phase = 'elderly'
            
            # 应用阶段特定效果
            if old_phase != agent.phase:
                self._on_phase_change(agent, old_phase, agent.phase)
            else:
                self._apply_phase_effects(agent)
    
    def _on_phase_change(self, agent, old_phase: str, new_phase: str) -> None:
        """阶段转换时的特殊效果"""
        if new_phase == 'mature':
            # 成熟: 恢复部分能量作为"成年礼"
            bonus = agent.internal_energy * 0.2
            agent.internal_energy += bonus
            
        elif new_phase == 'elderly':
            # 衰老: 一次性能量衰减作为"老年虚弱"
            decay = agent.internal_energy * 0.15
            agent.internal_energy -= decay
    
    def _apply_phase_effects(self, agent) -> None:
        """每帧应用阶段效果"""
        phase = agent.phase
        
        if phase == 'juvenile':
            # 幼年期: 低代谢 (受保护)
            agent.metabolic_multiplier = self.juvenile_metabolic_rate
            # 幼年期死亡率高 (自然选择)
            if hasattr(agent, 'death_prob'):
                agent.death_prob *= 1.5
                
        elif phase == 'mature':
            # 成熟期: 正常代谢
            agent.metabolic_multiplier = 1.0
            
        elif phase == 'elderly':
            # 衰老期: 高代谢,能力衰退
            agent.metabolic_multiplier = 1.5
            # 速度衰减
            if hasattr(agent, 'port_motion'):
                agent.port_motion *= 0.9
    
    def get_phase(self, agent) -> str:
        """获取Agent当前阶段"""
        return getattr(agent, 'phase', 'unknown')
    
    def get_age_in_phase(self, agent) -> int:
        """获取在当前阶段已度过的时间"""
        age = getattr(agent, 'age', 0)
        phase = getattr(agent, 'phase', 'juvenile')
        
        if phase == 'juvenile':
            return age
        elif phase == 'mature':
            return age - self.juvenile_duration
        else:
            return age - self.juvenile_duration * 3
    
    def is_prime(self, agent) -> str:
        """检查Agent是否在黄金时期 (成熟期中期)"""
        phase = getattr(agent, 'phase', '')
        age_in_phase = self.get_age_in_phase(agent)
        
        return phase == 'mature' and 10 < age_in_phase < self.juvenile_duration * 2


class StigmergicFrictionLaw(PhysicalLaw):
    """
    压痕系统法则 (P1)
    
    群感效应: 通过改变环境传递信息
    - 智能体移动时在地面留下印记
    - 其他智能体可以感知这些印记
    - 形成隐性的环境记忆
    
    属于"环境记忆"类法则
    """
    name = "stigmergic_friction"
    
    def __init__(self, manifest):
        super().__init__(manifest)
        # 压痕地图: (x, y) -> intensity
        self.impressions: Dict[tuple, float] = {}
        self.max_impressions = 1000  # 最多保留1000个印记
    
    def reset(self) -> None:
        """每代清空压痕"""
        self.impressions.clear()
    
    def apply(self, agents: List, world: Dict) -> None:
        """应用压痕系统"""
        if not self.manifest.stigmergic_friction_enabled:
            return
        
        # 缓存参数
        friction_coef = 0.1  # 可从manifest获取
        
        for agent in agents:
            if not agent.is_alive:
                continue
            
            # Agent移动时留下印记
            if agent.port_motion != 0:
                key = (int(agent.x), int(agent.y))
                
                # 增强现有印记或创建新印记
                if key in self.impressions:
                    self.impressions[key] = min(1.0, self.impressions[key] + friction_coef)
                else:
                    self.impressions[key] = friction_coef
                    
                    # 限制印记数量
                    if len(self.impressions) > self.max_impressions:
                        # 移除最弱的
                        weakest = min(self.impressions.items(), key=lambda x: x[1])
                        del self.impressions[weakest[0]]
        
        # 印记自然衰减
        decay_rate = 0.001
        for key in list(self.impressions.keys()):
            self.impressions[key] *= (1.0 - decay_rate)
            if self.impressions[key] < 0.01:
                del self.impressions[key]
    
    def get_impression_at(self, x: float, y: float) -> float:
        """获取某位置的印记强度"""
        key = (int(x), int(y))
        return self.impressions.get(key, 0.0)
    
    def get_nearby_impressions(self, x: float, y: float, radius: int = 3) -> float:
        """获取附近的印记总和"""
        cx, cy = int(x), int(y)
        total = 0.0
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                key = (cx + dx, cy + dy)
                total += self.impressions.get(key, 0.0)
        
        return total


class RedQueenLaw(PhysicalLaw):
    """
    红皇后竞争法则
    
    敌对Agent定期刷新,提供持续的选择压力
    防止生态系统坍缩为单一类型
    属于"实验性/核心"法则
    """
    name = "red_queen"
    
    def __init__(self, manifest):
        super().__init__(manifest)
        self.rivals: List = []
        self.rival_refresh_counter = 0
        self.generation = 0
    
    def reset(self) -> None:
        """重置敌对Agent"""
        self.rivals = []
        self.rival_refresh_counter = 0
    
    def set_generation(self, gen: int) -> None:
        """设置当前代数"""
        self.generation = gen
    
    def apply(self, agents: List, world: Dict) -> None:
        """应用红皇后竞争"""
        if not self.manifest.red_queen_enabled:
            return
        
        refresh_interval = self.manifest.rival_refresh_interval
        n_rivals = self.manifest.n_rivals
        
        # 检查是否需要刷新敌对
        if self.generation > 0 and self.generation % refresh_interval == 0:
            self._refresh_rivals(agents, n_rivals)
        
        # 更新敌对行为
        for rival in self.rivals:
            if rival.is_alive:
                self._update_rival_behavior(rival, agents)
    
    def _refresh_rivals(self, agents: List, n_rivals: int) -> None:
        """从精英个体刷新敌对"""
        if not agents:
            return
        
        # 按适应度排序,选择精英
        sorted_agents = sorted(agents, key=lambda a: a.fitness, reverse=True)
        elites = sorted_agents[:min(n_rivals * 2, len(sorted_agents))]
        
        if not elites:
            return
        
        self.rivals = []
        
        # 从精英复制敌对
        import copy
        for i in range(n_rivals):
            if i < len(elites):
                rival = copy.deepcopy(elites[i % len(elites)])
                rival.agent_id = 1000 + i  # 避免ID冲突
                rival.is_rival = True
                
                # 增强敌对能力
                self._buff_rival(rival)
                
                self.rivals.append(rival)
    
    def _buff_rival(self, rival) -> None:
        """增强敌对Agent能力"""
        # 增加初始能量
        rival.internal_energy *= 1.5
        
        # 增强感知范围
        if hasattr(rival, 'sensor_range'):
            rival.sensor_range *= 1.3
    
    def _update_rival_behavior(self, rival, agents: List) -> None:
        """更新敌对行为"""
        if not agents:
            return
        
        # 敌对策略: 寻找最近的Agent并竞争
        min_dist = float('inf')
        target = None
        
        for agent in agents:
            if agent.is_alive and not getattr(agent, 'is_rival', False):
                dist = ((rival.x - agent.x)**2 + (rival.y - agent.y)**2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    target = agent
        
        if target and min_dist < 10.0:
            # 竞争: 靠近并消耗目标能量
            dx = target.x - rival.x
            dy = target.y - rival.y
            dist = (dx**2 + dy**2) ** 0.5
            
            if dist > 0:
                rival.x += (dx / dist) * 0.5
                rival.y += (dy / dist) * 0.5
            
            # 能量抢夺
            if min_dist < 2.0:
                steal = min(target.internal_energy * 0.1, 5.0)
                target.internal_energy -= steal
                rival.internal_energy += steal
    
    def get_rivals(self) -> List:
        """获取当前敌对列表"""
        return self.rivals
    
    def get_competition_pressure(self) -> float:
        """计算当前竞争压力 (0-1)"""
        if not self.manifest.red_queen_enabled:
            return 0.0
        
        # 压力与敌对数量成正比
        return min(len(self.rivals) / max(self.manifest.n_rivals, 1), 1.0)


class ThermalSanctuaryLaw(PhysicalLaw):
    """
    热力学庇护所法则
    
    温度调节 + 食物热量辐射
    属于"可选/周期性法则"
    """
    name = "thermal_sanctuary"
    
    def __init__(self, manifest):
        super().__init__(manifest)
        self.current_temperature = manifest.summer_temperature
    
    def apply(self, agents: List, world: Dict) -> None:
        if not self.manifest.thermal_sanctuary_enabled:
            return
        
        # 获取当前温度
        season = world.get('season', 'summer')
        self.current_temperature = (
            self.manifest.summer_temperature 
            if season == 'summer' 
            else self.manifest.winter_temperature
        )
        
        # 缓存参数
        food_heat = self.manifest.food_heat_output
        heat_radius = self.manifest.food_heat_radius
        nest_insulation = self.manifest.nest_insulation
        
        for agent in agents:
            if not agent.is_alive:
                continue
            
            # 计算环境温度
            env_temp = self._calculate_local_temperature(agent, agents, food_heat, heat_radius)
            
            # 应用温度伤害/恢复
            if env_temp < 0:
                # 寒冷伤害
                damage = abs(env_temp) * 0.02 * (1.0 - nest_insulation)
                agent.internal_energy -= damage
            elif env_temp > 30:
                # 炎热消耗
                heat_damage = (env_temp - 30) * 0.01
                agent.internal_energy -= heat_damage
    
    def _calculate_local_temperature(self, agent, agents, food_heat: float, heat_radius: float) -> float:
        """计算Agent所在位置的局部温度"""
        # 基础温度
        temp = self.current_temperature
        
        # 食物热辐射
        for other in agents:
            if other.is_alive and other.food_carried > 0:
                dist = ((agent.x - other.x)**2 + (agent.y - other.y)**2) ** 0.5
                if dist < heat_radius:
                    # 距离越近,温度越高
                    heat_boost = food_heat * (1.0 - dist / heat_radius)
                    temp += heat_boost
        
        return temp


# ============================================================
# 便捷函数
# ============================================================

def create_default_manifest() -> PhysicsManifest:
    """创建默认配置"""
    return PhysicsManifest()


def load_manifest_from_config(config_path: str = "physics_config.json") -> PhysicsManifest:
    """从配置文件加载"""
    import os
    # 尝试多种路径
    paths = [
        config_path,
        os.path.join(os.path.dirname(__file__), "..", "..", config_path),
    ]
    
    for path in paths:
        if os.path.exists(path):
            return PhysicsManifest.from_json(path)
    
    return create_default_manifest()


if __name__ == "__main__":
    # 测试机制注册系统
    manifest = PhysicsManifest(
        metabolic_alpha=0.003,
        seasonal_cycle=True,
        fatigue_system_enabled=True,
        energy_decay_k=0.00005,
    )
    
    registry = MechanismRegistry(manifest)
    
    # 注册法则
    registry.register("metabolism", MetabolismLaw)
    registry.register("seasonal", SeasonalCycleLaw)
    registry.register("fatigue", FatigueSystemLaw)
    registry.register("interference", PortInterferenceLaw)
    
    print("已注册的物理法则:")
    for law_info in registry.list_laws():
        print(f"  {law_info['name']}: {law_info['enabled']} ({law_info['class']})")
    
    # 模拟应用
    print("\n模拟step()调用:")
    registry.apply_all(agents=[], world={})