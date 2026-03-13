# ============================================================
# eoe_mvp/config/agent_mechanisms.py
# ============================================================
"""
机制配置系统
============
从 YAML 文件加载配置,使用 Bool 控制机制启用/禁用

Usage:
    from config.agent_mechanisms import Mechanisms, EnvMechanisms, load_config
    
    # 加载预设
    load_config("full")
    
    # 检查 Agent 机制
    if Mechanisms.SENSOR_EPF:
        # ...
        
    # 检查环境机制
    if EnvMechanisms.EPF:
        # ...
"""

import os
import yaml
from typing import Dict, Any, Optional

# 配置目录
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_FILE = os.path.join(CONFIG_DIR, "mechanisms.yaml")

# 缓存配置数据
_config_cache: Optional[Dict] = None


def _load_yaml() -> Dict:
    """加载 YAML 配置"""
    global _config_cache
    if _config_cache is None:
        with open(YAML_FILE, 'r', encoding='utf-8') as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def _get_path(key_path: str) -> tuple:
    """解析路径如 'sensor.epf' -> ('sensor', 'epf')"""
    parts = key_path.lower().split('.')
    return tuple(parts)


def get_value(key_path: str, default: Any = None) -> Any:
    """从配置获取值"""
    data = _load_yaml()
    parts = _get_path(key_path)
    
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


# ============================================================
# Agent 机制开关类
# ============================================================

class Mechanisms:
    """
    Agent 机制开关
    ==============
    """
    
    # ---------- 感知系统 (Sensors) ----------
    SENSOR_EPF: bool = True
    SENSOR_KIF: bool = True
    SENSOR_ISF: bool = True
    SENSOR_ENERGY: bool = True
    
    # ---------- 运动系统 (Actuators) ----------
    ACTUATOR_THRUST: bool = True
    ACTUATOR_PERMEABILITY: bool = True
    ACTUATOR_DEFENSE: bool = True
    
    # ---------- 信号系统 (Communication) ----------
    SIGNAL_DEPOSIT: bool = True
    SIGNAL_RECEIVE: bool = True
    
    # ---------- 能量系统 (Thermodynamics) ----------
    ENERGY_EXTRACTION: bool = True
    ENERGY_DEPLETABLE: bool = True
    ENERGY_INFINITE: bool = False
    ENERGY_METABOLIC: bool = True
    ENERGY_DEATH: bool = True
    
    # ---------- 进化系统 (Evolution) ----------
    EVOLUTION_SELECTION: bool = True
    EVOLUTION_MUTATION: bool = True
    EVOLUTION_CROSSOVER: bool = True
    EVOLUTION_ISF_DECAY: bool = True
    EVOLUTION_ENABLED: bool = True
    
    _key_map = {
        'SENSOR_EPF': 'sensor.epf',
        'SENSOR_KIF': 'sensor.kif',
        'SENSOR_ISF': 'sensor.isf',
        'SENSOR_ENERGY': 'sensor.energy',
        'ACTUATOR_THRUST': 'actuator.thrust',
        'ACTUATOR_PERMEABILITY': 'actuator.permeability',
        'ACTUATOR_DEFENSE': 'actuator.defense',
        'SIGNAL_DEPOSIT': 'signal.deposit',
        'SIGNAL_RECEIVE': 'signal.receive',
        'ENERGY_EXTRACTION': 'energy.extraction',
        'ENERGY_DEPLETABLE': 'energy.depletable',
        'ENERGY_INFINITE': 'energy.infinite',
        'ENERGY_METABOLIC': 'energy.metabolic',
        'ENERGY_DEATH': 'energy.death',
        'EVOLUTION_SELECTION': 'evolution.selection',
        'EVOLUTION_MUTATION': 'evolution.mutation',
        'EVOLUTION_CROSSOVER': 'evolution.crossover',
        'EVOLUTION_ISF_DECAY': 'evolution.isf_decay',
        'EVOLUTION_ENABLED': 'evolution.enabled',
    }
    
    @classmethod
    def load_from_yaml(cls, preset: Optional[str] = None):
        """从 YAML 加载配置"""
        data = _load_yaml()
        
        # 确定使用哪个配置
        if preset and preset in data.get('presets', {}):
            config = data['presets'][preset]
        else:
            config = {k: v for k, v in data.items() if k != 'presets'}
        
        # 从配置字典中获取值 (而非再次调用get_value)
        for attr, path in cls._key_map.items():
            parts = path.split('.')
            value = config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = False
                    break
            setattr(cls, attr, bool(value))
    
    @classmethod
    def to_dict(cls) -> Dict[str, bool]:
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and isinstance(v, bool)
        }
    
    @classmethod
    def enabled_count(cls) -> int:
        return sum(v for v in cls.to_dict().values())
    
    @classmethod
    def print_status(cls):
        print(f"\n{'='*50}")
        print(f"Agent 机制状态 (启用: {cls.enabled_count()})")
        print('='*50)
        
        by_category = {
            "感知": ["SENSOR_EPF", "SENSOR_KIF", "SENSOR_ISF", "SENSOR_ENERGY"],
            "运动": ["ACTUATOR_THRUST", "ACTUATOR_PERMEABILITY", "ACTUATOR_DEFENSE"],
            "信号": ["SIGNAL_DEPOSIT", "SIGNAL_RECEIVE"],
            "能量": ["ENERGY_EXTRACTION", "ENERGY_DEPLETABLE", "ENERGY_INFINITE", "ENERGY_METABOLIC", "ENERGY_DEATH"],
            "进化": ["EVOLUTION_SELECTION", "EVOLUTION_MUTATION", "EVOLUTION_CROSSOVER", "EVOLUTION_ISF_DECAY", "EVOLUTION_ENABLED"],
        }
        
        for cat, keys in by_category.items():
            print(f"\n[{cat}]")
            for k in keys:
                status = "✅" if getattr(cls, k) else "❌"
                print(f"  {status} {k.lower()}")


# ============================================================
# 环境机制开关类
# ============================================================

class EnvMechanisms:
    """
    Environment 机制开关
    ====================
    """
    
    # ---------- 环境场 (Environment) ----------
    EPF: bool = True      # 能量场 (Energy Potential Field)
    KIF: bool = True      # 阻抗场 (Kinetic Impedance Field)
    ISF: bool = True      # 信号场 (Intelligence/Info Field)
    WORLD_BOUNDS: bool = True  # 世界边界
    SOURCE_RESPAWN: bool = True  # 能量源重生
    DIFFUSION: bool = True      # 场扩散 (热传导)
    GRADIENT: bool = True       # 场梯度计算
    
    # ---------- 物理系统 (Physics) ----------
    COLLISION: bool = True       # 碰撞检测
    BOUNDARY_WRAP: bool = False  # 边界循环 (穿越边缘)
    VELOCITY_DECAY: bool = True  # 速度衰减
    FRICTION: bool = True        # 摩擦力
    
    _key_map = {
        'EPF': 'environment.epf',
        'KIF': 'environment.kif',
        'ISF': 'environment.isf',
        'WORLD_BOUNDS': 'environment.world_bounds',
        'SOURCE_RESPAWN': 'environment.source_respawn',
        'DIFFUSION': 'environment.diffusion',
        'GRADIENT': 'environment.gradient',
        'COLLISION': 'physics.collision',
        'BOUNDARY_WRAP': 'physics.boundary_wrap',
        'VELOCITY_DECAY': 'physics.velocity_decay',
        'FRICTION': 'physics.friction',
    }
    
    @classmethod
    def load_from_yaml(cls, preset: Optional[str] = None):
        """从 YAML 加载配置"""
        data = _load_yaml()
        
        # 确定使用哪个配置
        if preset and preset in data.get('presets', {}):
            config = data['presets'][preset]
        else:
            config = {k: v for k, v in data.items() if k != 'presets'}
        
        # 从配置字典中获取值 (而非再次调用get_value)
        for attr, path in cls._key_map.items():
            parts = path.split('.')
            value = config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = False
                    break
            setattr(cls, attr, bool(value))
    
    @classmethod
    def to_dict(cls) -> Dict[str, bool]:
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and isinstance(v, bool)
        }
    
    @classmethod
    def enabled_count(cls) -> int:
        return sum(v for v in cls.to_dict().values())
    
    @classmethod
    def print_status(cls):
        print(f"\n{'='*50}")
        print(f"环境机制状态 (启用: {cls.enabled_count()})")
        print('='*50)
        
        print("\n[环境场]")
        for k in ['EPF', 'KIF', 'ISF', 'WORLD_BOUNDS', 'SOURCE_RESPAWN', 'DIFFUSION', 'GRADIENT']:
            status = "✅" if getattr(cls, k) else "❌"
            print(f"  {status} {k.lower()}")
        
        print("\n[物理]")
        for k in ['COLLISION', 'BOUNDARY_WRAP', 'VELOCITY_DECAY', 'FRICTION']:
            status = "✅" if getattr(cls, k) else "❌"
            print(f"  {status} {k.lower()}")


# ============================================================
# 统一加载函数
# ============================================================

def load_config(preset: str):
    """加载预设配置 (同时加载 Agent 和环境机制)"""
    Mechanisms.load_from_yaml(preset)
    EnvMechanisms.load_from_yaml(preset)


def list_presets() -> list:
    """列出所有预设"""
    data = _load_yaml()
    return list(data.get('presets', {}).keys())


def is_enabled(key: str) -> bool:
    """检查机制是否启用"""
    if hasattr(Mechanisms, key):
        return getattr(Mechanisms, key, False)
    if hasattr(EnvMechanisms, key):
        return getattr(EnvMechanisms, key, False)
    return False


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("机制配置系统")
    print(f"配置文件: {YAML_FILE}")
    print(f"可用预设: {list_presets()}")
    
    # 测试加载各预设
    for preset in list_presets():
        load_config(preset)
        print(f"\n[{preset}] Agent: {Mechanisms.enabled_count()} | 环境: {EnvMechanisms.enabled_count()}")
    
    # 加载默认配置并打印
    load_config("full")
    Mechanisms.print_status()
    EnvMechanisms.print_status()
    
    print("\n" + "="*50)
    print("使用示例:")
    print("="*50)
    print('  from config.agent_mechanisms import Mechanisms, EnvMechanisms, load_config')
    print('  ')
    print('  # 加载预设')
    print('  load_config("full")')
    print('  ')
    print('  # 检查 Agent 机制')
    print('  if Mechanisms.SENSOR_EPF:')
    print('      print("能量感知启用")')
    print('  ')
    print('  # 检查环境机制')
    print('  if EnvMechanisms.EPF:')
    print('      print("能量场启用")')
    print('  ')
    print('  if EnvMechanisms.BOUNDARY_WRAP:')
    print('      print("穿越边界模式")')