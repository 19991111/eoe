# 机制配置与 Manifest 整合方案

## 现状

### 1. manifest.py 机制
- `PhysicsManifest`: 数据类,存储物理参数
- `MechanismRegistry`: 注册器,管理 PhysicalLaw 类
- 基于类名自动匹配 manifest 中的开关字段

### 2. config/ 机制 (新开发)
- `mechanisms.yaml`: 配置文件 (Bool 开关)
- `agent_mechanisms.py`: 加载器,提供 Mechanisms/EnvMechanisms 类

## 整合方案

### 方案 A: 在 PhysicsManifest 中集成 YAML 配置

```python
# config/agent_mechanisms.py 新增
from core.eoe.manifest import PhysicsManifest

class ConfiguredManifest(PhysicsManifest):
    """扩展 PhysicsManifest 支持 YAML 配置"""
    
    def __init__(self, preset: str = "full", **kwargs):
        super().__init__(**kwargs)
        
        # 加载 YAML 配置
        load_config(preset)
        
        # 将 YAML 配置映射到 manifest 字段
        self._map_yaml_to_manifest()
    
    def _map_yaml_to_manifest(self):
        """将 YAML 配置映射为 manifest 参数"""
        
        # 感知系统
        if not Mechanisms.SENSOR_EPF:
            self.sensor_range = 0  # 禁用能量感知
        
        # 能量系统
        if not Mechanisms.ENERGY_EXTRACTION:
            self.food_energy = 0  # 禁用能量获取
        
        # 进化系统
        if not Mechanisms.EVOLUTION_ENABLED:
            self.energy_decay_k = 0  # 禁用代谢
        
        # 环境系统
        if not EnvMechanisms.ISF:
            self.stigmergic_friction_enabled = False
        
        # ... 更多映射
```

### 方案 B: 在 MechanismRegistry 中集成 YAML 检查

```python
class IntegratedRegistry(MechanismRegistry):
    """整合 YAML 配置的注册器"""
    
    def _is_law_enabled(self, law_class: Type[PhysicalLaw]) -> bool:
        # 1. 先检查 manifest
        manifest_enabled = super()._is_law_enabled(law_class)
        
        # 2. 再检查 YAML 配置
        yaml_key = self._get_yaml_key(law_class)
        if yaml_key and hasattr(Mechanisms, yaml_key):
            yaml_enabled = getattr(Mechanisms, yaml_key, True)
            return manifest_enabled and yaml_enabled
        
        return manifest_enabled
    
    def _get_yaml_key(self, law_class) -> str:
        """映射 law 类名到 YAML 机制键"""
        mapping = {
            'MetabolismLaw': 'ENERGY_METABOLIC',
            'ExtractionLaw': 'ENERGY_EXTRACTION',
            'StigmergyLaw': 'SIGNAL_DEPOSIT',
            'MovementLaw': 'ACTUATOR_THRUST',
            # ...
        }
        return mapping.get(law_class.__name__, '')
```

## 推荐: 方案 C (分层配置)

```
┌─────────────────────────────────────────────────┐
│           配置文件 (mechanisms.yaml)            │
│  - 高层开关: sensor.*, actuator.*, energy.*     │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│        agent_mechanisms.py (加载层)             │
│  - Mechanisms / EnvMechanisms 类                │
│  - load_config(), is_enabled()                  │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         manifest.py (参数层)                    │
│  - PhysicsManifest: 具体数值参数                │
│  - MechanismRegistry: 法则注册                  │
└─────────────────────────────────────────────────┘
```

**当前问题:**
- mechanisms.yaml 定义了 28 个 Bool 开关
- manifest.py 定义了 50+ 个数值/布尔参数
- 两套系统独立运行,需要整合

**待审批后实施整合**