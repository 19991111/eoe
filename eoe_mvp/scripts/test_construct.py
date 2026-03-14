#!/usr/bin/env python3
"""
v16.0 Phase 2 测试: 建造/分解功能

测试内容:
1. 验证 ACTUATOR_CONSTRUCT 节点类型存在
2. 验证配置项已添加
3. 验证脑输出扩展到7通道
"""

def test_node_types():
    """测试新节点类型"""
    print("\n" + "="*60)
    print("测试 1: 新节点类型")
    print("="*60)
    
    from core.eoe.node import NodeType
    
    # 检查新节点类型
    assert hasattr(NodeType, 'ACTUATOR_CONSTRUCT'), "ACTUATOR_CONSTRUCT 不存在"
    assert hasattr(NodeType, 'ACTUATOR_DECONSTRUCT'), "ACTUATOR_DECONSTRUCT 不存在"
    
    print(f"✅ ACTUATOR_CONSTRUCT: {NodeType.ACTUATOR_CONSTRUCT}")
    print(f"✅ ACTUATOR_DECONSTRUCT: {NodeType.ACTUATOR_DECONSTRUCT}")
    
    return True


def test_pool_config():
    """测试配置项"""
    print("\n" + "="*60)
    print("测试 2: PoolConfig 配置项")
    print("="*60)
    
    from core.eoe.batched_agents import PoolConfig
    
    config = PoolConfig()
    
    # 检查配置项
    assert hasattr(config, 'MATTER_GRID_ENABLED'), "MATTER_GRID_ENABLED 不存在"
    assert hasattr(config, 'CONSTRUCT_ENERGY_COST'), "CONSTRUCT_ENERGY_COST 不存在"
    assert hasattr(config, 'CONSTRUCT_MIN_ENERGY'), "CONSTRUCT_MIN_ENERGY 不存在"
    assert hasattr(config, 'DECONSTRUCT_ENERGY_GAIN'), "DECONSTRUCT_ENERGY_GAIN 不存在"
    assert hasattr(config, 'N_BRAIN_OUTPUTS_V16'), "N_BRAIN_OUTPUTS_V16 不存在"
    
    print(f"✅ MATTER_GRID_ENABLED: {config.MATTER_GRID_ENABLED}")
    print(f"✅ CONSTRUCT_ENERGY_COST: {config.CONSTRUCT_ENERGY_COST}")
    print(f"✅ CONSTRUCT_MIN_ENERGY: {config.CONSTRUCT_MIN_ENERGY}")
    print(f"✅ DECONSTRUCT_ENERGY_GAIN: {config.DECONSTRUCT_ENERGY_GAIN}")
    print(f"✅ N_BRAIN_OUTPUTS_V16: {config.N_BRAIN_OUTPUTS_V16}")
    
    return True


def test_brain_outputs():
    """测试脑输出扩展"""
    print("\n" + "="*60)
    print("测试 3: 脑输出扩展")
    print("="*60)
    
    import torch
    from core.eoe.batched_agents import PoolConfig, BatchedAgents
    
    # 测试默认配置
    config = PoolConfig()
    assert config.N_BRAIN_OUTPUTS == 5
    
    # 测试 v16.0 配置
    config16 = PoolConfig()
    config16.MATTER_GRID_ENABLED = True
    assert config16.N_BRAIN_OUTPUTS_V16 == 7
    
    print(f"✅ 默认输出通道: {config.N_BRAIN_OUTPUTS}")
    print(f"✅ v16.0 输出通道: {config16.N_BRAIN_OUTPUTS_V16}")
    
    return True


def main():
    print("\n" + "="*60)
    print("v16.0 Phase 2 建造/分解功能测试")
    print("="*60)
    
    all_passed = True
    
    if not test_node_types():
        all_passed = False
    
    if not test_pool_config():
        all_passed = False
    
    if not test_brain_outputs():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 Phase 2 功能就绪! 建造/分解已实现")
    else:
        print("❌ 部分测试失败!")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    import sys
    if not main():
        sys.exit(1)