#!/usr/bin/env python3
"""
v16.0 Phase 1 测试: MatterGrid 基础设施验证

测试内容:
1. CPU Environment matter_grid 初始化
2. is_solid / add_matter / remove_matter 方法
3. 能量存储机制
4. GPU Environment matter_grid 初始化 (如果可用)
"""

import sys
import numpy as np

def test_cpu_matter_grid():
    """测试 CPU 环境的 MatterGrid"""
    print("\n" + "="*60)
    print("测试 1: CPU Environment MatterGrid")
    print("="*60)
    
    from core.eoe.environment import Environment
    
    # 创建启用 matter_grid 的环境
    env = Environment(
        width=50,
        height=50,
        matter_grid_enabled=True,
        matter_resolution=1.0,
        # 关闭其他场以简化测试
        energy_field_enabled=False,
        impedance_field_enabled=False,
        stigmergy_field_enabled=False,
        stress_field_enabled=False
    )
    
    print(f"✅ Environment created: {env.matter_grid_width}x{env.matter_grid_height}")
    print(f"   matter_grid shape: {env.matter_grid.shape}")
    print(f"   matter_energy shape: {env.matter_energy.shape}")
    
    # 测试 is_solid (空地点)
    result = env.is_solid(5.0, 5.0)
    assert result == False, "Empty point should not be solid"
    print(f"✅ is_solid(5,5) on empty: {result}")
    
    # 测试 add_matter
    success = env.add_matter(5.0, 5.0, stored_energy=15.0)
    assert success == True, "Should successfully add matter"
    print(f"✅ add_matter(5,5) with energy=15.0: {success}")
    
    # 测试 is_solid (刚添加的点)
    result = env.is_solid(5.0, 5.0)
    assert result == True, "Point with matter should be solid"
    print(f"✅ is_solid(5,5) after add: {result}")
    
    # 测试能量存储
    energy = env.get_matter_energy(5.0, 5.0)
    assert energy == 15.0, f"Stored energy should be 15.0, got {energy}"
    print(f"✅ get_matter_energy(5,5): {energy}")
    
    # 测试 add_matter 到已有位置（应该失败）
    success = env.add_matter(5.0, 5.0, stored_energy=10.0)
    assert success == False, "Should fail to add matter on existing matter"
    print(f"✅ add_matter(5,5) on existing: {success} (expected False)")
    
    # 测试 remove_matter
    success = env.remove_matter(5.0, 5.0)
    assert success == True, "Should successfully remove matter"
    print(f"✅ remove_matter(5,5): {success}")
    
    # 测试 remove 后的状态
    result = env.is_solid(5.0, 5.0)
    assert result == False, "Point after remove should not be solid"
    print(f"✅ is_solid(5,5) after remove: {result}")
    
    # 测试边界环绕 (toroidal)
    env2 = Environment(width=20, height=20, matter_grid_enabled=True)
    env2.add_matter(19.0, 19.0)  # 边界
    result = env2.is_solid(0.0, 0.0)  # 应该环绕到 19,19
    print(f"✅ Toroidal wrap test: is_solid(0,0) near edge(19,19): {result}")
    
    print("\n✅ 所有 CPU MatterGrid 测试通过!")
    return True


def test_gpu_matter_grid():
    """测试 GPU 环境的 MatterGrid"""
    print("\n" + "="*60)
    print("测试 2: GPU Environment MatterGrid")
    print("="*60)
    
    try:
        import torch
        from core.eoe.environment_gpu import EnvironmentGPU
        
        if not torch.cuda.is_available():
            print("⚠️  CUDA 不可用，跳过 GPU 测试")
            return True
        
        # 创建启用 matter_grid 的 GPU 环境
        env = EnvironmentGPU(
            width=50,
            height=50,
            matter_grid_enabled=True,
            matter_resolution=1.0,
            energy_field_enabled=False,
            impedance_field_enabled=False,
            stigmergy_field_enabled=False,
            danger_field_enabled=False
        )
        
        print(f"✅ EnvironmentGPU created: {env.matter_grid_width}x{env.matter_grid_height}")
        print(f"   matter_grid shape: {env.matter_grid.shape}")
        print(f"   matter_energy shape: {env.matter_energy.shape}")
        
        # 测试 is_solid
        result = env.is_solid(5.0, 5.0)
        assert result == False, "Empty point should not be solid"
        print(f"✅ is_solid(5,5) on empty: {result}")
        
        # 测试 add_matter
        success = env.add_matter(5.0, 5.0, stored_energy=15.0)
        assert success == True
        print(f"✅ add_matter(5,5) with energy=15.0: {success}")
        
        # 测试 is_solid
        result = env.is_solid(5.0, 5.0)
        assert result == True
        print(f"✅ is_solid(5,5) after add: {result}")
        
        # 测试能量存储
        energy = env.get_matter_energy(5.0, 5.0)
        assert energy == 15.0
        print(f"✅ get_matter_energy(5,5): {energy}")
        
        # 测试 remove_matter
        success = env.remove_matter(5.0, 5.0)
        assert success == True
        print(f"✅ remove_matter(5,5): {success}")
        
        result = env.is_solid(5.0, 5.0)
        assert result == False
        print(f"✅ is_solid(5,5) after remove: {result}")
        
        print("\n✅ 所有 GPU MatterGrid 测试通过!")
        return True
        
    except ImportError as e:
        print(f"⚠️  导入错误: {e}")
        return False
    except Exception as e:
        print(f"⚠️  GPU 测试失败: {e}")
        return False


def test_u_wall():
    """测试 U 型墙的碰撞"""
    print("\n" + "="*60)
    print("测试 3: U型墙碰撞检测")
    print("="*60)
    
    from core.eoe.environment import Environment
    
    env = Environment(
        width=50,
        height=50,
        matter_grid_enabled=True,
        matter_resolution=1.0,
        energy_field_enabled=False,
        impedance_field_enabled=False,
        stigmergy_field_enabled=False
    )
    
    # 构建 U 型墙
    # 墙的形状:
    # #####
    # #   #
    # #   #
    # #####
    
    # 垂直段
    for y in range(20, 30):
        env.add_matter(20.0, float(y))
        env.add_matter(21.0, float(y))
    
    # 底部
    for x in range(20, 31):
        env.add_matter(float(x), 29.0)
    
    # 垂直段 (右侧)
    for y in range(20, 30):
        env.add_matter(30.0, float(y))
    
    print("✅ U型墙已构建")
    
    # 测试墙内
    assert env.is_solid(20.0, 25.0) == True
    assert env.is_solid(25.0, 29.0) == True
    assert env.is_solid(30.0, 25.0) == True
    print("✅ 墙内位置检测正确")
    
    # 测试墙外
    assert env.is_solid(25.0, 25.0) == False  # U型内部
    assert env.is_solid(25.0, 15.0) == False  # 墙上
    assert env.is_solid(35.0, 25.0) == False  # 墙右
    print("✅ 墙外位置检测正确")
    
    # 统计墙的格子数
    wall_count = np.sum(env.matter_grid)
    print(f"   U型墙格子数: {wall_count}")
    
    print("\n✅ U型墙碰撞检测测试通过!")
    return True


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("v16.0 MatterGrid 基础设施测试")
    print("="*60)
    
    all_passed = True
    
    # 测试 1: CPU
    if not test_cpu_matter_grid():
        all_passed = False
    
    # 测试 2: GPU
    if not test_gpu_matter_grid():
        all_passed = False
    
    # 测试 3: U型墙
    if not test_u_wall():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有测试通过! MatterGrid 基础设施就绪.")
    else:
        print("❌ 部分测试失败!")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    main()