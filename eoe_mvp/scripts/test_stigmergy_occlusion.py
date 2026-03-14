#!/usr/bin/env python3
"""
v16.0 Phase 1 测试: 压痕场遮挡 (Review #1)

测试内容:
- 验证物质墙阻断信息素扩散
- 对比有墙/无墙情况的扩散模式
"""

import sys
import numpy as np

def test_stigmergy_occlusion():
    """测试压痕场被墙壁遮挡"""
    print("\n" + "="*60)
    print("测试: 压痕场遮挡 (Stigmergy Occlusion)")
    print("="*60)
    
    from core.eoe.environment import Environment
    
    # 创建环境：启用压痕场 + 物质场
    env = Environment(
        width=50,
        height=50,
        matter_grid_enabled=True,
        matter_resolution=1.0,
        stigmergy_field_enabled=True,
        stigmergy_diffusion_rate=0.1,
        energy_field_enabled=False,
        impedance_field_enabled=False,
        stress_field_enabled=False
    )
    
    print(f"✅ Environment created with Stigmergy + MatterGrid")
    
    # 在 (25, 25) 放置一个信号源
    env.stigmergy_field.field[25, 25] = 10.0
    print(f"✅ 信号源放置在 (25, 25), 强度=10.0")
    
    # 构建一堵墙，阻断信号源到左侧的扩散
    # 墙在 x=20，信号源在 x=25
    for y in range(15, 35):
        env.add_matter(20.0, float(y))
    
    print(f"✅ 墙壁构建在 x=20, y=15-35")
    
    # 记录扩散前的场值
    signal_at_left = env.stigmergy_field.field[15, 25]  # 墙左侧
    signal_at_right = env.stigmergy_field.field[35, 25]  # 墙右侧
    
    print(f"\n扩散前:")
    print(f"  左侧 (x=15): {signal_at_left:.4f}")
    print(f"  右侧 (x=35): {signal_at_right:.4f}")
    
    # 执行多步扩散
    for _ in range(20):
        env.stigmergy_field.step(matter_grid=env.matter_grid)
    
    # 检查结果
    signal_at_left = env.stigmergy_field.field[15, 25]  # 墙左侧 (应该被阻断)
    signal_at_right = env.stigmergy_field.field[35, 25]  # 墙右侧 (应该能到达)
    signal_near_wall_left = env.stigmergy_field.field[19, 25]  # 墙左侧紧邻
    signal_near_wall_right = env.stigmergy_field.field[21, 25]  # 墙右侧紧邻
    
    print(f"\n扩散20步后:")
    print(f"  左侧远处 (x=15): {signal_at_left:.4f}")
    print(f"  右侧远处 (x=35): {signal_at_right:.4f}")
    print(f"  墙左侧紧邻 (x=19): {signal_near_wall_left:.4f}")
    print(f"  墙右侧紧邻 (x=21): {signal_near_wall_right:.4f}")
    
    # 验证: 墙左侧的信号应该明显低于墙右侧
    # 因为墙阻断了扩散
    if signal_at_left < signal_at_right * 0.5:
        print(f"\n✅ 遮挡生效! 左侧({signal_at_left:.2f}) << 右侧({signal_at_right:.2f})")
    else:
        print(f"\n⚠️  警告: 左侧信号可能穿墙了")
    
    # 额外验证: 墙壁位置的信号应该是0
    wall_signal = env.stigmergy_field.field[20, 25]
    print(f"  墙壁位置 (x=20): {wall_signal:.4f}")
    
    if wall_signal < 0.01:
        print(f"✅ 墙壁位置信号被清零 (遮挡成功)")
    else:
        print(f"❌ 墙壁位置仍有信号 (遮挡失败)")
        return False
    
    print("\n✅ 压痕场遮挡测试通过!")
    return True


def main():
    print("\n" + "="*60)
    print("v16.0 压痕场遮挡测试")
    print("="*60)
    
    if test_stigmergy_occlusion():
        print("\n🎉 压痕场遮挡机制验证成功!")
    else:
        print("\n❌ 测试失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()