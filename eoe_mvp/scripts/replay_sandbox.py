#!/usr/bin/env python3
"""
🧪 沙盒回放 (Replay Sandbox)
==============================
用于观察和测试脑结构的行为逻辑

功能:
1. 读取JSON格式的脑结构
2. 在无突变、无死亡的环境下运行
3. 实时观察节点激活和决策过程

作者: 104助手
日期: 2026-03-10
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import json
from eoe.genome import OperatorGenome
from eoe.node import NodeType


class ReplaySandbox:
    """沙盒回放环境"""
    
    def __init__(self, genome_json_path: str):
        # 加载脑结构
        self.genome = OperatorGenome.load_json(genome_json_path)
        
        # Agent状态
        self.x = 50.0
        self.y = 50.0
        self.angle = 0.0
        self.energy = 100.0
        
        # 环境状态
        self.food_positions = [(50, 30), (70, 50), (50, 70), (30, 50)]
        self.nest_pos = (50, 50)
        
        # 记录
        self.history = []
        
        print(f"🧠 已加载脑结构: {len(self.genome.nodes)}节点, {len(self.genome.edges)}边")
        
    def get_sensor_inputs(self) -> np.ndarray:
        """获取传感器输入"""
        # 传感器1: 最近食物的方向和距离
        min_dist = 999
        nearest_food_angle = 0
        for fx, fy in self.food_positions:
            dist = np.sqrt((self.x - fx)**2 + (self.y - fy)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_food_angle = np.arctan2(fy - self.y, fx - self.x)
        
        # 传感器2: 巢穴方向
        nest_angle = np.arctan2(self.nest_pos[1] - self.y, self.nest_pos[0] - self.x)
        nest_dist = np.sqrt((self.x - self.nest_pos[0])**2 + (self.y - self.nest_pos[1])**2)
        
        # 输入: [食物方向, 食物距离, 巢穴方向, 巢穴距离]
        inputs = np.array([
            np.sin(nearest_food_angle),
            np.cos(nearest_food_angle),
            min_dist / 100.0,
            np.sin(nest_angle),
            np.cos(nest_angle),
            nest_dist / 100.0,
            self.energy / 100.0  # 能量水平
        ])
        
        return inputs
    
    def step(self, action: np.ndarray) -> dict:
        """执行一步"""
        # 动作: [速度, 转向]
        speed = action[0] if len(action) > 0 else 0
        turn = action[1] if len(action) > 1 else 0
        
        # 更新位置
        self.angle += turn * 0.3
        self.x += np.cos(self.angle) * speed * 2
        self.y += np.sin(self.angle) * speed * 2
        
        # 边界限制
        self.x = max(0, min(100, self.x))
        self.y = max(0, min(100, self.y))
        
        # 能量消耗
        self.energy -= 0.1 + speed * 0.1
        
        # 检查食物
        food_eaten = 0
        for i, (fx, fy) in enumerate(list(self.food_positions)):
            if np.sqrt((self.x - fx)**2 + (self.y - fy)**2) < 8:
                self.food_positions.pop(i)
                self.energy = min(200, self.energy + 30)
                food_eaten = 1
                break
        
        return {
            'x': self.x,
            'y': self.y,
            'energy': self.energy,
            'food_eaten': food_eaten
        }
    
    def run(self, steps: int = 100, verbose: bool = True):
        """运行回放"""
        if verbose:
            print(f"\n{'='*60}")
            print("🧪 沙盒回放开始")
            print(f"{'='*60}")
        
        for step in range(steps):
            # 获取输入
            inputs = self.get_sensor_inputs()
            
            # 前向传播
            outputs = self.genome.forward(inputs)
            
            # 执行动作
            result = self.step(outputs)
            
            # 记录
            self.history.append({
                'step': step,
                'inputs': inputs,
                'outputs': outputs,
                **result
            })
            
            if verbose and step % 10 == 0:
                # 节点激活统计
                activations = {n: node.activation for n, node in self.genome.nodes.items()}
                active_nodes = sum(1 for a in activations.values() if abs(a) > 0.1)
                
                print(f"Step {step:3d} | Pos=({result['x']:.0f},{result['y']:.0f}) | "
                      f"能量={result['energy']:.0f} | 活跃节点={active_nodes}/33 | "
                      f"输出=[{outputs[0]:.2f},{outputs[1]:.2f}]")
        
        if verbose:
            print(f"\n{'='*60}")
            print("🏁 回放结束")
            print(f"{'='*60}")
            
        return self.history
    
    def analyze(self):
        """分析行为模式"""
        print("\n📊 行为分析")
        print("-" * 40)
        
        # 移动模式
        total_movement = 0
        for h in self.history:
            if h['step'] > 0:
                prev = self.history[h['step']-1]
                total_movement += np.sqrt((h['x']-prev['x'])**2 + (h['y']-prev['y'])**2)
        
        print(f"总移动距离: {total_movement:.1f}")
        print(f"平均速度: {total_movement/len(self.history):.2f}")
        
        # 节点激活分析
        print("\n🧠 节点激活分析")
        
        # 按类型统计激活
        node_activations = {nt: [] for nt in NodeType}
        for h in self.history:
            for node_id, node in self.genome.nodes.items():
                node_activations[node.node_type].append(abs(node.activation))
        
        for ntype, acts in node_activations.items():
            if acts:
                avg = sum(acts) / len(acts)
                max_act = max(acts)
                if avg > 0.01:
                    print(f"  {ntype.name:15s}: avg={avg:.3f}, max={max_act:.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='沙盒回放')
    parser.add_argument('--brain', type=str, 
                        default='../champions/best_v086_genius.json',
                        help='脑结构JSON文件路径')
    parser.add_argument('--steps', type=int, default=100,
                        help='运行步数')
    parser.add_argument('--analyze', action='store_true',
                        help='运行后分析')
    args = parser.parse_args()
    
    # 创建沙盒
    sandbox = ReplaySandbox(args.brain)
    
    # 运行
    sandbox.run(steps=args.steps)
    
    # 分析
    if args.analyze:
        sandbox.analyze()


if __name__ == "__main__":
    main()
