"""
T型迷宫任务 (T-Maze Task)
==========================
POMDP - 强制记忆测试

任务流程:
1. 出生时: 短暂光信号(左/右指示)，持续SIGNAL_DURATION步
2. 盲区: BLIND_ZONE步无信号，Agent必须"记住"指示
3. 决策点: T型岔路口，只有正确方向有奖励
4. 重置: 环境/Agent重置，新一轮随机方向

设计原理:
- 经典神经演化记忆测试范式
- 强制DELAY/循环连接节点的涌现
- 信号消失后必须依赖内部状态
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TMazeConfig:
    """T型迷宫配置"""
    # 迷宫几何
    arm_length: float = 30.0      # 臂长
    stem_length: float = 20.0     # 干长
    corridor_width: float = 4.0   # 走廊宽度
    
    # 信号时序
    signal_duration: int = 5      # 信号持续步数
    blind_zone: int = 20          # 盲区步数
    decision_delay: int = 25      # 信号结束到决策点的步数
    
    # 信号参数
    signal_strength: float = 50.0 # 信号强度
    signal_decay: float = 0.95    # 信号衰减
    
    # 奖励
    correct_reward: float = 100.0 # 正确奖励
    wrong_reward: float = 0.0     # 错误惩罚
    step_penalty: float = 0.1     # 每步惩罚
    
    # 周期性资源消失
    resource_cycle: int = 500     # 资源周期
    resource_fade_steps: int = 50 # 消失过渡步数


class TMazeEnvironment:
    """
    T型迷宫环境
    
    空间结构:
           [左臂]              [右臂]
              ↑                  ↑
              |                  |
    ---Stem--→●←---决策点---盲区---Start(出生点)
    
    信号类型:
    - SIGNAL_LEFT: 左转奖励
    - SIGNAL_RIGHT: 右转奖励
    """
    
    SIGNAL_NONE = 0
    SIGNAL_LEFT = 1
    SIGNAL_RIGHT = 2
    
    def __init__(self, config: Optional[TMazeConfig] = None, device='cpu'):
        self.config = config or TMazeConfig()
        self.device = device
        
        # 当前任务状态
        self.current_signal = self.SIGNAL_NONE
        self.signal_timer = 0          # 信号剩余步数
        self.step_counter = 0          # 总步数
        self.episode_steps = 0         # 当前回合步数
        self.correct_direction = None  # 当前回合正确方向
        
        # 资源周期
        self.resource_visible = True
        self.resource_timer = 0
        
        # 统计
        self.total_episodes = 0
        self.correct_choices = 0
        
    def reset_episode(self) -> int:
        """开始新回合，随机选择方向"""
        self.episode_steps = 0
        
        # 随机选择正确方向 (0=左, 1=右)
        self.correct_direction = np.random.randint(0, 2)
        
        # 立即开始信号
        self.current_signal = self.SIGNAL_LEFT if self.correct_direction == 0 else self.SIGNAL_RIGHT
        self.signal_timer = self.config.signal_duration
        
        self.total_episodes += 1
        
        return self.correct_direction
    
    def step(self) -> Tuple[int, float, bool, dict]:
        """
        推进一步
        
        Returns:
            signal: 当前信号类型 (SIGNAL_NONE/LEFT/RIGHT)
            reward: 当前步的奖励
            at_decision: 是否到达决策点
            info: 额外信息
        """
        self.step_counter += 1
        self.episode_steps += 1
        
        # 更新信号计时器
        if self.signal_timer > 0:
            self.signal_timer -= 1
            if self.signal_timer == 0:
                self.current_signal = self.SIGNAL_NONE
        
        # 检查是否到达决策点 (盲区结束)
        decision_step = self.config.signal_duration + self.config.decision_delay
        at_decision = (self.episode_steps == decision_step)
        
        # 计算奖励
        reward = -self.config.step_penalty  # 每步基础惩罚
        
        if at_decision:
            # 决策点 - 需要外部判断对错
            # 奖励在agent行动后通过record_decision()发放
            pass
            
        # 周期性资源消失
        self.resource_timer += 1
        cycle_pos = self.resource_timer % self.config.resource_cycle
        self.resource_visible = cycle_pos < (self.config.resource_cycle - self.config.resource_fade_steps)
        
        info = {
            'signal': self.current_signal,
            'signal_timer': self.signal_timer,
            'at_decision': at_decision,
            'episode_step': self.episode_steps,
            'resource_visible': self.resource_visible,
            'correct_direction': self.correct_direction,
        }
        
        return self.current_signal, reward, at_decision, info
    
    def record_decision(self, agent_choice: int) -> float:
        """
        记录agent的决策并返回奖励
        
        Args:
            agent_choice: 0=左, 1=右
            
        Returns:
            reward: 奖励值
        """
        is_correct = (agent_choice == self.correct_direction)
        
        if is_correct:
            self.correct_choices += 1
            reward = self.config.correct_reward
        else:
            reward = self.config.wrong_reward
            
        return reward
    
    def get_signal_at_position(self, x: float, y: float) -> float:
        """
        获取某位置的信号强度
        
        用于agent感知
        """
        if self.current_signal == self.SIGNAL_NONE:
            return 0.0
        
        # 信号强度随距离衰减
        # 假设agent在stem上移动
        return self.config.signal_strength * (self.signal_timer / self.config.signal_duration)
    
    def is_in_decision_zone(self, x: float, y: float) -> bool:
        """检查是否在决策区域 (T型路口)"""
        stem_end = self.config.stem_length
        return x >= stem_end - 2 and x <= stem_end + 2
    
    def get_accuracy(self) -> float:
        """获取当前准确率"""
        if self.total_episodes == 0:
            return 0.0
        return self.correct_choices / self.total_episodes
    
    def get_state_summary(self) -> dict:
        """获取状态摘要"""
        return {
            'signal': self.current_signal,
            'signal_timer': self.signal_timer,
            'step': self.episode_steps,
            'resource_visible': self.resource_visible,
            'accuracy': self.get_accuracy(),
            'total_episodes': self.total_episodes,
        }


class TMazePerception:
    """
    T迷宫感知处理器
    
    将T迷宫状态转换为agent可感知的信号
    """
    
    def __init__(self, t_maze: TMazeEnvironment):
        self.t_maze = t_maze
        
    def get_perception_vector(self, agent_x: float, agent_y: float) -> dict:
        """
        获取agent的感知向量
        
        Returns:
            dict: {
                'signal_left': float,   # 左转信号强度
                'signal_right': float,  # 右转信号强度  
                'signal_any': float,    # 任意信号强度
                'in_decision_zone': bool,
                'resource_visible': bool,
            }
        """
        tm = self.t_maze
        
        # 信号感知
        signal_left = 1.0 if tm.current_signal == tm.SIGNAL_LEFT else 0.0
        signal_right = 1.0 if tm.current_signal == tm.SIGNAL_RIGHT else 0.0
        signal_any = max(signal_left, signal_right)
        
        # 决策区域检测
        in_decision = tm.is_in_decision_zone(agent_x, agent_y)
        
        return {
            'signal_left': signal_left,
            'signal_right': signal_right,
            'signal_any': signal_any,
            'in_decision_zone': in_decision,
            'resource_visible': tm.resource_visible,
        }


# ============================================================================
# 集成到主环境的适配器
# ============================================================================

class TMazeAdapter:
    """
    将T迷宫适配到现有Environment类
    """
    
    def __init__(self, env, t_maze_config: Optional[TMazeConfig] = None):
        self.env = env
        self.t_maze = TMazeEnvironment(t_maze_config, env.device)
        self.perception = TMazePerception(self.t_maze)
        
    def on_agent_born(self, agent_idx: int) -> dict:
        """Agent出生时调用 - 开始新回合"""
        correct_dir = self.t_maze.reset_episode()
        return {
            't_maze_active': True,
            'correct_direction': correct_dir,
        }
    
    def on_step(self, agent_idx: int, agent_x: float, agent_y: float) -> dict:
        """每步调用 - 返回感知和奖励"""
        signal, reward, at_decision, info = self.t_maze.step()
        perception = self.perception.get_perception_vector(agent_x, agent_y)
        
        return {
            **perception,
            'signal': signal,
            'at_decision': at_decision,
            'reward': reward,
            'resource_visible': info['resource_visible'],
        }
    
    def record_decision(self, agent_idx: int, agent_choice: int) -> float:
        """记录决策并返回奖励"""
        return self.t_maze.record_decision(agent_choice)


if __name__ == '__main__':
    # 简单测试
    import sys
    sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')
    
    tm = TMazeEnvironment()
    
    print("=" * 50)
    print("T型迷宫测试")
    print("=" * 50)
    
    # 测试3个回合
    for ep in range(3):
        correct_dir = tm.reset_episode()
        direction_name = "左" if correct_dir == 0 else "右"
        print(f"\n回合 {ep+1}: 正确方向={direction_name}")
        
        # 模拟信号阶段
        for step in range(5):
            sig, _, at_dec, info = tm.step()
            print(f"  步{step+1}: 信号={sig}, 盲区={at_dec}")
        
        # 模拟盲区
        for step in range(20):
            sig, _, at_dec, info = tm.step()
            if at_dec:
                print(f"  步{step+6}: 到达决策点!")
        
        # 模拟决策 (随机)
        choice = np.random.randint(0, 2)
        reward = tm.record_decision(choice)
        result = "✓" if choice == correct_dir else "✗"
        print(f"  选择: {'左' if choice==0 else '右'}, 奖励: {reward} {result}")
    
    print(f"\n准确率: {tm.get_accuracy():.1%}")