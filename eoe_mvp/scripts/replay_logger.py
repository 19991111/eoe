"""
回放录制器 - 将训练过程导出为JSON供可视化使用
"""
import json
import numpy as np

class ReplayLogger:
    """录制每一帧的环境和Agent状态"""
    
    def __init__(self, filepath="replay_data.json"):
        self.filepath = filepath
        self.frames = []
        self.metadata = {}
    
    def set_metadata(self, **kwargs):
        """设置实验元数据"""
        self.metadata.update(kwargs)
    
    def record_frame(self, frame_idx, agents, environment, env_state=None):
        """录制单帧"""
        frame = {
            'frame': frame_idx,
            'agents': [],
            'food': [],
            'obstacles': [],
            'season': getattr(environment, 'current_season', 'summer'),
            'metabolic_mult': getattr(environment, 'winter_metabolic_multiplier', 1.0)
        }
        
        # 录制Agent状态
        for a in agents:
            if not hasattr(a, 'x'): continue
            
            agent_data = {
                'id': id(a),
                'x': float(a.x),
                'y': float(a.y),
                'energy': float(a.internal_energy),
                'food_in_stomach': int(a.food_in_stomach),
                'food_carried': int(a.food_carried),
                'food_stored': int(a.food_stored),
                'food_eaten': int(a.food_eaten),
                'is_alive': a.is_alive,
                'steps_alive': int(a.steps_alive)
            }
            frame['agents'].append(agent_data)
        
        # 录制食物位置
        if hasattr(environment, 'food_positions'):
            for fx, fy in environment.food_positions:
                frame['food'].append({'x': float(fx), 'y': float(fy)})
        
        # 录制障碍物（如果有）
        if env_state and 'obstacles' in env_state:
            frame['obstacles'] = env_state['obstacles']
        
        self.frames.append(frame)
    
    def save(self):
        """保存到JSON文件"""
        data = {
            'metadata': self.metadata,
            'frames': self.frames
        }
        
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ 回放数据已保存: {self.filepath}")
        print(f"   总帧数: {len(self.frames)}")
        print(f"   文件大小: {len(json.dumps(data)) / 1024:.1f} KB")


# 全局实例
logger = ReplayLogger()

def record_training_run(env_class, agent_class, n_generations=10, **kwargs):
    """录制完整训练过程（简短版本用于测试）"""
    from core import Population
    
    logger.set_metadata(
        experiment='EOE障碍物绕行实验',
        n_generations=n_generations,
        **kwargs
    )
    
    pop = Population(**kwargs)
    
    for gen in range(n_generations):
        # 重置
        for a in pop.agents:
            a.food_in_stomach = a.food_carried = a.food_stored = 0
        
        for step in range(pop.lifespan):
            # 录制这一帧
            logger.record_frame(
                gen * pop.lifespan + step,
                pop.agents,
                pop.environment,
                {'obstacles': [{'x': 50, 'y': 55, 'r': 8}]}
            )
            
            # 正常训练步骤
            pop.environment.agents = pop.agents
            pop.environment.step()
            
            for a in pop.agents:
                if a.is_alive and hasattr(a, 'genome'):
                    try:
                        acts = {n: node.activation for n, node in a.genome.nodes.items()}
                        a.genome.hebbian_update(acts, lr=0.01)
                    except:
                        pass
        
        pop.reproduce(verbose=False)
    
    logger.save()
    return logger


if __name__ == '__main__':
    # 测试录制
    from core import Population
    import pickle
    
    # 加载天才脑
    with open('champions/best_v086_genius.pkl', 'rb') as f:
        genius = pickle.load(f)
    
    print("开始录制回放数据...")
    record_training_run(
        None, None,
        n_generations=5,
        population_size=10,
        elite_ratio=0.2,
        lifespan=100,
        use_champion=True,
        n_food=5,
        food_energy=50
    )
