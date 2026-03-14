"""
有限代际测试 - 验证掩码池架构
==============================
运行 10 代，每代 100 步，观察种群动态
"""

import torch
import time
import sys
sys.path.insert(0, '.')

from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.environment_gpu import EnvironmentGPU


def run_generation_test(
    n_generations: int = 10,
    steps_per_gen: int = 100,
    initial_pop: int = 100,
    max_agents: int = 1000
):
    """有限代际测试"""
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"🧪 有限代际测试: {n_generations} 代 x {steps_per_gen} 步")
    print(f"{'='*60}")
    print(f"设备: {device}")
    print(f"初始人口: {initial_pop}, 最大容量: {max_agents}")
    print(f"分裂阈值: {PoolConfig.REPRODUCTION_THRESHOLD}")
    
    # 创建环境
    env = EnvironmentGPU(
        width=100, height=100,
        device=device,
        energy_field_enabled=True,
        impedance_field_enabled=True
    )
    
    # 能量场初始化 (提供食物)
    env.energy_field.field[0, 0] = 50.0  # 中心区域高能量
    
    # 创建 Agent 池
    agents = BatchedAgents(
        initial_population=initial_pop,
        max_agents=max_agents,
        env_width=100,
        env_height=100,
        device=device,
        init_energy=150.0
    )
    
    print(f"\n{'Gen':>4} | {'存活':>6} | {'出生':>4} | {'死亡':>4} | {'平均能量':>10} | {'步时(ms)':>8}")
    print("-" * 65)
    
    total_births = 0
    total_deaths = 0
    
    for gen in range(n_generations):
        gen_births = 0
        gen_deaths = 0
        step_times = []
        
        gen_start = time.perf_counter()
        
        for step in range(steps_per_gen):
            step_start = time.perf_counter()
            
            # 步进
            stats = agents.step(env, dt=0.1)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            step_time = (time.perf_counter() - step_start) * 1000
            step_times.append(step_time)
            
            gen_births += stats['births']
            gen_deaths += stats['deaths']
        
        gen_time = (time.perf_counter() - gen_start) * 1000
        
        # 统计
        pop_stats = agents.get_population_stats()
        
        print(f"{gen:>4} | {pop_stats['n_alive']:>6} | {gen_births:>4} | {gen_deaths:>4} | "
              f"{pop_stats['mean_energy']:>10.2f} | {gen_time/steps_per_gen:>8.2f}")
        
        total_births += gen_births
        total_deaths += gen_deaths
        
        # 补充死亡人口 (模拟代际补充，保持种群稳定用于测试)
        batch = agents.get_active_batch()
        if batch.n < initial_pop:
            # 复活一些死掉的槽位来维持测试
            needed = initial_pop - batch.n
            dead_mask = ~agents.alive_mask
            if dead_mask.any():
                respawn_indices = dead_mask.nonzero(as_tuple=True)[0][:needed]
                agents.alive_mask[respawn_indices] = True
                agents.state.energies[respawn_indices] = 150.0
                agents._indices_dirty = True
    
    # 最终统计
    print("-" * 65)
    final_stats = agents.get_population_stats()
    
    print(f"\n📊 测试完成:")
    print(f"  总代数: {n_generations}")
    print(f"  最终存活: {final_stats['n_alive']}")
    print(f"  总出生: {total_births}")
    print(f"  总死亡: {total_deaths}")
    print(f"  平均能量: {final_stats['mean_energy']:.2f}")
    
    return agents, env


if __name__ == "__main__":
    run_generation_test(
        n_generations=10,
        steps_per_gen=100,
        initial_pop=100,
        max_agents=1000
    )