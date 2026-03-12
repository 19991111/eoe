#!/usr/bin/env python3
"""
v13.0 可视化渲染器 (The Demiurge Render)
=========================================
上帝之眼 - 实时观测GPU仿真宇宙

功能:
- EPF 能量场热力图
- ISF 压痕场发光轨迹
- Agent 散点图 (大小=能量, 颜色=状态)
- 动画保存

运行:
    python scripts/demiurge_render.py --steps 1500 --agents 100
"""

import sys
sys.path.insert(0, '.')

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap


def create_custom_cmaps():
    """创建自定义配色方案"""
    
    # EPF 能量场: 蓝 -> 青 -> 黄 -> 红
    colors_epf = ['#000033', '#003366', '#006699', '#00CCFF', 
                  '#FFFF00', '#FF6600', '#FF0000', '#660000']
    cmap_epf = LinearSegmentedColormap.from_list('epf', colors_epf)
    
    # ISF 压痕场: 黑 -> 紫 -> 粉 -> 白
    colors_isf = ['#000000', '#330033', '#660066', '#FF00FF', 
                  '#FF66FF', '#FFFFFF']
    cmap_isf = LinearSegmentedColormap.from_list('isf', colors_isf)
    
    # Agent 能量: 红 -> 黄 -> 绿
    colors_agent = ['#FF0000', '#FF6600', '#FFFF00', '#00FF00']
    cmap_agent = LinearSegmentedColormap.from_list('agent', colors_agent)
    
    return cmap_epf, cmap_isf, cmap_agent


class DemiurgeRender:
    """
    上帝之眼渲染器
    ==============
    """
    
    def __init__(self, sim: 'IntegratedSimulation', figsize: tuple = (12, 10)):
        self.sim = sim
        self.figsize = figsize
        
        # 创建配色
        self.cmap_epf, self.cmap_isf, self.cmap_agent = create_custom_cmaps()
        
        # 图形设置
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        self.fig.suptitle('EOE v13.0 GPU 仿真宇宙', fontsize=14, fontweight='bold')
        
        # 子图
        self.ax_epf = self.axes[0, 0]
        self.ax_isf = self.axes[0, 1]
        self.ax_agent = self.axes[1, 0]
        self.ax_stats = self.axes[1, 1]
        
        # 初始化空图
        self._init_plots()
        
    def _init_plots(self):
        """初始化各子图"""
        # EPF 热力图
        self.im_epf = self.ax_epf.imshow(
            np.zeros((100, 100)), 
            cmap=self.cmap_epf, 
            origin='lower',
            vmin=0, vmax=100
        )
        self.ax_epf.set_title('EPF 能量场 (Energy Field)')
        self.ax_epf.set_xlabel('X')
        self.ax_epf.set_ylabel('Y')
        plt.colorbar(self.im_epf, ax=self.ax_epf, label='Energy')
        
        # ISF 热力图
        self.im_isf = self.ax_isf.imshow(
            np.zeros((100, 100)), 
            cmap=self.cmap_isf, 
            origin='lower',
            vmin=0, vmax=10
        )
        self.ax_isf.set_title('ISF 压痕场 (Stigmergy Field)')
        self.ax_isf.set_xlabel('X')
        self.ax_isf.set_ylabel('Y')
        plt.colorbar(self.im_isf, ax=self.ax_isf, label='Signal')
        
        # Agent 散点图
        self.scatter_agent = self.ax_agent.scatter(
            [], [], c=[], cmap=self.cmap_agent, 
            s=50, vmin=0, vmax=200, alpha=0.8
        )
        self.ax_agent.set_xlim(0, 100)
        self.ax_agent.set_ylim(0, 100)
        self.ax_agent.set_title('Agent 分布 (存活: 0)')
        self.ax_agent.set_xlabel('X')
        self.ax_agent.set_ylabel('Y')
        self.ax_agent.grid(True, alpha=0.3)
        
        # 统计图表
        self.ax_stats.set_title('演化统计')
        self.ax_stats.set_xlabel('Step')
        self.ax_stats.set_ylabel('Energy')
        self.line_energy, = self.ax_stats.plot([], [], 'b-', label='总能量')
        self.line_alive, = self.ax_stats.plot([], [], 'r-', label='存活数')
        self.ax_stats.legend()
        self.ax_stats.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def update(self, frame: int):
        """更新一帧"""
        # 运行一步仿真
        state = self.sim.step()
        
        # 获取可视化数据
        data = self.sim.get_visualization_data()
        
        # 更新 EPF
        self.im_epf.set_data(data['epf_field'])
        
        # 更新 ISF
        self.im_isf.set_data(data['isf_field'])
        
        # 更新 Agent
        alive = data['alive_mask']
        positions = data['positions'][alive]
        energies = data['energies'][alive]
        
        if len(positions) > 0:
            self.scatter_agent.set_offsets(positions)
            self.scatter_agent.set_array(energies)
        
        self.ax_agent.set_title(f'Agent 分布 (存活: {len(positions)})')
        
        # 更新统计
        history = self.sim.history
        steps = [s.step for s in history]
        energies_hist = [s.total_energy for s in history]
        alive_hist = [s.alive_count for s in history]
        
        self.line_energy.set_data(steps, energies_hist)
        self.line_alive.set_data(steps, alive_hist)
        
        self.ax_stats.relim()
        self.ax_stats.autoscale_view()
        
        # 更新标题
        self.fig.suptitle(
            f'EOE v13.0 GPU 仿真 - Gen {state.generation} Step {state.step} '
            f'| 存活: {state.alive_count} | 能量: {state.total_energy:.1f}',
            fontsize=12
        )
        
        return self.im_epf, self.im_isf, self.scatter_agent, self.line_energy, self.line_alive
    
    def run(self, steps: int, interval: int = 50, save_path: str = None):
        """运行动画
        
        Args:
            steps: 仿真步数
            interval: 动画间隔 (ms)
            save_path: 保存路径 (可选)
        """
        print(f"\n🎬 启动渲染器: {steps} 步")
        
        ani = animation.FuncAnimation(
            self.fig, 
            self.update, 
            frames=steps,
            interval=interval,
            blit=False,
            repeat=False
        )
        
        if save_path:
            print(f"💾 保存动画到: {save_path}")
            ani.save(save_path, writer='pillow', fps=20)
        
        plt.show()
        
        return ani
    
    def snapshot(self, step: int, save_path: str = None):
        """单帧快照"""
        # 运行到指定步数
        for _ in range(step):
            self.sim.step()
        
        # 获取数据
        data = self.sim.get_visualization_data()
        
        # 绘制快照
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # EPF
        axes[0].imshow(data['epf_field'], cmap=self.cmap_epf, origin='lower')
        axes[0].set_title('EPF 能量场')
        
        # ISF
        axes[1].imshow(data['isf_field'], cmap=self.cmap_isf, origin='lower')
        axes[1].set_title('ISF 压痕场')
        
        # Agent
        alive = data['alive_mask']
        positions = data['positions'][alive]
        energies = data['energies'][alive]
        
        scatter = axes[2].scatter(
            positions[:, 0], positions[:, 1], 
            c=energies, cmap=self.cmap_agent, 
            s=50, vmin=0, vmax=200, alpha=0.8
        )
        axes[2].set_xlim(0, 100)
        axes[2].set_ylim(0, 100)
        axes[2].set_title(f'Agent @ Step {step} (存活: {len(positions)})')
        axes[2].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[2], label='Energy')
        
        plt.suptitle(f'EOE v13.0 Snapshot @ Step {step}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 保存快照到: {save_path}")
        
        plt.show()


def run_cli():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='EOE v13.0 Demiurge Render')
    parser.add_argument('--steps', type=int, default=500, help='仿真步数')
    parser.add_argument('--agents', type=int, default=100, help='Agent数量')
    parser.add_argument('--width', type=float, default=100.0, help='环境宽度')
    parser.add_argument('--height', type=float, default=100.0, help='环境高度')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU设备')
    parser.add_argument('--save', type=str, default=None, help='保存动画路径')
    parser.add_argument('--interval', type=int, default=30, help='动画间隔(ms)')
    parser.add_argument('--snapshot', type=int, default=0, help='保存快照的步数')
    
    args = parser.parse_args()
    
    # 导入
    from core.eoe.integrated_simulation import IntegratedSimulation
    
    # 创建仿真
    print(f"\n🚀 初始化仿真: {args.agents} agents, {args.steps} steps")
    sim = IntegratedSimulation(
        n_agents=args.agents,
        env_width=args.width,
        env_height=args.height,
        lifespan=args.steps,
        device=args.device
    )
    
    # 创建渲染器
    render = DemiurgeRender(sim)
    
    if args.snapshot > 0:
        # 单帧快照
        render.snapshot(args.snapshot, save_path=args.save)
    else:
        # 动画
        render.run(args.steps, interval=args.interval, save_path=args.save)


if __name__ == "__main__":
    run_cli()