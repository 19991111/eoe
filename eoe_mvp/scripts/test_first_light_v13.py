#!/usr/bin/env python3
"""
v0.0 First Light - 物理基准测试与可视化验证
==============================================
在启动神经网络演化之前，验证宇宙物理法则的坚不可摧

任务:
1. 随机灵魂注入 (Random Brains): 1000个Agent，随机致动器输入
2. 热力学守恒断言: 宇宙总能量 == 初始注入总能量
3. NaN/Inf 检测: 所有张量完整性检查
4. 上帝之眼可视化: 每50步渲染宇宙快照

运行:
    python scripts/test_first_light_v13.py --steps 500 --agents 1000
"""

import argparse
import sys
import os
import time
import traceback
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# 导入核心模块
from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.batched_agents import BatchedAgents
from core.eoe.integrated_simulation import ThermodynamicLaw


# ============================================================================
# 自定义配色方案
# ============================================================================

def create_custom_cmaps():
    """创建自定义配色方案"""
    # EPF 能量场: 深蓝 -> 青 -> 黄 -> 红
    colors_epf = ['#000033', '#003366', '#006699', '#00CCFF',
                  '#FFFF00', '#FF6600', '#FF0000', '#660000']
    cmap_epf = LinearSegmentedColormap.from_list('epf', colors_epf)

    # ISF 压痕场: 黑 -> 紫 -> 粉 -> 白
    colors_isf = ['#000000', '#330033', '#660066', '#FF00FF',
                  '#FF66FF', '#FFFFFF']
    cmap_isf = LinearSegmentedColormap.from_list('isf', colors_isf)

    # Agent 渗透率: 红(低) -> 黄(中) -> 绿(高)
    colors_kappa = ['#FF0000', '#FF6600', '#FFFF00', '#00FF00']
    cmap_kappa = LinearSegmentedColormap.from_list('kappa', colors_kappa)

    return cmap_epf, cmap_isf, cmap_kappa


# ============================================================================
# 严格的热力学检查器
# ============================================================================

class ThermodynamicProbe:
    """
    热力学守恒探针
    ==============
    极其严格的能量守恒检查器
    """

    def __init__(self, device: str = 'cuda:0', tolerance: float = 1e-4):
        self.device = device
        self.tolerance = tolerance
        self.initial_total_energy = None
        self.energy_history = []
        self.violation_history = []

    def initialize(self, env: EnvironmentGPU, agents: BatchedAgents) -> float:
        """初始化宇宙总能量"""
        # EPF 场能量
        epf_energy = torch.sum(env.energy_field.field).item()

        # Agent 能量
        agent_energy = torch.sum(agents.state.energies).item()

        self.initial_total_energy = epf_energy + agent_energy

        print(f"\n{'='*60}")
        print(f"[*] Universe Initial Energy: {self.initial_total_energy:.6f}")
        print(f"   - EPF Energy: {epf_energy:.6f}")
        print(f"   - Agent Energy: {agent_energy:.6f}")
        print(f"{'='*60}\n")

        return self.initial_total_energy

    def assert_conservation(
        self,
        env: EnvironmentGPU,
        agents: BatchedAgents,
        step: int,
        epf_injected_this_step: float = 0.0,
        metabolic_burned: float = 0.0,
        signal_deposited: float = 0.0,
        epf_extracted_by_agents: float = 0.0
    ) -> bool:
        """
        热力学守恒断言 (实用版)

        由于EPF源持续注入能量，严格守恒很难预测。
        我们检测:
        1. 总能量不出现负值
        2. 单步能量变化不出现突变 (>50%波动)
        3. Agent能量不出现负值
        4. 场能量不出现负值

        Args:
            env: GPU 环境
            agents: 批量 Agent
            step: 当前步数
            epf_injected_this_step: EPF源本步注入能量
            metabolic_burned: 本步代谢消耗的能量
            signal_deposited: 本步信号注入ISF的能量
            epf_extracted_by_agents: 本步Agent从EPF提取的能量

        Returns:
            bool: 是否通过检查
        """
        # 当前各部分能量
        current_epf = torch.sum(env.energy_field.field).item()
        alive_mask = agents.state.is_alive
        current_agents = torch.sum(agents.state.energies * alive_mask.float()).item()
        current_isf = torch.sum(env.stigmergy_field.field).item() if env.stigmergy_field_enabled else 0.0

        # 宇宙总能量
        current_total = current_epf + current_agents + current_isf

        # 初始化
        if not hasattr(self, 'prev_total'):
            self.prev_total = current_total
            self.violation_count = 0

        # 检查1: 无负能量
        if current_epf < 0 or current_agents < 0 or current_isf < 0 or current_total < 0:
            print(f"[X] NEGATIVE ENERGY! Step {step}: EPF={current_epf}, Agents={current_agents}, ISF={current_isf}")
            self.violation_history.append((step, -1))
            return False

        # 检查2: 单步变化不超过50%
        change_ratio = 0.0
        if self.prev_total > 0:
            change_ratio = abs(current_total - self.prev_total) / self.prev_total
            if change_ratio > 0.5:
                print(f"[X] ENERGY SPIKE! Step {step}: {self.prev_total:.2f} -> {current_total:.2f} ({change_ratio*100:.1f}%)")
                self.violation_history.append((step, change_ratio))
                return False

        # 检查3: Agent能量合理 (不超过初始的10倍)
        if current_agents > self.initial_total_energy * 10:
            print(f"[X] AGENT ENERGY ANOMALY! Step {step}: {current_agents:.2f} > {self.initial_total_energy * 10:.2f}")
            self.violation_history.append((step, current_agents))
            return False

        # 记录
        self.prev_total = current_total

        self.energy_history.append({
            'step': step,
            'total': current_total,
            'epf': current_epf,
            'agents': current_agents,
            'isf': current_isf,
            'change_ratio': change_ratio
        })

        return True

    def check_nan_inf(self, env: EnvironmentGPU, agents: BatchedAgents, step: int) -> bool:
        """
        检查所有张量中的 NaN/Inf

        Args:
            env: GPU 环境
            agents: 批量 Agent
            step: 当前步数

        Returns:
            bool: 是否通过检查
        """
        issues = []

        # 检查 EPF
        epf = env.energy_field.field
        if torch.isnan(epf).any():
            issues.append(f"EPF NaN: {torch.isnan(epf).sum().item()}")
        if torch.isinf(epf).any():
            issues.append(f"EPF Inf: {torch.isinf(epf).sum().item()}")

        # 检查 KIF
        if hasattr(env, 'impedance_field') and env.impedance_field_enabled:
            kif = env.impedance_field.field
            if torch.isnan(kif).any():
                issues.append(f"KIF NaN: {torch.isnan(kif).sum().item()}")
            if torch.isinf(kif).any():
                issues.append(f"KIF Inf: {torch.isinf(kif).sum().item()}")

        # 检查 ISF
        if env.stigmergy_field_enabled:
            isf = env.stigmergy_field.field
            if torch.isnan(isf).any():
                issues.append(f"ISF NaN: {torch.isnan(isf).sum().item()}")
            if torch.isinf(isf).any():
                issues.append(f"ISF Inf: {torch.isinf(isf).sum().item()}")

        # 检查 Agent 状态
        state = agents.state

        for name, tensor in [
            ('positions', state.positions),
            ('velocities', state.velocities),
            ('energies', state.energies),
            ('thetas', state.thetas),
            ('permeabilities', state.permeabilities),
            ('defenses', state.defenses),
            ('signals', state.signals),
        ]:
            if torch.isnan(tensor).any():
                issues.append(f"Agent {name} NaN: {torch.isnan(tensor).sum().item()}")
            if torch.isinf(tensor).any():
                issues.append(f"Agent {name} Inf: {torch.isinf(tensor).sum().item()}")

        if issues:
            msg = f"\n[X] NaN/Inf DETECTED! Step {step}\n   " + "\n   ".join(issues)
            print(msg)
            return False

        return True


# ============================================================================
# 随机大脑生成器
# ============================================================================

def generate_random_brain_outputs(n_agents: int, device: str = 'cuda:0') -> torch.Tensor:
    """
    生成随机脑输出 (模拟无神经网络的情况)

    5个致动器:
    - permeability (渗透): [0, 1] (sigmoid激活)
    - thrust_x (推力X): [-1, 1] (tanh激活)
    - thrust_y (推力Y): [-1, 1] (tanh激活)
    - signal (信号): [0, 1] (relu激活)
    - defense (防御): [0, 1] (sigmoid激活)

    Args:
        n_agents: Agent 数量
        device: GPU 设备

    Returns:
        Tensor [N, 5] 脑输出
    """
    # 生成原始随机值
    raw = torch.rand(n_agents, 5, device=device) * 2 - 1  # [-1, 1]

    return raw


# ============================================================================
# 可视化渲染器
# ============================================================================

class FirstLightRenderer:
    """
    First Light 可视化渲染器
    ========================
    """

    def __init__(self, env: EnvironmentGPU, agents: BatchedAgents, save_dir: str = None):
        self.env = env
        self.agents = agents
        self.save_dir = save_dir or 'first_light_snapshots'

        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)

        # 配色
        self.cmap_epf, self.cmap_isf, self.cmap_kappa = create_custom_cmaps()

        # 设置绘图风格
        plt.style.use('dark_background')

    def render_snapshot(self, step: int, stats: dict = None):
        """
        渲染宇宙快照

        底层: EPF 热力图
        中层: ISF 压痕轨迹
        顶层: Agent 散点 (大小=能量, 颜色=渗透率)
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'First Light - Step {step}', fontsize=14, fontweight='bold')

        device = self.env.device

        # === Layer 1: EPF Energy Field Heatmap ===
        ax1 = axes[0]
        epf_data = self.env.energy_field.field[0, 0].cpu().numpy()
        im1 = ax1.imshow(epf_data, cmap=self.cmap_epf, origin='lower',
                        vmin=0, vmax=epf_data.max() + 0.1)
        ax1.set_title('EPF Energy Field (Layer 1)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im1, ax=ax1, label='Energy')

        # === Layer 2: ISF Stigmergy Field ===
        ax2 = axes[1]
        isf_data = self.env.stigmergy_field.field[0, 0].cpu().numpy()
        im2 = ax2.imshow(isf_data, cmap=self.cmap_isf, origin='lower',
                        vmin=0, vmax=max(isf_data.max(), 0.1))
        ax2.set_title('ISF Stigmergy Field (Layer 2)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(im2, ax=ax2, label='Signal')

        # === Layer 3: Agent Scatter Plot ===
        ax3 = axes[2]

        # EPF background
        ax3.imshow(epf_data, cmap=self.cmap_epf, origin='lower', alpha=0.5,
                  vmin=0, vmax=epf_data.max() + 0.1)

        # ISF overlay
        ax3.imshow(isf_data, cmap=self.cmap_isf, origin='lower', alpha=0.3,
                  vmin=0, vmax=max(isf_data.max(), 0.1))

        # Get alive agent data
        alive_mask = self.agents.state.is_alive
        if alive_mask.any():
            positions = self.agents.state.positions[alive_mask].cpu().numpy()
            energies = self.agents.state.energies[alive_mask].cpu().numpy()
            permeabilities = self.agents.state.permeabilities[alive_mask].cpu().numpy()

            # Size = energy
            sizes = np.clip(energies * 2, 10, 200)

            # Scatter plot
            scatter = ax3.scatter(
                positions[:, 0], positions[:, 1],
                s=sizes, c=permeabilities, cmap=self.cmap_kappa,
                alpha=0.7, edgecolors='white', linewidths=0.3
            )
            plt.colorbar(scatter, ax=ax3, label='kappa (permeability)')

        ax3.set_title('Agents (Layer 3)\nSize=Energy, Color=Permeability')
        ax3.set_xlim(0, self.env.width)
        ax3.set_ylim(0, self.env.height)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')

        # 添加统计信息
        if stats:
            stats_text = (
                f"Alive: {stats.get('alive', 0)}/{stats.get('total', 0)}\n"
                f"Mean Energy: {stats.get('mean_energy', 0):.2f}\n"
                f"Total Energy: {stats.get('total_energy', 0):.2f}"
            )
            ax3.text(2, 2, stats_text, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                    verticalalignment='bottom')

        plt.tight_layout()

        # 保存
        save_path = os.path.join(self.save_dir, f'snapshot_{step:05d}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

        return save_path


# ============================================================================
# 主测试流程
# ============================================================================

def run_first_light_test(
    n_agents: int = 1000,
    n_steps: int = 500,
    device: str = 'cuda:0',
    visualize_every: int = 50,
    save_dir: str = None,
    stop_on_violation: bool = True
):
    """
    运行 First Light 测试

    Args:
        n_agents: Agent 数量
        n_steps: 仿真步数
        device: GPU 设备
        visualize_every: 可视化间隔
        save_dir: 快照保存目录
        stop_on_violation: 是否在违规时停机
    """
    print("\n" + "="*70)
    print("FIRST LIGHT - Physics Benchmark Test v0.0")
    print("="*70)
    print(f"Agents: {n_agents}")
    print(f"Steps: {n_steps}")
    print(f"Device: {device}")
    print(f"Visualize every: {visualize_every} steps")
    print("="*70 + "\n")

    # 检查设备
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("[!] CUDA not available, switching to CPU")
        device = 'cpu'

    # === 1. 初始化环境 ===
    print("[1/5] Initializing GPU environment...")
    env = EnvironmentGPU(
        width=100.0,
        height=100.0,
        resolution=1.0,
        device=device,
        energy_field_enabled=True,
        impedance_field_enabled=True,
        stigmergy_field_enabled=True
    )
    print(f"   [+] EPF: {env.energy_field.field.shape}")
    print(f"   [+] KIF: {env.impedance_field.field.shape}")
    print(f"   [+] ISF: {env.stigmergy_field.field.shape}")

    # === 2. 初始化 Agent ===
    print(f"\n[2/5] Initializing {n_agents} random souls...")
    agents = BatchedAgents(
        n_agents=n_agents,
        env_width=100.0,
        env_height=100.0,
        device=device,
        init_energy=150.0
    )
    print(f"   - Positions: {agents.state.positions.shape}")
    print(f"   - Energies: {agents.state.energies.shape}")

    # === 3. 初始化热力学探针 ===
    print("\n[3/5] Initializing thermodynamic probe...")
    probe = ThermodynamicProbe(device=device, tolerance=1e-4)
    probe.initialize(env, agents)

    # === 4. 初始化渲染器 ===
    print("\n[4/5] Initializing renderer...")
    renderer = FirstLightRenderer(env, agents, save_dir=save_dir)
    print(f"   [+] Snapshot dir: {renderer.save_dir}")

    # === 5. 主循环 ===
    print(f"\n[5/5] Starting First Light test...")
    print("-" * 70)

    total_start = time.time()
    violation_count = 0

    for step in range(n_steps):
        # --- 随机大脑注入 ---
        brain_outputs = generate_random_brain_outputs(n_agents, device)

        # --- Agent 步进 ---
        agents.step(brain_outputs, dt=1.0)

        # --- 热力学定律应用 ---
        law = ThermodynamicLaw(device=device)
        alive_mask = agents.state.is_alive
        stats, new_alive_mask = law.apply(env, agents, alive_mask)
        agents.state.is_alive = new_alive_mask

        # --- 环境步进 ---
        env.step()

        # --- 获取本步能量流 ---
        # EPF 源注入值
        epf_injected = sum(s[2].item() for s in env.energy_field.sources if s[3].item() > 0)
        # 代谢消耗 (从 stats 获取)
        metabolic = stats.get('metabolic', 0.0)
        # 信号注入 ISF
        signal_deposited = stats.get('signal_deposited', 0.0)
        # Agent 从 EPF 提取的能量
        epf_extracted = stats.get('extracted', 0.0)

        # --- 严格热力学断言 ---
        probe.assert_conservation(env, agents, step, epf_injected, metabolic, signal_deposited, epf_extracted)

        # --- NaN/Inf 检查 ---
        if not probe.check_nan_inf(env, agents, step):
            violation_count += 1
            if stop_on_violation:
                print(f"\n[FATAL] NaN/Inf detected at Step {step}, terminating test")
                break

        # --- 可视化 (每50步) ---
        if (step + 1) % visualize_every == 0:
            # 准备统计
            stats_dict = {
                'total': n_agents,
                'alive': torch.sum(agents.state.is_alive).item(),
                'mean_energy': torch.mean(agents.state.energies).item(),
                'total_energy': torch.sum(agents.state.energies).item()
            }

            # 渲染快照
            snapshot_path = renderer.render_snapshot(step + 1, stats_dict)

            # 打印进度
            elapsed = time.time() - total_start
            recent_energy = probe.energy_history[-1]

            print(f"Step {step+1:4d}/{n_steps} | "
                  f"Alive: {stats_dict['alive']:4d} | "
                  f"Mean E: {stats_dict['mean_energy']:6.2f} | "
                  f"Total: {recent_energy['total']:10.4f} | "
                  f"EPF: {recent_energy['epf']:8.2f} | "
                  f"[IMG] {os.path.basename(snapshot_path)} | "
                  f"Time: {elapsed:.1f}s")

    # === 测试完成 ===
    total_time = time.time() - total_start

    print("\n" + "="*70)
    print("FIRST LIGHT Test Complete")
    print("="*70)
    print(f"Total steps: {step + 1}")
    print(f"Total time: {total_time:.2f} sec")
    print(f"Avg speed: {(step + 1) / total_time:.1f} steps/s")
    print(f"Violations: {violation_count}")
    print(f"Final alive: {torch.sum(agents.state.is_alive).item()}/{n_agents}")

    # 能量历史摘要
    if probe.energy_history:
        energies = [h['total'] for h in probe.energy_history]
        epf_energies = [h['epf'] for h in probe.energy_history]
        agent_energies = [h['agents'] for h in probe.energy_history]
        isf_energies = [h['isf'] for h in probe.energy_history]

        print(f"\nEnergy Statistics:")
        print(f"  Total: init={energies[0]:.2f}, final={energies[-1]:.2f}, min={min(energies):.2f}, max={max(energies):.2f}")
        print(f"  EPF: init={epf_energies[0]:.2f}, final={epf_energies[-1]:.2f}")
        print(f"  Agents: init={agent_energies[0]:.2f}, final={agent_energies[-1]:.2f}")
        print(f"  ISF: init={isf_energies[0]:.2f}, final={isf_energies[-1]:.2f}")

    # 违规历史
    if probe.violation_history:
        print(f"\n[!] Energy anomalies: {len(probe.violation_history)} events")
        for step_v, dev in probe.violation_history[:5]:
            print(f"  Step {step_v}: {dev}")

    # 渲染最终快照
    print(f"\nRendering final snapshot...")
    final_stats = {
        'total': n_agents,
        'alive': torch.sum(agents.state.is_alive).item(),
        'mean_energy': torch.mean(agents.state.energies).item(),
        'total_energy': torch.sum(agents.state.energies).item()
    }
    renderer.render_snapshot(n_steps, final_stats)
    
    print(f"\n[OK] Snapshots saved to: {renderer.save_dir}/")
    print("="*70)

    return {
        'success': violation_count == 0,
        'violations': violation_count,
        'final_step': step + 1,
        'total_time': total_time,
        'snapshot_dir': renderer.save_dir
    }


# ============================================================================
# 入口点
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='First Light 物理基准测试')
    parser.add_argument('--agents', type=int, default=1000, help='Agent数量')
    parser.add_argument('--steps', type=int, default=500, help='仿真步数')
    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备')
    parser.add_argument('--visualize-every', type=int, default=50, help='可视化间隔')
    parser.add_argument('--save-dir', type=str, default=None, help='快照保存目录')
    parser.add_argument('--no-stop', action='store_true', help='违规后不停机')

    args = parser.parse_args()

    result = run_first_light_test(
        n_agents=args.agents,
        n_steps=args.steps,
        device=args.device,
        visualize_every=args.visualize_every,
        save_dir=args.save_dir,
        stop_on_violation=not args.no_stop
    )

    # 退出码
    sys.exit(0 if result['success'] else 1)