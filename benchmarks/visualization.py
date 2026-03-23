"""
轨迹可视化和报告生成模块
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from benchmarks.benchmark_runner import BenchmarkResult, BenchmarkRunner


class TrajectoryVisualizer:
    """轨迹可视化器"""
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_trajectory(
        self,
        trajectory: List[Tuple[float, float]],
        walls: List[Tuple[float, float, float, float]] = None,
        start_pos: Tuple[float, float] = None,
        goal_pos: Tuple[float, float] = None,
        title: str = "Trajectory",
        save_path: str = None
    ) -> str:
        """
        生成轨迹ASCII艺术可视化
        
        Returns:
            可视化字符串
        """
        if not trajectory:
            return "Empty trajectory"
        
        # 边界
        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # 扩展边界
        margin = 5
        x_min, x_max = max(0, x_min - margin), x_max + margin
        y_min, y_max = max(0, y_min - margin), y_max + margin
        
        # 网格大小
        width, height = 60, 30
        scale_x = (x_max - x_min) / width
        scale_y = (y_max - y_min) / height
        
        # 初始化网格
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        def to_grid(x, y):
            gx = int((x - x_min) / scale_x)
            gy = int((y - y_min) / scale_y)
            gx = max(0, min(width - 1, gx))
            gy = max(0, min(height - 1, gy))
            return gx, gy
        
        # 绘制墙壁
        if walls:
            for wx1, wy1, wx2, wy2 in walls:
                for t in np.linspace(0, 1, 20):
                    wx = wx1 + (wx2 - wx1) * t
                    wy = wy1 + (wy2 - wy1) * t
                    gx, gy = to_grid(wx, wy)
                    grid[gy][gx] = '█'
        
        # 绘制轨迹
        for i, (x, y) in enumerate(trajectory):
            gx, gy = to_grid(x, y)
            if i == 0:
                grid[gy][gx] = 'S'  # 起点
            elif i == len(trajectory) - 1:
                grid[gy][gx] = 'E'  # 终点
            else:
                grid[gy][gx] = '·'
        
        # 绘制起点/终点标记
        if start_pos:
            gx, gy = to_grid(start_pos[0], start_pos[1])
            grid[gy][gx] = 'S'
        if goal_pos:
            gx, gy = to_grid(goal_pos[0], goal_pos[1])
            grid[gy][gx] = 'G'
        
        # 生成字符串
        lines = [f"\n{'='*60}\n{title}\n{'='*60}"]
        
        # Y轴标签
        for i, row in enumerate(grid):
            y_val = y_max - (i / height) * (y_max - y_min)
            line = f"{y_val:6.1f} |{''.join(row)}|"
            lines.append(line)
        
        # X轴
        x_line = "       +" + "-"*width + "+"
        lines.append(x_line)
        x_labels = f"       {x_min:.1f}" + " "*(width-10) + f"{x_max:.1f}"
        lines.append(x_labels)
        
        # 图例
        lines.append("\nLegend: S=Start, G=Goal, ·=Path, █=Wall")
        
        return '\n'.join(lines)
    
    def save_trajectory_image(
        self,
        trajectory: List[Tuple[float, float]],
        walls: List[Tuple[float, float, float, float]] = None,
        save_path: str = None
    ) -> str:
        """保存轨迹为PNG图片 (使用matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.output_dir / f"trajectory_{timestamp}.png"
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # 绘制墙壁
            if walls:
                for wx1, wy1, wx2, wy2 in walls:
                    ax.plot([wx1, wx2], [wy1, wy2], 'k-', linewidth=3)
            
            # 绘制轨迹
            if trajectory:
                xs = [p[0] for p in trajectory]
                ys = [p[1] for p in trajectory]
                ax.plot(xs, ys, 'b-', linewidth=1, alpha=0.5)
                ax.scatter(xs, ys, c=range(len(xs)), cmap='viridis', s=20)
                ax.scatter(xs[0], ys[0], c='green', s=100, marker='o', label='Start')
                ax.scatter(xs[-1], ys[-1], c='red', s=100, marker='x', label='End')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Agent Trajectory')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            
            return str(save_path)
        except ImportError:
            return "matplotlib not available"
    
    def compare_trajectories(
        self,
        trajectories: Dict[str, List[Tuple[float, float]]],
        save_path: str = None
    ) -> str:
        """对比多条轨迹"""
        try:
            import matplotlib.pyplot as plt
            
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.output_dir / f"comparison_{timestamp}.png"
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for i, (name, traj) in enumerate(trajectories.items()):
                if traj:
                    xs = [p[0] for p in traj]
                    ys = [p[1] for p in traj]
                    ax.plot(xs, ys, '-', color=colors[i % len(colors)], 
                           linewidth=2, label=name, alpha=0.7)
                    ax.scatter(xs[0], ys[0], color=colors[i % len(colors)], 
                              s=100, marker='o')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Trajectory Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            
            return str(save_path)
        except ImportError:
            return "matplotlib not available"


class BenchmarkReportGenerator:
    """基准测试报告生成器"""
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = TrajectoryVisualizer(output_dir)
    
    def generate_report(
        self,
        results: List[BenchmarkResult],
        title: str = "Benchmark Report",
        include_trajectories: bool = True,
        walls: List[Tuple[float, float, float, float]] = None
    ) -> str:
        """生成完整报告"""
        
        lines = [
            "="*70,
            title.center(70),
            "="*70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # 统计摘要
        lines.extend(self._generate_summary(results))
        
        # 详细结果
        lines.extend(self._generate_details(results))
        
        # 轨迹可视化
        if include_trajectories:
            lines.extend(self._generate_trajectories(results, walls))
        
        # 建议
        lines.extend(self._generate_recommendations(results))
        
        report = '\n'.join(lines)
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"report_{timestamp}.txt"
        report_path.write_text(report)
        
        print(f"\nReport saved to: {report_path}")
        
        return report
    
    def _generate_summary(self, results: List[BenchmarkResult]) -> List[str]:
        """生成统计摘要"""
        lines = [
            "-"*70,
            "SUMMARY",
            "-"*70
        ]
        
        if not results:
            lines.append("No results available")
            return lines
        
        total = len(results)
        success = sum(1 for r in results if r.success)
        success_rate = success / total * 100 if total > 0 else 0
        
        # 按Level分组
        by_level = {}
        for r in results:
            if r.level not in by_level:
                by_level[r.level] = {'total': 0, 'success': 0, 'avg_steps': 0}
            by_level[r.level]['total'] += 1
            by_level[r.level]['success'] += int(r.success)
            by_level[r.level]['avg_steps'] += r.steps_taken
        
        lines.append(f"Total Tasks: {total}")
        lines.append(f"Success Rate: {success_rate:.1f}% ({success}/{total})")
        lines.append("")
        
        lines.append("Results by Level:")
        for level in sorted(by_level.keys()):
            data = by_level[level]
            avg_steps = data['avg_steps'] / data['total'] if data['total'] > 0 else 0
            rate = data['success'] / data['total'] * 100 if data['total'] > 0 else 0
            lines.append(f"  Level {level}: {rate:.0f}% success, avg {avg_steps:.0f} steps")
        
        # 平均指标
        avg_entropy = np.mean([r.trajectory_entropy for r in results])
        avg_efficiency = np.mean([r.path_efficiency for r in results])
        
        lines.append("")
        lines.append(f"Average Trajectory Entropy: {avg_entropy:.2f}")
        lines.append(f"Average Path Efficiency: {avg_efficiency:.2f}")
        lines.append("")
        
        return lines
    
    def _generate_details(self, results: List[BenchmarkResult]) -> List[str]:
        """生成详细结果"""
        lines = [
            "-"*70,
            "DETAILED RESULTS",
            "-"*70
        ]
        
        for r in results:
            status = "✓ PASS" if r.success else "✗ FAIL"
            lines.append(f"\n{r.task_name} (Level {r.level})")
            lines.append(f"  Status: {status}")
            lines.append(f"  Steps: {r.steps_taken}")
            lines.append(f"  Fitness: {r.fitness:.2f}")
            lines.append(f"  Position: ({r.final_position[0]:.1f}, {r.final_position[1]:.1f})")
            lines.append(f"  Trajectory: {len(r.trajectory)} points")
            lines.append(f"  Entropy: {r.trajectory_entropy:.2f}")
            lines.append(f"  Efficiency: {r.path_efficiency:.2f}")
        
        return lines
    
    def _generate_trajectories(
        self, 
        results: List[BenchmarkResult],
        walls: List[Tuple[float, float, float, float]] = None
    ) -> List[str]:
        """生成轨迹可视化"""
        lines = [
            "",
            "-"*70,
            "TRAJECTORIES",
            "-"*70
        ]
        
        for r in results:
            if r.trajectory:
                lines.append(f"\n--- {r.task_name} ---")
                viz = self.visualizer.visualize_trajectory(
                    r.trajectory,
                    walls=walls,
                    goal_pos=r.final_position,
                    title=f"{r.task_name} - Level {r.level}"
                )
                lines.append(viz)
                
                # 保存图片
                try:
                    img_path = self.visualizer.save_trajectory_image(
                        r.trajectory,
                        walls=walls,
                        save_path=self.output_dir / f"{r.task_name}.png"
                    )
                    if img_path and img_path.endswith('.png'):
                        lines.append(f"[Image saved: {img_path}]")
                except Exception as e:
                    pass
        
        return lines
    
    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """生成建议"""
        lines = [
            "",
            "="*70,
            "RECOMMENDATIONS",
            "="*70
        ]
        
        success = sum(1 for r in results if r.success)
        
        if success == 0:
            lines.extend([
                "⚠ All tasks failed - possible issues:",
                "  1. Agent brain is not trained (random weights)",
                "  2. Environment too difficult for the agent",
                "  3. Sensors/actuators not properly connected",
                "",
                "Recommendations:",
                "  - Use a pre-trained brain for testing",
                "  - Start with simpler tasks (Level 1)",
                "  - Verify sensor readings and motor outputs"
            ])
        elif success < len(results) * 0.5:
            lines.extend([
                "⚠ Partial success - consider:",
                "  - Improving brain complexity",
                "  - Adjusting task parameters",
                "  - Checking for specific failure modes"
            ])
        else:
            lines.append("✓ Good performance across most tasks!")
        
        return lines
    
    def export_json(self, results: List[BenchmarkResult], name: str = None) -> str:
        """导出JSON格式"""
        data = []
        for r in results:
            data.append({
                'task_name': r.task_name,
                'level': r.level,
                'success': r.success,
                'steps_taken': r.steps_taken,
                'fitness': r.fitness,
                'final_position': list(r.final_position),
                'trajectory_length': len(r.trajectory),
                'trajectory_entropy': r.trajectory_entropy,
                'path_efficiency': r.path_efficiency
            })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = name or f"results_{timestamp}.json"
        path = self.output_dir / name
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(path)


def visualize_single_run():
    """单次运行可视化示例"""
    from core.eoe.agent import Agent
    from benchmarks.benchmark_runner import BenchmarkRunner
    
    print("Running test with visualization...")
    
    # 创建Agent
    agent = Agent(agent_id=0, x=10.0, y=50.0, add_predictors=False)
    brain = agent.genome
    
    # 运行测试
    runner = BenchmarkRunner(verbose=False)
    results = runner.run_level(1, brain, start_pos=(10.0, 50.0))
    
    # 生成报告
    generator = BenchmarkReportGenerator("benchmarks/results")
    report = generator.generate_report(
        results,
        title="Test Benchmark Report",
        include_trajectories=True
    )
    
    print(report)
    
    # 导出JSON
    json_path = generator.export_json(results)
    print(f"\nJSON exported: {json_path}")


if __name__ == "__main__":
    visualize_single_run()