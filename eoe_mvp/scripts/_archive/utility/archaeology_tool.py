#!/usr/bin/env python3
"""
EOE 神经考古工具 (Neural Archaeology Tool)

功能:
- 分析历代冠军大脑的拓扑结构
- 自动寻找高频子图 (Neural Motifs)
- 识别功能微电路: 延迟回路、抑制回路、振荡器等
- 可视化输出

作者: EOE Research Team
版本: v1.0
"""

import os
import sys
import json
import time
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import random

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 尝试导入可视化库
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ matplotlib未安装，可视化功能受限")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("⚠️ networkx未安装，图分析功能受限")


# ============================================================
# 数据结构
# ============================================================

class BrainGraph:
    """大脑图结构"""
    
    NODE_TYPES = ['SENSOR', 'DELAY', 'MOTOR', 'META', 'THRESHOLD', 'PREDICTOR']
    
    def __init__(self, nodes: List[Dict], edges: List[Dict]):
        self.nodes = nodes  # [{id, type, ...}]
        self.edges = edges  # [{source, target, weight, ...}]
        self.node_map = {n['id']: n for n in nodes}
        
        # 构建邻接表
        self.adj = defaultdict(list)  # node_id -> [(target_id, edge_data), ...]
        self.rev_adj = defaultdict(list)  # reverse
        
        for e in edges:
            src, tgt = e.get('source', e.get('source_id')), e.get('target', e.get('target_id'))
            if src is not None and tgt is not None:
                self.adj[src].append((tgt, e))
                self.rev_adj[tgt].append((src, e))
    
    def get_node_type(self, node_id: int) -> str:
        return self.node_map.get(node_id, {}).get('type', 'UNKNOWN')
    
    def get_in_degree(self, node_id: int) -> int:
        return len(self.rev_adj[node_id])
    
    def get_out_degree(self, node_id: int) -> int:
        return len(self.adj[node_id])
    
    def get_subgraph(self, node_ids: Set[int]) -> 'BrainGraph':
        """提取子图"""
        sub_nodes = [n for n in self.nodes if n['id'] in node_ids]
        sub_edges = [e for e in self.edges 
                     if (e.get('source', e.get('source_id')) in node_ids and 
                         e.get('target', e.get('target_id')) in node_ids)]
        return BrainGraph(sub_nodes, sub_edges)
    
    def get_all_paths(self, max_depth: int = 3) -> List[List[int]]:
        """获取所有不超过max_depth的路径"""
        paths = []
        
        def dfs(node: int, path: List[int], depth: int):
            if depth > max_depth:
                return
            paths.append(path.copy())
            for next_node, _ in self.adj[node]:
                dfs(next_node, path + [next_node], depth + 1)
        
        # 从所有节点开始
        for start in self.node_map.keys():
            dfs(start, [start], 0)
        
        return paths
    
    def find_cycles(self) -> List[List[int]]:
        """查找反馈回路"""
        cycles = []
        visited = set()
        
        def dfs(node: int, path: List[int]):
            if node in path and len(path) >= 3:
                # 找到回路
                idx = path.index(node)
                cycles.append(path[idx:])
                return
            if node in visited:
                return
            
            visited.add(node)
            for next_node, _ in self.adj[node]:
                dfs(next_node, path + [next_node])
        
        for start in self.node_map.keys():
            dfs(start, [])
        
        return cycles


class MotifDetector:
    """神经基元检测器"""
    
    # 已知的典型Motif模式
    KNOWN_MOTIFS = {
        'delay_loop': {
            'description': '延迟记忆回路 - 用于记忆巢穴位置',
            'pattern': 'SENSOR -> DELAY*2 -> MOTOR',
            'check': lambda g: self._check_delay_loop(g)
        },
        'inhibition_circuit': {
            'description': '抑制回路 - 用于巢穴内强制卸货',
            'pattern': 'MOTOR -> THRESHOLD -> MOTOR(负向)',
            'check': lambda g: self._check_inhibition(g)
        },
        'oscillator': {
            'description': '振荡器 - 周期性行为',
            'pattern': 'A -> B -> A (循环)',
            'check': lambda g: self._check_oscillator(g)
        },
        'predator_detector': {
            'description': '捕食者检测 - 双传感器协同',
            'pattern': 'SENSOR(SAME) -> THRESHOLD -> MOTOR',
            'check': lambda g: self._check_predator_detector(g)
        },
        'value_predictor': {
            'description': '价值预测 - 误差修正',
            'pattern': 'SENSOR -> PREDICTOR -> THRESHOLD',
            'check': lambda g: self._check_value_predictor(g)
        }
    }
    
    def __init__(self):
        self.motif_library = defaultdict(list)  # motif_name -> [(brain_id, subgraph), ...]
    
    def _check_delay_loop(self, graph: BrainGraph) -> bool:
        """检查是否存在延迟回路"""
        # 找 SENSOR -> DELAY -> DELAY -> MOTOR 模式
        for src, edges in graph.adj.items():
            if graph.get_node_type(src) != 'SENSOR':
                continue
            for tgt, _ in edges:
                if graph.get_node_type(tgt) == 'DELAY':
                    # 找第二层DELAY
                    for tgt2, _ in graph.adj[tgt]:
                        if graph.get_node_type(tgt2) == 'DELAY':
                            # 找MOTOR输出
                            if any(graph.get_node_type(m) == 'MOTOR' 
                                   for m, _ in graph.adj[tgt2]):
                                return True
        return False
    
    def _check_inhibition(self, graph: BrainGraph) -> bool:
        """检查是否存在抑制回路"""
        # 找负权重边
        for edge in graph.edges:
            weight = edge.get('weight', 1.0)
            if weight < 0:
                return True
        return False
    
    def _check_oscillator(self, graph: BrainGraph) -> bool:
        """检查是否存在振荡器"""
        cycles = graph.find_cycles()
        return len(cycles) >= 1
    
    def _check_predator_detector(self, graph: BrainGraph) -> bool:
        """检查是否存在双传感器协同"""
        sensors = [n for n in graph.nodes if n.get('type') == 'SENSOR']
        if len(sensors) >= 2:
            return True
        return False
    
    def _check_value_predictor(self, graph: BrainGraph) -> bool:
        """检查是否存在价值预测器"""
        predictors = [n for n in graph.nodes if n.get('type') == 'PREDICTOR']
        return len(predictors) >= 1
    
    def analyze_brain(self, brain_data: Dict) -> Dict:
        """分析单个大脑，返回检测到的Motif"""
        nodes = brain_data.get('nodes', [])
        edges = brain_data.get('edges', [])
        
        graph = BrainGraph(nodes, edges)
        
        detected = {}
        
        # 基础拓扑统计
        stats = {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'node_types': Counter(n.get('type', 'UNKNOWN') for n in nodes),
            'avg_degree': len(edges) * 2 / len(nodes) if nodes else 0,
            'has_cycles': len(graph.find_cycles()) > 0,
            'sensors': len([n for n in nodes if n.get('type') == 'SENSOR']),
            'motors': len([n for n in nodes if n.get('type') == 'MOTOR']),
            'delays': len([n for n in nodes if n.get('type') == 'DELAY']),
            'predictors': len([n for n in nodes if n.get('type') == 'PREDICTOR']),
            'meta_nodes': len([n for n in nodes if n.get('type') == 'META'])
        }
        
        # 检测已知Motif
        for motif_name, motif_info in self.KNOWN_MOTIFS.items():
            try:
                if motif_info['check'](graph):
                    detected[motif_name] = {
                        'description': motif_info['description'],
                        'pattern': motif_info['pattern']
                    }
            except:
                pass
        
        return {
            'stats': stats,
            'motifs': detected
        }
    
    def find_common_subgraphs(self, brains: List[BrainGraph], min_occurrences: int = 2) -> List[Dict]:
        """查找多个大脑中的共同子图 (高频Motif)"""
        
        # 收集所有3节点路径
        path_counts = Counter()
        
        for brain in brains:
            paths = brain.get_all_paths(max_depth=2)
            for path in paths:
                if len(path) >= 2:
                    # 转换为节点类型序列
                    type_seq = tuple(brain.get_node_type(n) for n in path)
                    path_counts[type_seq] += 1
        
        # 找出高频模式
        common = []
        for path_type, count in path_counts.items():
            if count >= min_occurrences:
                common.append({
                    'pattern': ' -> '.join(path_type),
                    'occurrences': count,
                    'frequency': count / len(brains)
                })
        
        # 按频率排序
        common.sort(key=lambda x: x['occurrences'], reverse=True)
        
        return common[:20]  # 返回Top 20


class ArchaeologyTool:
    """神经考古主工具"""
    
    def __init__(self, champions_dir: str = None):
        self.champions_dir = champions_dir or os.path.join(PROJECT_ROOT, 'champions')
        self.detector = MotifDetector()
        self.brains = []
        
        os.makedirs(self.champions_dir, exist_ok=True)
    
    def load_champions(self, pattern: str = "*.json") -> List[Dict]:
        """加载所有冠军大脑"""
        import glob
        
        files = glob.glob(os.path.join(self.champions_dir, pattern))
        self.brains = []
        
        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    data['_source_file'] = os.path.basename(filepath)
                    self.brains.append(data)
            except Exception as e:
                print(f"⚠️ 加载失败 {filepath}: {e}")
        
        print(f"✅ 加载 {len(self.brains)} 个冠军大脑")
        return self.brains
    
    def analyze_all(self) -> Dict:
        """分析所有大脑"""
        if not self.brains:
            self.load_champions()
        
        results = []
        
        for brain in self.brains:
            analysis = self.detector.analyze_brain(brain)
            analysis['source'] = brain.get('_source_file', 'unknown')
            results.append(analysis)
        
        return results
    
    def find_frequent_motifs(self) -> List[Dict]:
        """查找高频子图"""
        if not self.brains:
            self.load_champions()
        
        # 转换为BrainGraph对象
        graphs = []
        for brain in self.brains:
            nodes = brain.get('nodes', [])
            edges = brain.get('edges', [])
            if nodes:
                graphs.append(BrainGraph(nodes, edges))
        
        if not graphs:
            print("⚠️ 没有有效的脑结构数据")
            return []
        
        return self.detector.find_common_subgraphs(graphs)
    
    def generate_report(self) -> str:
        """生成考古报告"""
        if not self.brains:
            self.load_champions()
        
        analysis = self.analyze_all()
        motifs = self.find_frequent_motifs()
        
        report = []
        report.append("="*70)
        report.append("🧠 EOE 神经考古报告")
        report.append(f"生成时间: {datetime.now().isoformat()}")
        report.append("="*70)
        
        # 总体统计
        total_nodes = sum(a['stats']['total_nodes'] for a in analysis)
        total_edges = sum(a['stats']['total_edges'] for a in analysis)
        
        report.append(f"\n📊 总体统计:")
        report.append(f"  分析大脑数: {len(analysis)}")
        report.append(f"  平均节点数: {total_nodes / len(analysis):.1f}")
        report.append(f"  平均边数: {total_edges / len(analysis):.1f}")
        
        # 节点类型分布
        type_counts = Counter()
        for a in analysis:
            for t, c in a['stats']['node_types'].items():
                type_counts[t] += c
        
        report.append(f"\n🔬 节点类型分布:")
        for node_type, count in type_counts.most_common():
            report.append(f"  {node_type}: {count}")
        
        # 检测到的Motif
        all_motifs = Counter()
        for a in analysis:
            for motif in a['motifs'].keys():
                all_motifs[motif] += 1
        
        if all_motifs:
            report.append(f"\n🔍 检测到的神经基元 (Motifs):")
            for motif, count in all_motifs.most_common():
                report.append(f"  {motif}: {count}/{len(analysis)} 个大脑")
        
        # 高频子图模式
        if motifs:
            report.append(f"\n🧩 高频子图模式:")
            for m in motifs[:10]:
                report.append(f"  {m['pattern']}: {m['occurrences']}次 ({m['frequency']:.1%})")
        
        report.append("\n" + "="*70)
        
        return '\n'.join(report)
    
    def visualize_motifs(self, output_file: str = "motifs_visualization.png"):
        """可视化Motif"""
        if not HAS_MATPLOTLIB:
            print("⚠️ matplotlib未安装，跳过可视化")
            return
        
        if not self.brains:
            self.load_champions()
        
        analysis = self.analyze_all()
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. 节点数分布
        ax1 = axes[0, 0]
        node_counts = [a['stats']['total_nodes'] for a in analysis]
        ax1.hist(node_counts, bins=10, color='steelblue', alpha=0.7)
        ax1.set_xlabel('节点数')
        ax1.set_ylabel('频数')
        ax1.set_title('大脑复杂度分布')
        
        # 2. 边数分布
        ax2 = axes[0, 1]
        edge_counts = [a['stats']['total_edges'] for a in analysis]
        ax2.hist(edge_counts, bins=10, color='coral', alpha=0.7)
        ax2.set_xlabel('边数')
        ax2.set_ylabel('频数')
        ax2.set_title('连接密度分布')
        
        # 3. 节点类型堆叠图
        ax3 = axes[1, 0]
        types = ['SENSOR', 'MOTOR', 'DELAY', 'META', 'PREDICTOR']
        type_data = {t: [] for t in types}
        
        for a in analysis:
            for t in types:
                type_data[t].append(a['stats'].get(t.lower() + 's', 0))
        
        x = range(len(analysis))
        bottom = [0] * len(analysis)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, t in enumerate(types):
            ax3.bar(x, type_data[t], bottom=bottom, label=t, color=colors[i])
            bottom = [b + v for b, v in zip(bottom, type_data[t])]
        
        ax3.set_xlabel('大脑编号')
        ax3.set_ylabel('节点数')
        ax3.set_title('节点类型组成')
        ax3.legend()
        
        # 4. Motif检测统计
        ax4 = axes[1, 1]
        all_motifs = Counter()
        for a in analysis:
            for motif in a['motifs'].keys():
                all_motifs[motif] += 1
        
        if all_motifs:
            motifs_names = list(all_motifs.keys())
            motifs_counts = list(all_motifs.values())
            ax4.barh(motifs_names, motifs_counts, color='mediumpurple')
            ax4.set_xlabel('检测到的大脑数')
            ax4.set_title('神经基元分布')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📊 可视化已保存: {output_file}")
        
        return output_file
    
    def save_report(self, output_file: str = None):
        """保存报告到文件"""
        report = self.generate_report()
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.champions_dir, f"archaeology_report_{timestamp}.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 报告已保存: {output_file}")
        return output_file


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='EOE 神经考古工具')
    parser.add_argument('--champions-dir', '-d', default='champions', help='冠军目录')
    parser.add_argument('--pattern', '-p', default='*.json', help='文件匹配模式')
    parser.add_argument('--visualize', '-v', action='store_true', help='生成可视化')
    parser.add_argument('--output', '-o', default=None, help='输出报告文件')
    
    args = parser.parse_args()
    
    tool = ArchaeologyTool(champions_dir=args.champions_dir)
    
    # 加载数据
    tool.load_champions(pattern=args.pattern)
    
    # 生成报告
    print(tool.generate_report())
    
    # 保存报告
    tool.save_report(args.output)
    
    # 可视化
    if args.visualize and HAS_MATPLOTLIB:
        tool.visualize_motifs()


if __name__ == '__main__':
    main()