#!/usr/bin/env python3
"""
EOE Observer Dashboard - 监控看板

功能:
- 4张A100 GPU利用率实时展示
- 进化成功率曲线 (每个纪元贮粮数)
- DeepSeek最近三次"神启"记录

作者: EOE Research Team
版本: v1.0
"""

import os
import sys
import json
import time
import subprocess
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

LOG_FILE = os.path.join(PROJECT_ROOT, "demiurge_decisions.log")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "champions")
STATS_FILE = os.path.join(PROJECT_ROOT, "eoe_stats.json")
REFRESH_INTERVAL = 5  # 秒


# ============================================================
# 数据收集器
# ============================================================

class StatsCollector:
    """统计数据收集器"""
    
    def __init__(self):
        self.history = []
        self.llm_decisions = []
        self._load_stats()
        self._load_llm_decisions()
    
    def _load_stats(self):
        """加载历史统计"""
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, 'r') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])
            except:
                self.history = []
    
    def _load_llm_decisions(self):
        """加载LLM决策历史"""
        self.llm_decisions = []
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析决策记录
                entries = content.split('='*60)
                for entry in entries[-4:]:  # 最近3条
                    if '纪元:' in entry and 'LLM Reasoning:' in entry:
                        decision = self._parse_decision(entry)
                        if decision:
                            self.llm_decisions.append(decision)
            except:
                pass
    
    def _parse_decision(self, text: str) -> Optional[Dict]:
        """解析决策文本"""
        try:
            lines = text.strip().split('\n')
            decision = {}
            
            for line in lines:
                if line.startswith('纪元:'):
                    decision['generation'] = int(line.split(':')[1].strip())
                elif line.startswith('时间:'):
                    decision['timestamp'] = line.split(':')[1].strip()
                elif line.startswith('LLM Reasoning:'):
                    decision['reasoning'] = line.split(':', 1)[1].strip()
                elif 'physics_config' in line.lower():
                    # 找到配置部分
                    pass
            
            # 尝试提取配置
            import re
            config_match = re.search(r'Physics Config: (\{[^}]+\})', text, re.DOTALL)
            if config_match:
                try:
                    decision['config'] = json.loads(config_match.group(1))
                except:
                    decision['config'] = {}
            
            return decision if decision.get('generation') else None
        except:
            return None
    
    def get_gpu_stats(self) -> List[Dict]:
        """获取GPU统计"""
        gpus = []
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 5:
                            gpus.append({
                                'id': int(parts[0]),
                                'name': parts[1],
                                'utilization': int(parts[2]),
                                'memory_used': int(parts[3]),
                                'memory_total': int(parts[4])
                            })
        except:
            pass
        
        return gpus
    
    def get_evolution_curve(self) -> Dict[str, List]:
        """获取进化曲线数据"""
        # 从检查点收集数据
        generations = []
        food_stored = []
        max_fitness = []
        
        if os.path.exists(CHECKPOINT_DIR):
            try:
                files = [f for f in os.listdir(CHECKPOINT_DIR) 
                        if f.startswith('checkpoint_gen') and f.endswith('.json')]
                
                for f in sorted(files)[-100:]:  # 最近100个
                    gen = int(f.replace('checkpoint_gen', '').replace('.json', ''))
                    filepath = os.path.join(CHECKPOINT_DIR, f)
                    
                    with open(filepath, 'r') as fp:
                        data = json.load(fp)
                    
                    # 统计贮粮数
                    total_stored = sum(
                        a.get('food_stored', 0) 
                        for a in data.get('population_state', [])
                    )
                    
                    generations.append(gen)
                    food_stored.append(total_stored)
                    
                    # 适应度
                    fitnesses = [a.get('fitness', 0) for a in data.get('population_state', [])]
                    max_fitness.append(max(fitnesses) if fitnesses else 0)
                    
            except:
                pass
        
        return {
            'generations': generations,
            'food_stored': food_stored,
            'max_fitness': max_fitness
        }


# ============================================================
# HTML生成器
# ============================================================

class DashboardGenerator:
    """仪表盘HTML生成器"""
    
    @staticmethod
    def generate_html(stats: StatsCollector) -> str:
        """生成HTML页面"""
        gpu_stats = stats.get_gpu_stats()
        evolution = stats.get_evolution_curve()
        decisions = stats.llm_decisions[-3:]  # 最近3条
        
        # 构建GPU卡片HTML
        gpu_cards = ""
        for gpu in gpu_stats:
            util = gpu['utilization']
            mem_used = gpu['memory_used']
            mem_total = gpu['memory_total']
            mem_pct = int(mem_used / mem_total * 100) if mem_total > 0 else 0
            
            color = '#4ade80' if util < 50 else '#fbbf24' if util < 80 else '#f87171'
            mem_color = '#4ade80' if mem_pct < 50 else '#fbbf24' if mem_pct < 80 else '#f87171'
            
            gpu_cards += f"""
            <div class="gpu-card">
                <div class="gpu-title">GPU {gpu['id']}</div>
                <div class="gpu-name">{gpu['name']}</div>
                <div class="metric">
                    <span>利用率</span>
                    <div class="bar-container">
                        <div class="bar" style="width:{util}%; background:{color}">{util}%</div>
                    </div>
                </div>
                <div class="metric">
                    <span>显存</span>
                    <div class="bar-container">
                        <div class="bar" style="width:{mem_pct}%; background:{mem_color}">{mem_used}MB / {mem_total}MB</div>
                    </div>
                </div>
            </div>
            """
        
        # 构建进化曲线HTML
        if evolution['generations']:
            gen_json = json.dumps(evolution['generations'])
            stored_json = json.dumps(evolution['food_stored'])
            fitness_json = json.dumps(evolution['max_fitness'])
            
            curve_html = f"""
            <div class="chart-container">
                <canvas id="evolutionChart"></canvas>
                <script>
                    const genData = {gen_json};
                    const storedData = {stored_json};
                    const fitnessData = {fitness_json};
                    
                    new Chart(document.getElementById('evolutionChart'), {{
                        type: 'line',
                        data: {{
                            labels: genData,
                            datasets: [{{
                                label: '贮粮数',
                                data: storedData,
                                borderColor: '#4ade80',
                                backgroundColor: 'rgba(74, 222, 128, 0.1)',
                                yAxisID: 'y'
                            }}, {{
                                label: '最高适应度',
                                data: fitnessData,
                                borderColor: '#60a5fa',
                                backgroundColor: 'rgba(96, 165, 250, 0.1)',
                                yAxisID: 'y1'
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            interaction: {{ mode: 'index', intersect: false }},
                            scales: {{
                                y: {{ type: 'linear', display: true, position: 'left', title: {{display: true, text: '贮粮数'}} }},
                                y1: {{ type: 'linear', display: true, position: 'right', title: {{display: true, text: '适应度'}}, grid: {{drawOnChartArea: false}}}}
                            }}
                        }}
                    }});
                </script>
            </div>
            """
        else:
            curve_html = "<div class='no-data'>暂无数据，运行守护进程后将自动更新</div>"
        
        # 构建LLM决策HTML
        decisions_html = ""
        for d in decisions:
            reasoning = d.get('reasoning', 'N/A')[:200]
            if len(d.get('reasoning', '')) > 200:
                reasoning += '...'
            
            config_str = json.dumps(d.get('config', {}), indent=2)[:300]
            
            decisions_html += f"""
            <div class="decision-card">
                <div class="decision-header">
                    <span class="generation">纪元 {d.get('generation', '?')}</span>
                    <span class="timestamp">{d.get('timestamp', '')}</span>
                </div>
                <div class="reasoning">{reasoning}</div>
                <div class="config">{config_str}</div>
            </div>
            """
        
        if not decisions_html:
            decisions_html = "<div class='no-data'>暂无LLM决策记录</div>"
        
        # 完整HTML
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EOE Observer Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a; color: #e2e8f0; min-height: 100vh;
        }}
        .header {{
            background: linear-gradient(135deg, #1e293b, #334155);
            padding: 20px; border-bottom: 1px solid #475569;
            display: flex; justify-content: space-between; align-items: center;
        }}
        .header h1 {{ font-size: 24px; color: #4ade80; }}
        .header .time {{ color: #94a3b8; font-size: 14px; }}
        
        .container {{ padding: 20px; max-width: 1400px; margin: 0 auto; }}
        
        .section {{ margin-bottom: 30px; }}
        .section-title {{
            font-size: 18px; color: #94a3b8; margin-bottom: 15px;
            padding-bottom: 10px; border-bottom: 1px solid #334155;
        }}
        
        .gpu-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
        }}
        .gpu-card {{
            background: #1e293b; border-radius: 12px; padding: 20px;
            border: 1px solid #334155;
        }}
        .gpu-title {{ font-size: 20px; font-weight: bold; color: #4ade80; }}
        .gpu-name {{ color: #64748b; font-size: 12px; margin-bottom: 15px; }}
        
        .metric {{ margin-bottom: 12px; }}
        .metric span {{ display: block; color: #94a3b8; font-size: 12px; margin-bottom: 5px; }}
        .bar-container {{ background: #334155; height: 20px; border-radius: 10px; overflow: hidden; }}
        .bar {{ height: 100%; line-height: 20px; text-align: center; font-size: 12px; color: #0f172a; font-weight: bold; }}
        
        .chart-container {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
        
        .decisions-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 15px;
        }}
        .decision-card {{
            background: #1e293b; border-radius: 12px; padding: 20px;
            border: 1px solid #334155;
        }}
        .decision-header {{ display: flex; justify-content: space-between; margin-bottom: 10px; }}
        .generation {{ color: #4ade80; font-weight: bold; }}
        .timestamp {{ color: #64748b; font-size: 12px; }}
        .reasoning {{ color: #e2e8f0; font-size: 14px; margin-bottom: 10px; line-height: 1.5; }}
        .config {{ background: #0f172a; padding: 10px; border-radius: 8px; font-size: 12px; color: #94a3b8; overflow: hidden; }}
        
        .no-data {{ color: #64748b; text-align: center; padding: 40px; }}
        
        .status {{ display: flex; gap: 20px; }}
        .status-item {{ display: flex; align-items: center; gap: 8px; }}
        .status-dot {{ width: 10px; height: 10px; border-radius: 50%; background: #4ade80; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 EOE Observer Dashboard</h1>
        <div class="status">
            <div class="status-item">
                <div class="status-dot"></div>
                <span>运行中</span>
            </div>
            <div class="time">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
    </div>
    
    <div class="container">
        <div class="section">
            <div class="section-title">🖥️ GPU 资源监控</div>
            <div class="gpu-grid">
                {gpu_cards}
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📈 进化成功率曲线</div>
            {curve_html}
        </div>
        
        <div class="section">
            <div class="section-title">🔮 DeepSeek 最近神启</div>
            <div class="decisions-grid">
                {decisions_html}
            </div>
        </div>
    </div>
    
    <script>
        // 自动刷新
        setTimeout(() => location.reload(), {REFRESH_INTERVAL * 1000});
    </script>
</body>
</html>
"""
        return html


# ============================================================
# 主程序
# ============================================================

def run_dashboard(port: int = 8051):
    """运行仪表盘"""
    import http.server
    import socketserver
    
    stats = StatsCollector()
    html = DashboardGenerator.generate_html(stats)
    
    # 写入HTML文件
    html_file = os.path.join(PROJECT_ROOT, "dashboard.html")
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n🌐 仪表盘已生成: {html_file}")
    print(f"📊 请在浏览器中打开: http://localhost:{port}/dashboard.html")
    print(f"🔄 自动刷新间隔: {REFRESH_INTERVAL}秒")


def generate_dashboard():
    """生成静态仪表盘"""
    stats = StatsCollector()
    html = DashboardGenerator.generate_html(stats)
    
    html_file = os.path.join(PROJECT_ROOT, "dashboard.html")
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ 仪表盘已更新: {html_file}")
    return html_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EOE Observer Dashboard")
    parser.add_argument("--port", type=int, default=8051, help="HTTP端口")
    parser.add_argument("--generate", action="store_true", help="仅生成HTML")
    args = parser.parse_args()
    
    if args.generate:
        generate_dashboard()
    else:
        run_dashboard(args.port)