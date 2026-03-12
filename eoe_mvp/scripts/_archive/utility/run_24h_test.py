#!/usr/bin/env python3
"""
EOE 24-hour Evolution Test - Fixed Version
"""

import os
import sys
import json
import time
import requests
import random
import numpy as np
from datetime import datetime
from typing import List, Dict

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.eoe import Environment, Agent
from core.eoe.node import Node, NodeType

FEISHU_WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/a581dcb1-8994-41e9-aa88-efcb9c0bf9b1"
LOG_FILE = os.path.join(PROJECT_ROOT, "demiurge_decisions.log")

C = type('C', (), {'HEADER': '\033[95m', 'BLUE': '\033[94m', 'GREEN': '\033[92m', 
                    'WARNING': '\033[93m', 'FAIL': '\033[91m', 'ENDC': '\033[0m', 'BOLD': '\033[1m'})()

def send_feishu(msg):
    try:
        requests.post(FEISHU_WEBHOOK, json={"msg_type": "text", "content": {"text": f"[EOE] {msg}"}}, timeout=10)
    except: pass

# ========== Progress Tracking ==========
CHAMPIONS_DIR = os.path.join(PROJECT_ROOT, "champions")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
STAGNATION_THRESHOLD = 20000  # 20000代无进展则提醒
STAGNATION_CHECK_INTERVAL = 5000  # 每5000代检查一次


def load_latest_champion():
    """
    自动扫描 champions/ 和 checkpoints/ 目录，加载最新的或复杂度最高的脑结构存档。
    
    Returns:
        tuple: (genome_dict, source_filename) or (None, None) if no checkpoint found
    """
    import glob
    
    # 搜索模式：champions/*.json 和 checkpoints/*.json
    search_dirs = [CHAMPIONS_DIR, CHECKPOINT_DIR]
    all_json_files = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            pattern = os.path.join(search_dir, "*.json")
            files = glob.glob(pattern)
            all_json_files.extend(files)
    
    if not all_json_files:
        print(f"{C.WARNING}⚠️ 未找到任何存档文件，从零开始演化{C.ENDC}")
        return None, None
    
    # 验证文件是否是有效的脑结构（包含nodes数组且node数>5）
    valid_files = []
    for f in all_json_files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
            if isinstance(data, dict) and 'nodes' in data and isinstance(data.get('nodes'), list):
                if len(data['nodes']) > 5:  # 简单脑结构至少有6个节点
                    node_count = len(data['nodes'])
                    edge_count = len(data.get('edges', []))
                    complexity = node_count + edge_count * 2
                    valid_files.append((f, complexity, node_count, edge_count))
        except:
            pass
    
    if not valid_files:
        print(f"{C.WARNING}⚠️ 未找到有效的脑结构存档，从零开始演化{C.ENDC}")
        return None, None
    
    # 优先选择复杂度最高的文件（其次按修改时间）
    best_file = max(valid_files, key=lambda x: (x[1], os.path.getmtime(x[0])))
    latest_file, complexity, node_count, edge_count = best_file
    
    print(f"{C.BLUE}📂 发现存档: {os.path.basename(latest_file)}")
    print(f"   复杂度: {node_count} 节点 + {edge_count} 边 = {complexity} 复杂度{C.ENDC}")
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        return data, os.path.basename(latest_file)
    except Exception as e:
        print(f"{C.WARNING}⚠️ 加载存档失败: {e}{C.ENDC}")
        return None, None


best_complexity_history = []  # [(gen, complexity, stored), ...]
last_progress_gen = 0
last_best_complexity = 0
last_best_stored = 0

def get_agent_complexity(agent):
    """计算Agent的大脑复杂度"""
    if not hasattr(agent, 'genome') or not agent.genome:
        return 0
    nodes = len(agent.genome.nodes)
    edges = len(agent.genome.edges)
    return nodes + edges

def save_fitness_champion(agents, generation):
    """基于适应度保存冠军 - 优先保存最能生存的个体"""
    if not agents:
        return
    
    # 优先：贮粮最多 > 适应度最高 > 复杂度最高
    def elite_score(a):
        food_stored = getattr(a, 'food_stored', 0)
        fitness = getattr(a, 'fitness', 0)
        complexity = get_agent_complexity(a)
        # 加权评分：贮粮最重要，其次适应度，最后复杂度
        return (food_stored * 1000) + (fitness * 10) + complexity
    
    best = max(agents, key=elite_score)
    score = elite_score(best)
    
    # 文件名: elite_genGENERATION_fitSCORE.json
    filename = f"elite_gen{generation}_fit{int(best.fitness)}_stored{int(getattr(best, 'food_stored', 0))}.json"
    filepath = os.path.join(CHAMPIONS_DIR, filename)
    
    # 序列化brain
    brain_data = {
        'generation': generation,
        'fitness': best.fitness,
        'food_stored': getattr(best, 'food_stored', 0),
        'food_eaten': getattr(best, 'food_eaten', 0),
        'complexity': get_agent_complexity(best),
        'nodes': len(best.genome.nodes),
        'edges': len(best.genome.edges),
        'elite_score': score,
        'genome': best.genome.to_dict() if hasattr(best.genome, 'to_dict') else str(best.genome)
    }
    
    with open(filepath, 'w') as f:
        json.dump(brain_data, f, indent=2)
    
    print(f"{C.GREEN}💾 已保存适应度冠军: {filename} (fitness={best.fitness:.1f}, stored={getattr(best, 'food_stored', 0)}){C.ENDC}")
    return best.fitness


def save_complexity_champion(agents, generation):
    """基于复杂度保存冠军"""
    if not agents:
        return
    
    # 找到复杂度最高的Agent
    best = max(agents, key=get_agent_complexity)
    complexity = get_agent_complexity(best)
    
    # 文件名: complexity_NODES_EDGES_genGENERATION.json
    filename = f"complexity_{len(best.genome.nodes)}_{len(best.genome.edges)}_gen{generation}.json"
    filepath = os.path.join(CHAMPIONS_DIR, filename)
    
    # 序列化brain
    brain_data = {
        'generation': generation,
        'complexity': complexity,
        'nodes': len(best.genome.nodes),
        'edges': len(best.genome.edges),
        'fitness': best.fitness,
        'food_stored': getattr(best, 'food_stored', 0),
        'food_eaten': getattr(best, 'food_eaten', 0),
        'genome': best.genome.to_dict() if hasattr(best.genome, 'to_dict') else str(best.genome)
    }
    
    with open(filepath, 'w') as f:
        json.dump(brain_data, f, indent=2)
    
    return complexity

def check_progress_and_alert(generation, agents):
    """检查进展并在停滞时发送飞书提醒"""
    global last_progress_gen, last_best_complexity, last_best_stored
    
    if not agents:
        return
    
    # 当前最佳
    current_complexity = max(get_agent_complexity(a) for a in agents)
    current_stored = max(getattr(a, 'food_stored', 0) for a in agents)
    
    # 有进展？
    has_progress = (current_complexity > last_best_complexity or 
                   current_stored > last_best_stored)
    
    if has_progress:
        last_progress_gen = generation
        last_best_complexity = current_complexity
        last_best_stored = current_stored
        print(f"{C.GREEN}📈 Progress: complexity={current_complexity}, stored={current_stored}{C.ENDC}")
    else:
        stagnant_for = generation - last_progress_gen
        if stagnant_for >= STAGNATION_THRESHOLD:
            # 发送飞书提醒
            msg = f"⚠️ 进化停滞警告!\n\n🧬 已有 {stagnant_for} 代无进展\n💡 复杂度: {current_complexity}\n🏠 贮粮: {current_stored}\n\n请咨询专家意见!"
            send_feishu(msg)
            print(f"{C.FAIL}{msg}{C.ENDC}")
            # 重置以避免重复提醒
            last_progress_gen = generation
    
    # 记录历史
    best_complexity_history.append((generation, current_complexity, current_stored))

def print_stats(gen, agents, env, speed, elapsed):
    if not agents:
        print(f"{C.FAIL}EXTINCTION!{C.ENDC}")
        return
    alive = sum(1 for a in agents if a.is_alive)
    max_fit = max((a.fitness for a in agents), default=0)
    avg_fit = np.mean([a.fitness for a in agents]) if agents else 0
    
    # 新增：复杂度统计
    complexities = [get_agent_complexity(a) for a in agents]
    max_complexity = max(complexities) if complexities else 0
    avg_complexity = sum(complexities) / len(complexities) if complexities else 0
    
    carried = sum(getattr(a, 'food_carried', 0) for a in agents)
    stored = sum(getattr(a, 'food_stored', 0) for a in agents)
    eaten = sum(getattr(a, 'food_eaten', 0) for a in agents)
    energy_spent = np.mean([a.energy_spent for a in agents]) if agents else 0
    season = "summer" if env.current_season == "summer" else "winter"
    print(f"{C.HEADER}{'='*70}{C.ENDC}")
    print(f"{C.BOLD}Gen {gen} | {elapsed/3600:.2f}h | {speed:.1f} gen/s{C.ENDC}")
    print(f"{C.HEADER}{'='*70}{C.ENDC}")
    print(f"  Pop: {alive} | Max Fit: {max_fit:.1f} | Avg: {avg_fit:.1f}")
    print(f"  Eaten: {eaten} | Carried: {carried} | Stored: {stored}")
    print(f"  Energy/step: {energy_spent:.2f}")
    print(f"  Season: {season}")
    print()
    sys.stdout.flush()

def make_genome():
    """Create genome with 2 sensors + 2 actuators - v3.3: Force 2 connections for cold start"""
    from core.eoe.genome import OperatorGenome
    g = OperatorGenome()
    g.add_node(Node(0, NodeType.SENSOR))
    g.add_node(Node(1, NodeType.SENSOR))
    g.add_node(Node(2, NodeType.ACTUATOR))
    g.add_node(Node(3, NodeType.ACTUATOR))
    # v3.3: 双连接 - 确保传感器能传到执行器
    g.add_edge(0, 2, weight=1.0)
    g.add_edge(1, 3, weight=0.5)  # 第二通道
    return g

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    # Load config
    config_path = os.path.join(PROJECT_ROOT, "physics_config.json")
    try:
        with open(config_path, 'r') as f:
            pc = json.load(f)
        print(f"{C.BLUE}Config: metabolic_alpha={pc.get('metabolic_alpha')}{C.ENDC}")
    except:
        pc = {}
    
    # Environment - lowered metabolic for longer survival
    env = Environment(
        width=100, height=100, n_food=50,   # v3.10: 适度密度
        seasonal_cycle=True, season_length=pc.get('season_length', 80),
        metabolic_alpha=pc.get('metabolic_alpha', 0.005),  # Very low!
        metabolic_beta=pc.get('metabolic_beta', 0.005),
        winter_metabolic_multiplier=pc.get('winter_metabolic_multiplier', 1.5),
        food_energy=pc.get('food_energy', 40)
    )
    env.sensor_range = pc.get('sensor_range', 40.0)  # v3.6: 保持40
    env.food_energy = pc.get('food_energy', 40)
    # v3.11: 禁用所有额外系统以隔离核心问题
    # env.enable_thermal_sanctuary(summer_temp=25, winter_temp=-10, food_heat=15)
    # env.enable_fatigue_system()  # v3.1: 禁用疲劳系统以强制非食即死
    # env.enable_morphological_computation(adhesion_range=3.0, carry_speed_penalty=0.7)
    
    # v2.0: 禁用所有额外环境法则
    # env.enable_entropy_siphon(enabled=True)  # 熵增压力
    # env.enable_kinetic_synapse(enabled=True, trigger_prob=0.8)
    env.enable_metabolic_gravity(enabled=False)
    
    # v2.0: 极端冬季压力 (5.0x!)
    env.winter_metabolic_multiplier = 1.0  # v3.11: 禁用极端冬天
    
    print(f"{C.GREEN}Thermal: ON | Morphological: ON | Season: ON | Siphon: ON | Kinetic: ON | Gravity: ON{C.ENDC}")
    print(f"{C.WARNING}Winter Pressure: 5.0x (EXTREME){C.ENDC}")
    
    # v3.4: 反射支架 - 追踪总进食量用于断奶
    env.total_eaten = 0
    
    # ========== 断点续传：尝试加载冠军脑 ==========
    champion_data, champion_source = load_latest_champion()
    
    # Population
    agents = []
    for i in range(50):   # v3.9: 小种群先测试
        a = Agent(agent_id=i, x=50, y=50, theta=0, initial_energy=150)  # v3.10: 足够25步
        
        if champion_data is not None and i == 0:
            # 零号个体直接使用冠军脑结构
            from core.eoe.genome import OperatorGenome
            a.genome = OperatorGenome.from_dict(champion_data)
        else:
            a.genome = make_genome()
            
            # 如果有冠军脑，对其他个体进行1-2次轻微突变以保持多样性
            if champion_data is not None and i > 0:
                mutation_count = random.randint(1, 2)
                for _ in range(mutation_count):
                    # 随机选择一种突变方式
                    mutation_type = random.choice(['weight', 'add_edge', 'add_node'])
                    if mutation_type == 'weight':
                        a.genome.mutate_weight(sigma=0.2, probability=0.3)
                    elif mutation_type == 'add_edge':
                        a.genome.mutate_add_edge(max_attempts=10)
                    else:
                        a.genome.mutate_add_node()
        
        env.add_agent(a)
        agents.append(a)
    
    # 如果成功加载冠军脑，打印高亮提示
    if champion_data is not None:
        node_count = len(champion_data.get('nodes', []))
        edge_count = len(champion_data.get('edges', []))
        print(f"{C.GREEN}{C.BOLD}✅ 成功从 [{champion_source}] 加载初始冠军脑结构")
        print(f"   (Nodes: {node_count}, Edges: {edge_count})，演化将以此为基点继续！{C.ENDC}\n")
    
    print(f"{C.HEADER}{'='*70}")
    print("EOE 24h Test - Fixed")
    print(f"Time: {datetime.now()}")
    print(f"{'='*70}{C.ENDC}\n")
    
    send_feishu("24h test started (fixed)")
    
    start_time = time.time()
    generation = 0
    last_llm = 0
    
    from scripts.llm_demiurge_loop import LLMDemiurge, generate_epoch_report, BASE_CONSTITUTION
    llm = None
    try:
        llm = LLMDemiurge()
        print(f"{C.GREEN}DeepSeek LLM connected{C.ENDC}\n")
    except Exception as e:
        print(f"{C.WARNING}LLM init failed: {e}{C.ENDC}\n")
    
    print_stats(generation, agents, env, 0, 0)
    
    try:
        STEPS_PER_GEN = 10   # v3.10: 快速迭代
        while True:
            if time.time() - start_time >= 24 * 3600:
                break
            
            # v3.4: 运行完整生命周期
            for step_idx in range(STEPS_PER_GEN):
                env.step()
                if step_idx % 10 == 0:
                    print(f"  Gen {generation} step {step_idx}/{STEPS_PER_GEN}", flush=True)
            
            alive = [a for a in agents if a.is_alive]
            
            # 如果100%死亡，下一代用冠军脑+突变重建（环境压力太大的自然结果）
            if len(alive) == 0:
                print(f"{C.WARNING}⚠️ 种群100%死亡，环境压力过大{C.ENDC}")
            
            # v3.5: 在繁殖前打印统计（获取真实能量消耗）
            if generation % 100 == 0:
                alive_now = len(alive)
                eaten_now = sum(a.food_eaten for a in alive)
                env.total_eaten = env.total_eaten + eaten_now
                all_energy = [a.internal_energy for a in alive]
                elapsed = time.time() - start_time
                speed = generation / elapsed if elapsed > 0 else 0
                # v3.6: Debug food and agent positions
                if env.food_positions:
                    food_pos = env.food_positions[:3]
                    agent_pos = [(a.x, a.y) for a in alive[:3]]
                    print(f"  [Debug] foods={food_pos}", flush=True)
                    print(f"  [Debug] agents={agent_pos}", flush=True)
                print(f"  [Gen {generation}] alive={alive_now}, eaten={eaten_now}, total={env.total_eaten}, speed={speed:.1f}g/s | min={min(all_energy):.1f} max={max(all_energy):.1f} avg={sum(all_energy)/len(all_energy):.1f}", flush=True)
            
            # Reproduce (after stats!)
            if len(alive) < 50:
                alive.sort(key=lambda a: a.fitness, reverse=True)
                best = alive[0] if alive else None
                for i in range(50 - len(alive)):   # v3.9: 保持小种群
                    a = Agent(agent_id=generation*100+i, x=50, y=50, theta=0, initial_energy=150)
                    if best and hasattr(best, 'genome'):
                        a.genome = best.genome.copy()
                        a.genome.mutate_add_node()
                        a.genome.mutate_add_edge()
                    else:
                        a.genome = make_genome()
                    env.add_agent(a)
                    alive.append(a)
            
            agents = alive
            generation += 1
            env.generation = generation  # v3.3: 同步到environment
            if generation % 200 == 0:
                print(f"[Debug] gen={generation}, env.gen={env.generation}", flush=True)
            elapsed = time.time() - start_time
            speed = generation / elapsed if elapsed > 0 else 0
            
            # v3.5: Stats moved to after step loop, before reproduction
            
            if generation % 1000 == 0:
                print_stats(generation, agents, env, speed, elapsed)
                sys.stdout.flush()
                # Also flush log file
                try:
                    with open(LOG_FILE, 'a') as f:
                        f.write(f"Gen {generation}: {len(alive)} alive, {sum(a.food_eaten for a in agents)} eaten\n")
                        f.flush()
                except:
                    pass
            
            # LLM call every 10k
            if llm and generation - last_llm >= 10000:
                print(f"\n{C.BLUE}Calling DeepSeek...{C.ENDC}")
                try:
                    report = generate_epoch_report({'generation': generation, 'agents': agents, 'env': env})
                    resp = llm._send_to_deepseek(BASE_CONSTITUTION[:600], report[:2000])
                    valid = llm._validate_response(resp)
                    if valid:
                        ref = valid.get('reflection', '')
                        story = valid.get('evolution_story', '')
                        pc_new = valid.get('physics_config', {})
                        print(f"{C.BLUE}Reflection: {ref[:100] if ref else 'N/A'}{C.ENDC}")
                        print(f"{C.GREEN}Params: {pc_new}{C.ENDC}")
                        
                        # Apply DeepSeek parameter adjustments to environment
                        if pc_new:
                            applied = []
                            if 'metabolic_alpha' in pc_new:
                                env.metabolic_alpha = pc_new['metabolic_alpha']
                                applied.append(f"metabolic_alpha={pc_new['metabolic_alpha']}")
                            if 'metabolic_beta' in pc_new:
                                env.metabolic_beta = pc_new['metabolic_beta']
                                applied.append(f"metabolic_beta={pc_new['metabolic_beta']}")
                            if 'food_energy' in pc_new:
                                env.food_energy = pc_new['food_energy']
                                applied.append(f"food_energy={pc_new['food_energy']}")
                            if 'season_length' in pc_new:
                                env.season_length = pc_new['season_length']
                                applied.append(f"season_length={pc_new['season_length']}")
                            if 'winter_metabolic_multiplier' in pc_new:
                                env.winter_metabolic_multiplier = pc_new['winter_metabolic_multiplier']
                                applied.append(f"winter_multiplier={pc_new['winter_metabolic_multiplier']}")
                            if 'sensor_range' in pc_new:
                                env.sensor_range = pc_new['sensor_range']
                                applied.append(f"sensor_range={pc_new['sensor_range']}")
                            if applied:
                                print(f"{C.GREEN}✓ Applied: {', '.join(applied)}{C.ENDC}")
                        
                        with open(LOG_FILE, 'a') as f:
                            f.write(f"\nGen: {generation}\nRef: {ref}\nStory: {story}\n")
                        
                        # 发送飞书通知
                        if ref:
                            feishu_msg = f"🤖 DeepSeek @ Gen {generation}\n\n📊 Reflection:\n{ref[:300]}...\n\n⚙️ Applied: {', '.join(applied) if applied else 'none'}"
                            send_feishu(feishu_msg)
                        
                        last_llm = generation
                except Exception as e:
                    print(f"{C.WARNING}LLM error: {e}{C.ENDC}")
            
            # Save complexity champion every 5k
            if generation % 5000 == 0 and generation > 0:
                complexity = save_complexity_champion(agents, generation)
                print(f"{C.GREEN}💾 Saved complexity champion: {complexity}{C.ENDC}")
            
            # 保存适应度冠军 - 每1000代或当有贮粮行为时
            if generation % 1000 == 0 and generation > 0:
                fitness = save_fitness_champion(agents, generation)
            
            # Check progress and stagnation every 5k
            if generation % STAGNATION_CHECK_INTERVAL == 0 and generation > 0:
                check_progress_and_alert(generation, agents)
            
            # Diagnostic every 5k
            if generation % 5000 == 0 and generation > 0:
                avg_e = np.mean([a.energy_spent for a in agents]) if agents else 0
                print(f"{C.WARNING}Diagnostic @ {generation}: energy_spent={avg_e:.2f}{C.ENDC}")
                # Brain complexity check
                if agents:
                    nodes = [len(a.genome.nodes) for a in agents]
                    edges = [len(a.genome.edges) for a in agents]
                    print(f"{C.BLUE}Brain: nodes avg={sum(nodes)/len(nodes):.1f} (max={max(nodes)}), edges avg={sum(edges)/len(edges):.1f} (max={max(edges)}){C.ENDC}")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    print(f"\n{C.GREEN}Done! Total: {generation} generations{C.ENDC}")