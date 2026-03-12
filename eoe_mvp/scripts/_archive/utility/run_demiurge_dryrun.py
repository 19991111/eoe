#!/usr/bin/env python3
"""
EOE Demiurge Dry Run - 预实验脚本

用于验证 llm_demiurge_loop.py 的 7 项修复是否正常工作：
1. 环境变量读取 API_KEY
2. JSON 解析优先提取 suggested/new/value
3. 冬季耗竭死亡分类
4. 负数拦截放宽
5. 夏天/冬天死亡分离统计
6. 15% 调整幅度限制
7. 死亡季节判定

运行方式:
    python scripts/run_demiurge_dryrun.py [--extreme-winter] [--no-llm]

输出目录: dryrun_output/
"""

import os
import sys
import json
import time
import random
import shutil
import argparse
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 导入 EOE 核心模块
from core.eoe.environment import Environment
from core.eoe.population import Population
from core.eoe.agent import Agent
from core.eoe.genome import OperatorGenome
from core.eoe.node import Node, NodeType

# 导入 Demiurge 模块
from scripts.llm_demiurge_loop import (
    LLMDemiurge,
    WarReportGenerator,
    BASE_CONSTITUTION
)


# ============================================================
# 配置
# ============================================================

# 快速迭代参数
DRYRUN_POPULATION_SIZE = 10  # 减半
DRYRUN_STEPS_PER_GEN = 80    # 每代步数（增加到80，让Agent活到冬天）
DRYRUN_TOTAL_GENS = 20       # 总代数
DRYRUN_EPOCH_INTERVAL = 3    # 每3代触发一次LLM（共6次）
DRYRUN_OUTPUT_DIR = "dryrun_output"

# 颜色输出
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


def print_subheader(text):
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}▶ {text}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'-'*60}{Colors.ENDC}\n")


def log_to_file(log_path, content):
    """同时输出到终端和日志文件"""
    print(content)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(content + '\n')


# ============================================================
# 初始化环境
# ============================================================

def setup_environment(extreme_winter=False):
    """初始化测试环境 - 寒武纪模式：高食物密度+低动作耗能+饥饿死线"""
    env = Environment(
        width=100.0,
        height=100.0,
        n_food=100,  # 原始汤：极高食物密度
        food_energy=30.0,
        metabolic_alpha=0.05,
        metabolic_beta=0.05,
        season_length=50,
        winter_metabolic_multiplier=2.0,
        seasonal_cycle=True,
    )
    
    # 默认传感器范围
    env.sensor_range = 30.0
    
    # 消除探索惩罚：大幅降低动作耗能
    env.port_weights = {
        'motion': 0.001,
        'offense': 0.001,
        'defense': 0.001,
        'repair': 0.001,
        'signal': 0.001
    }
    
    # 设定饥饿死线：16.5能量 = 30步基础代谢
    # 如果不吃东西，30步后必定饿死
    env.initial_energy = 16.5
    
    # 预实验：禁用疲劳系统
    env.enable_fatigue_system(
        enabled=False,
        max_fatigue=100.0,
        fatigue_build_rate=0.3,
        sleep_danger_prob=0.5,
        enable_wakeup_hunger=False,
        enable_sleep_drop=False
    )

    # 预实验：禁用形态计算
    env.enable_morphological_computation(
        enabled=False,
        adhesion_range=2.5,
        carry_speed_penalty=0.7
    )

    # 预实验：禁用热力学庇护所
    env.enable_thermal_sanctuary(
        enabled=False,
        summer_temp=25.0,
        winter_temp=-5.0,
        food_heat=12.0,
        nest_insulation=0.02
    )

    return env


CHAMPION_BRAIN_PATH = os.path.join(PROJECT_ROOT, "champions", "best_v097_gen280_fit651.json")

# 全局变量记录是否已加载冠军脑
_champion_loaded = False
_champion_data = None


def make_genome(seed=42):
    """Create genome - 优先使用冠军脑结构"""
    global _champion_loaded, _champion_data
    
    # 尝试加载冠军脑（只加载一次）
    if not _champion_loaded and os.path.exists(CHAMPION_BRAIN_PATH):
        try:
            with open(CHAMPION_BRAIN_PATH, 'r') as f:
                _champion_data = json.load(f)
            _champion_loaded = True
        except Exception as e:
            print(f"⚠️ 冠军脑加载失败: {e}")
            _champion_loaded = True  # 标记已尝试，避免重复
    
    # 回退：创建简单脑结构
    g = OperatorGenome()
    g.add_node(Node(0, NodeType.SENSOR))
    g.add_node(Node(1, NodeType.SENSOR))
    g.add_node(Node(2, NodeType.ACTUATOR))
    g.add_node(Node(3, NodeType.ACTUATOR))
    # 双连接 - 确保传感器能传到执行器
    g.add_edge(0, 2, weight=1.0)
    g.add_edge(1, 3, weight=0.5)
    return g


def get_champion_genome():
    """获取冠军脑基因组（用于零号个体）"""
    global _champion_data
    
    if _champion_data is None:
        # 尝试加载
        if os.path.exists(CHAMPION_BRAIN_PATH):
            try:
                with open(CHAMPION_BRAIN_PATH, 'r') as f:
                    _champion_data = json.load(f)
            except:
                pass
    
    if _champion_data is not None:
        return OperatorGenome.from_dict(_champion_data)
    return None


def setup_population(env, size=10):
    """初始化种群 - 直接返回 Agent 列表，手动管理"""
    global _champion_loaded, _champion_data
    
    # 触发冠军脑加载（调用 make_genome 一次来初始化）
    _ = make_genome(42)
    
    # 打印加载信息
    if _champion_loaded and _champion_data is not None:
        node_count = len(_champion_data.get('nodes', []))
        edge_count = len(_champion_data.get('edges', []))
        print(f"{Colors.OKGREEN}✅ 已加载冠军脑 (Nodes: {node_count}, Edges: {edge_count})")
        print(f"   演化将以此为基点继续！{Colors.ENDC}")
    
    agents = []
    champion_genome = get_champion_genome()
    
    for i in range(size):
        # 创建 Agent
        agent = Agent(
            agent_id=i,
            x=50.0 + (i % 5) * 2,
            y=50.0 + (i // 5) * 2
        )

        # 关键修复：不使用冠军脑（存在缺陷：只抓取不进食）
        # 直接使用随机基因组，让演化从0开始
        agent.genome = make_genome(seed=42 + i)
        
        env.add_agent(agent)
        agents.append(agent)

    # 直接返回列表，不包装成 Population 对象
    return agents


def run_generation(env, population, steps=30):
    """运行一代 - 直接使用环境的 step 方法"""
    for step in range(steps):
        # 环境单步更新（包含季节更新、传感器、神经网络、代谢等全部自动处理）
        env.step()

    return population


# ============================================================
# Dry Run 主流程
# ============================================================

def run_dryrun(extreme_winter=False, no_llm=False):
    """运行预实验"""

    # 创建输出目录
    os.makedirs(DRYRUN_OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(DRYRUN_OUTPUT_DIR, f"dryrun_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 备份原配置
    original_config_path = os.path.join(PROJECT_ROOT, "physics_config.json")
    backup_config_path = os.path.join(DRYRUN_OUTPUT_DIR, "physics_config_backup.json")
    if os.path.exists(original_config_path):
        shutil.copy(original_config_path, backup_config_path)
        print(f"✅ 已备份原配置到: {backup_config_path}")

    # 初始化环境
    print_header("🧪 EOE Demiurge Dry Run - 预实验")
    print(f"种群大小: {DRYRUN_POPULATION_SIZE}")
    print(f"每代步数: {DRYRUN_STEPS_PER_GEN}")
    print(f"总代数: {DRYRUN_TOTAL_GENS}")
    print(f"LLM触发间隔: 每 {DRYRUN_EPOCH_INTERVAL} 代")
    print(f"极端冬天: {'是' if extreme_winter else '否'}")
    print(f"跳过LLM: {'是' if no_llm else '否'}")
    print(f"输出目录: {DRYRUN_OUTPUT_DIR}")

    env = setup_environment(extreme_winter=extreme_winter)
    population = setup_population(env, size=DRYRUN_POPULATION_SIZE)

    # 包装成简单的类以便与 WarReportGenerator 兼容
    class SimplePopulation:
        def __init__(self, agents_list):
            self.agents = agents_list

    pop_wrapper = SimplePopulation(population)

    # 当前配置（动态更新）
    current_config = {
        'metabolic_alpha': env.metabolic_alpha,
        'metabolic_beta': env.metabolic_beta,
        'sensor_range': env.sensor_range,
        'season_length': env.season_length,
        'winter_metabolic_multiplier': env.winter_metabolic_multiplier,
        'fatigue_build_rate': env.fatigue_build_rate,
        'food_energy': env.food_energy,
        'winter_temperature': getattr(env, 'winter_temperature', -5.0),
    }

    # 保存初始配置
    initial_config = current_config.copy()

    dryrun_config_path = os.path.join(DRYRUN_OUTPUT_DIR, "physics_config.json")
    with open(dryrun_config_path, 'w') as f:
        json.dump(initial_config, f, indent=2)

    # 创建 Demiurge 实例（使用输出目录的配置）
    if not no_llm:
        demiurge = LLMDemiurge(config_path=dryrun_config_path)

    # 主循环
    for gen in range(1, DRYRUN_TOTAL_GENS + 1):
        print_header(f"📊 Generation {gen}/{DRYRUN_TOTAL_GENS}")

        # ===== 修复土拨鼠之日：重置所有Agent统计 =====
        for agent in population:
            agent.steps_alive = 0
            agent.age = 0
            agent.food_eaten = 0
            agent.food_carried = 0
            agent.food_stored = 0
            agent.fitness = 0
            agent.is_alive = True
            agent.internal_energy = env.initial_energy  # 使用饥饿死线能量
            # 重置其他可能累积的统计
            agent.energy_spent = 0.0
            agent.energy_gained = 0.0
            agent.metabolic_waste = 0.0

        print(f"{Colors.OKBLUE}[DEBUG] Gen {gen} 初始能量: {population[0].internal_energy}, 数量: {len(population)}{Colors.ENDC}")

        # 运行一代
        run_generation(env, population, steps=DRYRUN_STEPS_PER_GEN)

        # 运行后检查
        print(f"{Colors.OKBLUE}[DEBUG] Gen {gen} 结束后能量: {population[0].internal_energy}, steps_alive: {population[0].steps_alive}{Colors.ENDC}")

        # 统计死亡
        alive = sum(1 for a in population if a.is_alive)
        dead = len(population) - alive

        print(f"{Colors.OKGREEN}存活: {alive}/{len(population)}{Colors.ENDC} | "
              f"{Colors.FAIL}死亡: {dead}{Colors.ENDC}")

        # 检查是否触发 LLM
        if not no_llm and gen % DRYRUN_EPOCH_INTERVAL == 0:
            epoch_num = gen // DRYRUN_EPOCH_INTERVAL
            print_subheader(f"🔮 第 {epoch_num} 次 LLM 交互 (Gen {gen})")

            # 生成战报
            war_report = WarReportGenerator(env, pop_wrapper, gen)
            report_text = war_report.generate_report()

            # 打印战报
            log_to_file(log_path, "\n" + "="*70)
            log_to_file(log_path, "📋 完整战报 (War Report)")
            log_to_file(log_path, "="*70)
            log_to_file(log_path, report_text)

            # 调用 LLM
            try:
                # 构建用户提示
                user_prompt = f"""基于以下第 {gen} 纪元的种群生存战报，请分析当前演化瓶颈，并给出物理参数调整建议:

{report_text}

请分析种群面临的核心问题，并返回一个严格的 JSON 对象来调整物理参数。记住:
1. 每次参数调整幅度不得超过当前值的15%
2. 目标是创造"适应度缓坡"而非"断崖"
3. 如果贮粮行为未涌现，考虑降低饥饿压力或增加贮粮奖励
"""

                # 发送请求（带重试机制）
                print(f"{Colors.WARNING}📡 正在咨询 DeepSeek...{Colors.ENDC}")
                
                max_retries = 3
                retry_delays = [5, 10, 20]  # 指数退避
                response = None
                llm_success = False
                
                for attempt in range(max_retries):
                    try:
                        response = demiurge._send_to_deepseek(BASE_CONSTITUTION, user_prompt)
                        llm_success = True
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"{Colors.WARNING}⚠️ LLM调用失败 (尝试 {attempt+1}/{max_retries}): {e}{Colors.ENDC}")
                            print(f"   等待 {retry_delays[attempt]} 秒后重试...")
                            import time
                            time.sleep(retry_delays[attempt])
                        else:
                            print(f"{Colors.FAIL}❌ LLM调用彻底失败: {e}{Colors.ENDC}")
                
                if not llm_success:
                    # 返回虚拟回复，维持原参数
                    print(f"{Colors.WARNING}⚠️ 使用虚拟配置进入下一代{Colors.ENDC}")
                    response = '{"physics_config": {}, "confidence": 0.0, "reasoning": "LLM调用失败，使用原配置"}'

                # 打印原始响应
                log_to_file(log_path, "\n" + "="*70)
                log_to_file(log_path, "💬 DeepSeek 原始回复")
                log_to_file(log_path, "="*70)
                log_to_file(log_path, response[:2000] + "..." if len(response) > 2000 else response)

                # 验证响应
                validated = demiurge._validate_response(response)

                if validated:
                    # 应用配置 - 使用当前配置作为基准
                    old_config = current_config.copy()
                    success = demiurge._apply_config(validated, env)
                    
                    print(f"{Colors.OKBLUE}[LLM] 配置应用结果: success={success}, 当前env.metabolic_alpha={env.metabolic_alpha}{Colors.ENDC}")

                    # 打印参数对比
                    log_to_file(log_path, "\n" + "="*70)
                    log_to_file(log_path, "⚙️ 参数变更对比")
                    log_to_file(log_path, "="*70)

                    pc = validated.get('physics_config', {})
                    for key in ['metabolic_alpha', 'metabolic_beta', 'sensor_range',
                                'season_length', 'winter_metabolic_multiplier',
                                'fatigue_build_rate', 'food_energy', 'winter_temperature']:
                        old_val = old_config.get(key, 'N/A')
                        new_val = pc.get(key, 'N/A')

                        # 确保是数值类型
                        try:
                            if isinstance(old_val, str) and old_val != 'N/A':
                                old_val = float(old_val)
                            if isinstance(new_val, str) and new_val != 'N/A':
                                new_val = float(new_val)
                        except (ValueError, TypeError):
                            log_to_file(log_path, f"  {key}: {old_val} → {new_val} (类型错误)")
                            continue

                        if old_val != new_val and old_val not in ['N/A', 0] and new_val not in ['N/A', None]:
                            try:
                                change_pct = abs(float(new_val) - float(old_val)) / abs(float(old_val)) * 100
                                log_to_file(log_path, f"  {key}: {old_val} → {new_val} ({change_pct:+.1f}%)")
                            except (TypeError, ValueError) as e:
                                log_to_file(log_path, f"  {key}: {old_val} → {new_val} (计算错误)")
                        else:
                            log_to_file(log_path, f"  {key}: {old_val} (无变化)")

                    if success:
                        print(f"{Colors.OKGREEN}✅ 配置已更新!{Colors.ENDC}")
                        # 更新当前配置跟踪
                        current_config['metabolic_alpha'] = env.metabolic_alpha
                        current_config['metabolic_beta'] = env.metabolic_beta
                        current_config['sensor_range'] = env.sensor_range
                        current_config['season_length'] = env.season_length
                        current_config['winter_metabolic_multiplier'] = env.winter_metabolic_multiplier
                        current_config['fatigue_build_rate'] = env.fatigue_build_rate
                        current_config['food_energy'] = env.food_energy
                        if hasattr(env, 'winter_temperature'):
                            current_config['winter_temperature'] = env.winter_temperature
                    else:
                        print(f"{Colors.FAIL}❌ 配置更新失败{Colors.ENDC}")
                else:
                    print(f"{Colors.FAIL}❌ 响应校验失败{Colors.ENDC}")

            except Exception as e:
                print(f"{Colors.FAIL}❌ LLM 调用失败: {e}{Colors.ENDC}")
                log_to_file(log_path, f"错误: {e}")

        # 打印当前环境状态
        if gen % DRYRUN_EPOCH_INTERVAL == 0 or gen == DRYRUN_TOTAL_GENS:
            current_season = getattr(env, 'current_season', 'summer')
            print(f"当前季节: {current_season}")
            print(f"冬季代谢倍率: {env.winter_metabolic_multiplier}")
            if hasattr(env, 'winter_temperature'):
                print(f"冬季温度: {env.winter_temperature}°C")

        # 存活Agent进入下一代 - 重置统计状态
        survivors = [a for a in population if a.is_alive]
        
        # 如果100%死亡，下一代用冠军脑+突变重建（环境压力太大的自然结果）
        if not survivors:
            print(f"{Colors.WARNING}⚠️ 种群100%死亡，环境压力过大{Colors.ENDC}")
            print(f"   下一代将使用冠军脑+突变重建")
        
        if survivors:
            # 关键：清理环境中的死亡Agent，只保留存活者
            env.agents = list(survivors)  # 重建env.agents，只包含存活Agent
            
            # 重置存活Agent的统计状态（避免跨代累积）
            # 使用饥饿死线能量：只有吃到食物才能活过30步
            for agent in survivors:
                agent.steps_alive = 0
                agent.age = 0
                agent.food_eaten = 0
                agent.food_carried = 0
                agent.food_stored = 0
                agent.fitness = 0
                agent.internal_energy = env.initial_energy  # 饥饿死线：16.5能量

            # 补充新Agent以保持种群大小
            # 核心改进：使用存活者的脑结构（精英选择）而不是随机脑
            # 寒武纪模式：大幅提升突变率，强迫结构变异
            while len(survivors) < DRYRUN_POPULATION_SIZE:
                # 精英选择：从存活者中随机选择一个作为"父亲"
                parent = random.choice(survivors)
                
                # 创建新 Agent - 继承父亲的脑结构
                new_agent = Agent(agent_id=len(survivors), x=50.0, y=50.0)
                
                # 复制父亲的脑结构
                new_agent.genome = parent.genome.copy()
                
                # 寒武纪爆发：大幅提升突变率
                # 50%概率添加新节点
                if random.random() < 0.5:
                    new_agent.genome.mutate_add_node()
                # 80%概率添加新边
                if random.random() < 0.8:
                    new_agent.genome.mutate_add_edge(max_attempts=10)
                # 90%概率变异权重
                if random.random() < 0.9:
                    new_agent.genome.mutate_weight(sigma=0.5, probability=0.5)
                
                # 使用饥饿死线能量
                new_agent.internal_energy = env.initial_energy
                
                env.add_agent(new_agent)  # 关键：添加到环境
                survivors.append(new_agent)
            
            print(f"{Colors.OKGREEN}✅ 精英选择: {len(survivors)}个Agent，基础来自{len([s for s in survivors if s.food_eaten > 0])}个有进食经验的个体{Colors.ENDC}")

            population = survivors
            pop_wrapper.agents = population  # 修复：同步更新pop_wrapper的引用
        else:
            # 100%死亡：用随机脑重建整个种群（不再使用冠军脑）
            print(f"{Colors.WARNING}⚠️ 种群团灭! 使用随机脑重建...{Colors.ENDC}")
            
            # 关键：先清空环境中的死亡Agent
            env.agents = []
            
            new_population = []  # 重建种群列表
            for i in range(DRYRUN_POPULATION_SIZE):
                new_agent = Agent(agent_id=i, x=50.0 + (i % 5) * 2, y=50.0 + (i // 5) * 2)
                # 使用随机基因组
                new_agent.genome = make_genome(seed=42 + i)
                # 使用饥饿死线能量
                new_agent.internal_energy = env.initial_energy
                
                env.add_agent(new_agent)
                new_population.append(new_agent)
            
            population = new_population
            pop_wrapper.agents = population  # 修复：同步更新pop_wrapper的引用
            continue  # 继续下一 generation

    # 保存最终配置
    final_config = {
        'metabolic_alpha': env.metabolic_alpha,
        'metabolic_beta': env.metabolic_beta,
        'sensor_range': env.sensor_range,
        'season_length': env.season_length,
        'winter_metabolic_multiplier': env.winter_metabolic_multiplier,
        'fatigue_build_rate': env.fatigue_build_rate,
        'food_energy': env.food_energy,
        'winter_temperature': getattr(env, 'winter_temperature', -5.0),
    }

    with open(dryrun_config_path, 'w') as f:
        json.dump(final_config, f, indent=2)

    print_header("🎉 Dry Run 完成!")
    print(f"日志文件: {log_path}")
    print(f"配置备份: {backup_config_path}")
    print(f"输出配置: {dryrun_config_path}")

    return log_path


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EOE Demiurge Dry Run")
    parser.add_argument("--extreme-winter", action="store_true",
                        help="启用极端冬天模式 (winter_temp=-15°C)")
    parser.add_argument("--no-llm", action="store_true",
                        help="跳过LLM调用，仅测试演化流程")

    args = parser.parse_args()

    run_dryrun(extreme_winter=args.extreme_winter, no_llm=args.no_llm)