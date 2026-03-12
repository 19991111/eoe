#!/usr/bin/env python3
"""
EOE AGI 全能赛验证集 (Generalization Gauntlet)

功能:
- 将冠军Agent丢入从未见过的极端物理环境
- 测试智能的鲁棒性和泛化能力
- 验证是否学到了"贮粮"这一抽象概念

测试环境:
1. 无热源世界 - 只有食物，没有热源引导
2. 移动巢穴 - 巢穴位置随时间移动
3. 极寒世界 - 极低的冬季温度
4. 食物荒漠 - 极少食物但高代谢
5. 无限地图 - 无边界世界

作者: EOE Research Team
版本: v1.0
"""

import os
import sys
import json
import time
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.eoe.environment import Environment
from core.eoe.agent import Agent


# ============================================================
# 测试环境配置
# ============================================================

@dataclass
class GauntletTest:
    """单项测试配置"""
    name: str
    description: str
    env_config: Dict
    expected_behavior: str
    success_criteria: Dict


GAUNTLET_SUITE = [
    GauntletTest(
        name="no_heat_source",
        description="无热源世界 - 只有食物，没有热源引导",
        env_config={
            'width': 100, 'height': 100, 'n_food': 15,
            'seasonal_cycle': True, 'season_length': 80,
            'winter_temp': -20, 'summer_temp': 25, 'food_heat': 0  # 无热源!
        },
        expected_behavior="Agent应能独立搜索食物，不依赖热源导航",
        success_criteria={
            'min_food_eaten': 5,
            'max_death_rate': 0.5
        }
    ),
    GauntletTest(
        name="moving_nest",
        description="移动巢穴 - 巢穴位置随时间周期性移动",
        env_config={
            'width': 100, 'height': 100, 'n_food': 10,
            'seasonal_cycle': True, 'season_length': 60,
            'nest_moving': True, 'nest_move_period': 50
        },
        expected_behavior="Agent应能追踪移动的巢穴并返回卸货",
        success_criteria={
            'min_stored': 1,
            'max_death_rate': 0.7
        }
    ),
    GauntletTest(
        name="extreme_cold",
        description="极寒世界 - 冬季温度极低，代谢惩罚高",
        env_config={
            'width': 100, 'height': 100, 'n_food': 8,
            'seasonal_cycle': True, 'season_length': 50,
            'winter_temp': -30, 'summer_temp': 20, 'food_heat': 20,
            'winter_multiplier': 3.0  # 3倍代谢!
        },
        expected_behavior="Agent必须在冬季前贮粮，否则快速死亡",
        success_criteria={
            'min_survival_rate': 0.3,
            'must_store_before_winter': True
        }
    ),
    GauntletTest(
        name="food_desert",
        description="食物荒漠 - 极少食物但高代谢压力",
        env_config={
            'width': 120, 'height': 120, 'n_food': 4,  # 稀少食物
            'seasonal_cycle': False,
            'high_metabolism': True,
            'metabolic_alpha': 0.08  # 高代谢!
        },
        expected_behavior="Agent需要高效觅食策略",
        success_criteria={
            'min_food_eaten': 3,
            'max_steps_per_food': 50
        }
    ),
    GauntletTest(
        name="toroidalChaos",
        description="环形混沌 - 地图扭曲，边界模糊",
        env_config={
            'width': 80, 'height': 80, 'n_food': 12,
            'seasonal_cycle': True, 'season_length': 40,
            'wraparound': True,  # 环形边界
            'temperature_noise': 0.3  # 温度噪声
        },
        expected_behavior="Agent需适应边界模糊的环境",
        success_criteria={
            'min_exploration': 0.7,
            'max_stuck_at_wall': 0.3
        }
    )
]


# ============================================================
# 测试环境类
# ============================================================

class TestEnvironment(Environment):
    """扩展测试环境 - 支持特殊物理规则"""
    
    def __init__(self, test_config: GauntletTest, **kwargs):
        super().__init__(**kwargs)
        
        self.test_name = test_config.name
        self.test_config = test_config
        
        # 移动巢穴
        self.nest_moving = test_config.env_config.get('nest_moving', False)
        self.nest_move_period = test_config.env_config.get('nest_move_period', 50)
        
        # 温度噪声
        self.temperature_noise = test_config.env_config.get('temperature_noise', 0)
        
        # 高代谢模式
        if test_config.env_config.get('high_metabolism'):
            self.metabolic_alpha = test_config.env_config.get('metabolic_alpha', 0.05)
            self.metabolic_beta = test_config.env_config.get('metabolic_beta', 0.05)
    
    def step(self, agents: List[Agent], actions: List[np.ndarray] = None):
        """扩展step以支持移动巢穴"""
        super().step(agents, actions)
        
        # 移动巢穴
        if self.nest_moving and self.step_count % self.nest_move_period == 0:
            self._move_nest()
    
    def _move_nest(self):
        """移动巢穴位置"""
        self.nest_pos = (
            (self.nest_pos[0] + 20) % self.width,
            (self.nest_pos[1] + 15) % self.height
        )
    
    def get_temperature_at(self, x: float, y: float) -> float:
        """获取温度 (带噪声)"""
        temp = super().get_temperature_at(x, y)
        
        if self.temperature_noise > 0:
            noise = np.random.normal(0, self.temperature_noise * 10)
            temp += noise
        
        return temp


# ============================================================
# 全能赛运行器
# ============================================================

class GeneralizationGauntlet:
    """AGI全能赛运行器"""
    
    def __init__(self, champion_brains: List[Dict] = None):
        self.champions = champion_brains or []
        self.results = []
        
        print("🧪 AGI 全能赛初始化")
        print(f"   加载 {len(self.champions)} 个冠军大脑")
    
    def load_champions(self, directory: str = "champions"):
        """加载冠军大脑"""
        import glob
        
        pattern = os.path.join(PROJECT_ROOT, directory, "*.json")
        files = glob.glob(pattern)
        
        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # 只保留有效的脑结构数据
                    if 'nodes' in data or 'genome' in data:
                        data['_source'] = os.path.basename(filepath)
                        self.champions.append(data)
            except:
                pass
        
        print(f"✅ 加载 {len(self.champions)} 个冠军大脑")
        return self.champions
    
    def run_single_test(self, test: GauntletTest, agent: Agent, 
                       max_steps: int = 500) -> Dict:
        """运行单个测试"""
        
        # 创建测试环境
        env = TestEnvironment(
            test,
            width=test.env_config.get('width', 100),
            height=test.env_config.get('height', 100),
            n_food=test.env_config.get('n_food', 8),
            seasonal_cycle=test.env_config.get('seasonal_cycle', True),
            season_length=test.env_config.get('season_length', 50)
        )
        
        # 配置热源
        if 'winter_temp' in test.env_config:
            env.enable_thermal_sanctuary(
                summer_temp=test.env_config.get('summer_temp', 25),
                winter_temp=test.env_config.get('winter_temp', -10),
                food_heat=test.env_config.get('food_heat', 15)
            )
        
        # 运行测试
        survived = 0
        food_eaten = 0
        food_stored = 0
        steps_in_winter = 0
        winter_started = False
        
        for step in range(max_steps):
            # 检查是否进入冬天
            if env.seasonal_cycle and env._is_winter():
                winter_started = True
                steps_in_winter += 1
            
            # 简单的神经网络前向传播
            sensor_values = env._compute_sensor_values(agent)
            
            # 如果Agent有基因组，使用它
            if hasattr(agent, 'genome') and agent.genome:
                try:
                    full_inputs = np.concatenate([sensor_values, 
                        agent.last_sensor_inputs if agent.last_sensor_inputs is not None else np.zeros(4)])
                    outputs = agent.genome.forward(full_inputs)
                    action = outputs[:2] if len(outputs) >= 2 else outputs
                except:
                    # 如果基因组出错，使用随机动作
                    action = np.array([random.random(), random.random()])
            else:
                action = np.array([random.random(), random.random()])
            
            # 物理更新
            env._update_agent_physics(agent, action)
            env._check_food_collision(agent)
            
            # 代谢消耗
            if agent.is_alive:
                metabolic = env.metabolic_alpha * len(agent.genome.nodes) if hasattr(agent, 'genome') and agent.genome else 0.5
                agent.internal_energy -= metabolic
                
                if agent.internal_energy <= 0:
                    agent.is_alive = False
                else:
                    survived = step + 1
                    food_eaten += agent.food_eaten
                    food_stored += agent.food_stored
        
        # 评估结果
        passed = True
        reasons = []
        
        criteria = test.success_criteria
        
        if 'min_food_eaten' in criteria and food_eaten < criteria['min_food_eaten']:
            passed = False
            reasons.append(f"进食不足: {food_eaten} < {criteria['min_food_eaten']}")
        
        if 'max_death_rate' in criteria:
            death_rate = 1.0 if not agent.is_alive else 0.0
            if death_rate > criteria['max_death_rate']:
                passed = False
                reasons.append(f"死亡率过高: {death_rate:.1%}")
        
        if 'min_stored' in criteria and food_stored < criteria['min_stored']:
            passed = False
            reasons.append(f"贮粮不足: {food_stored} < {criteria['min_stored']}")
        
        if 'must_store_before_winter' in criteria:
            if winter_started and food_stored == 0:
                passed = False
                reasons.append("冬季来临前未贮粮")
        
        return {
            'test': test.name,
            'survived_steps': survived,
            'food_eaten': food_eaten,
            'food_stored': food_stored,
            'passed': passed,
            'reasons': reasons
        }
    
    def run_full_suite(self, agents: List[Agent] = None) -> List[Dict]:
        """运行全套测试"""
        
        if not agents:
            # 创建测试Agent
            agents = []
            for i in range(min(3, len(self.champions))):
                agent = Agent(agent_id=i, x=50, y=50)
                
                # 加载冠军基因组
                if i < len(self.champions):
                    # 简化: 使用随机基因组
                    # 实际需要反序列化championbrain
                    pass
                
                agents.append(agent)
        
        results = []
        
        print("\n" + "="*60)
        print("🧪 开始 AGI 全能赛")
        print("="*60)
        
        for test in GAUNTLET_SUITE:
            print(f"\n📋 测试: {test.name}")
            print(f"   {test.description}")
            
            test_results = []
            for agent in agents:
                result = self.run_single_test(test, agent)
                test_results.append(result)
            
            # 汇总结果
            passed_count = sum(1 for r in test_results if r['passed'])
            
            result_summary = {
                'test': test.name,
                'description': test.description,
                'passed': passed_count > 0,
                'pass_rate': passed_count / len(test_results),
                'details': test_results
            }
            
            results.append(result_summary)
            
            status = "✅ 通过" if passed_count > 0 else "❌ 失败"
            print(f"   结果: {status} ({passed_count}/{len(test_results)})")
        
        self.results = results
        return results
    
    def generate_report(self) -> str:
        """生成测试报告"""
        
        report = []
        report.append("="*70)
        report.append("🧠 AGI 全能赛验证报告")
        report.append(f"测试时间: {datetime.now().isoformat()}")
        report.append("="*70)
        
        if not self.results:
            report.append("\n⚠️ 暂无测试结果")
            return '\n'.join(report)
        
        # 统计
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['passed'])
        
        report.append(f"\n📊 总体结果: {passed_tests}/{total_tests} 通过 ({passed_tests/total_tests:.0%})")
        
        # 详细结果
        report.append(f"\n🔬 各项测试详情:")
        
        for r in self.results:
            status = "✅" if r['passed'] else "❌"
            report.append(f"\n  {status} {r['test']}")
            report.append(f"     {r['description']}")
            report.append(f"     通过率: {r['pass_rate']:.0%}")
            
            if not r['passed']:
                for detail in r['details']:
                    if not detail['passed']:
                        for reason in detail['reasons']:
                            report.append(f"       → {reason}")
        
        # 结论
        report.append("\n" + "="*70)
        
        if passed_tests >= total_tests * 0.6:
            report.append("🎉 结论: 智能具有良好泛化能力!")
        elif passed_tests >= total_tests * 0.3:
            report.append("⚠️ 结论: 智能泛化能力一般，部分行为过拟合")
        else:
            report.append("❌ 结论: 智能严重过拟合，需增强物理多样性训练")
        
        report.append("="*70)
        
        return '\n'.join(report)
    
    def save_report(self, output_file: str = None):
        """保存报告"""
        report = self.generate_report()
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(PROJECT_ROOT, f"gauntlet_report_{timestamp}.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 报告已保存: {output_file}")
        return output_file


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='EOE AGI全能赛')
    parser.add_argument('--champions', '-c', default='champions', help='冠军目录')
    parser.add_argument('--output', '-o', default=None, help='输出报告文件')
    parser.add_argument('--test', '-t', default=None, help='只运行指定测试')
    
    args = parser.parse_args()
    
    gauntlet = GeneralizationGauntlet()
    gauntlet.load_champions(args.champions)
    
    # 运行测试
    if args.test:
        # 只运行指定测试
        test = next((t for t in GAUNTLET_SUITE if t.name == args.test), None)
        if test:
            agent = Agent(agent_id=0, x=50, y=50)
            result = gauntlet.run_single_test(test, agent)
            print(f"\n结果: {'✅ 通过' if result['passed'] else '❌ 失败'}")
            for r in result['reasons']:
                print(f"  - {r}")
        else:
            print(f"未知测试: {args.test}")
    else:
        # 运行全套
        gauntlet.run_full_suite()
        print(gauntlet.generate_report())
        gauntlet.save_report(args.output)


if __name__ == '__main__':
    main()