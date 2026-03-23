# EOE: Evolving Organisms Engine

**A GPU-Accelerated Evolutionary Simulation Platform for Studying Cognitive Emergence in Digital Organisms**

*Technical Report v18.0 - Complexity-Driven Co-Evolution*

---

## 1. Abstract

This technical report presents EOE (Evolving Organisms Engine), a GPU-accelerated evolutionary simulation framework designed to investigate how digital brains evolve learning capabilities, memory structures, reasoning mechanisms, and complex cognitive architectures through Darwinian selection. The system simulates populations of neural network-based agents in a 2D environment where they must acquire energy, reproduce, and adapt to evolving ecological pressures.

---

## 2. System Overview

### 2.1 Core Architecture

EOE consists of four primary subsystems:

| Component | Description |
|-----------|-------------|
| **Agent Pool** (batched_agents.py) | GPU-accelerated population management supporting 10,000+ concurrent agents |
| **DAG Brain** (genome.py) | Directed acyclic graph neural architecture with 7 node types |
| **Field System** | Multi-physics environment: Energy, Impedance, Stigmergy, Stress, Wind fields |
| **Evolution Engine** | Population dynamics: mutation, reproduction, selection, death |

### 2.2 Neural Architecture

Agents possess brains structured as directed acyclic graphs (DAGs) with the following node types:

- **SENSOR**: Environmental perception (energy, impedance, stigmergy, position, heading)
- **ADD**: Summation aggregation
- **MULTIPLY**: Gating and multiplicative interactions
- **THRESHOLD**: Non-linear activation
- **DELAY**: Temporal state retention / working memory
- **MODULATOR**: Neuromodulation for learning rate adaptation
- **ACTUATOR**: Motor outputs (force, rotation)

---

## 3. Methodology

### 3.1 Evolutionary Mechanisms

| Mechanism | Implementation | Version |
|-----------|----------------|---------|
| **Baldwin Effect** | Hebbian learning enables "learning to learn" | v14 |
| **Evolutionary Ratchet** | SuperNode pattern freezing to preserve complexity | v14 |
| **Net-2-Net** | Noisy identity initialization for structural expansion | v17.1 |
| **Soft Carrying Capacity** | Global energy budget调节种群规模 | v17.2 |
| **Crowding Penalty** | Density-dependent energy acquisition decay | v17.2 |
| **Nonlinear Metabolism** | Sigmoid cost curve for complex brains | v15 |

### 3.2 Ecological Pressures

To drive cognitive emergence, EOE implements several environmental challenges:

- **T-Maze Task**: POMDP navigation requiring memory and reasoning
- **Red Queen Dynamics**: Intelligent prey that evade hunters
- **Deceptive Landscape**: Periodically invisible energy sources testing prediction
- **Curriculum Learning**: Progressive difficulty escalation

### 3.3 Field Physics

The environment operates on a field-based physics system:

```
Field_t+1 = Field_t + Sources - Consumption + Diffusion + Decay
```

- **Kinetic Impedance Field (KIF)**: Walls and obstacles (repulsive)
- **Energy Field (EPF)**: Food sources (attractive)
- **Stigmergy Field (ISF)**: Pheromone-like traces left by agents

---

## 4. Experimental Results

### 4.1 Benchmark Performance (v17.2)

| Level | Task | Success Rate |
|-------|------|--------------|
| 1 | T-Maze Straight | 0% |
| 2 | T-Maze Delayed | 33% (1/3) |
| 3 | T-Maze Stigmergy | 0% |

**Analysis**: The 0-33% success rate indicates that evolved topologies lack sufficient memory回路 for complex navigation. This reveals that the current evolutionary environment is too simple to drive emergence of reasoning capabilities.

### 4.2 Emergent Structures

| Structure ID | Complexity | Topology | Feature |
|--------------|------------|----------|---------|
| struct_66957 | 16.89 | 5 nodes, 6 edges | Feedback + Multiplication |
| struct_58092 | 12.08 | 4 nodes, 5 edges | Feedback connections |
| struct_37797 | 12.08 | 4 nodes, 5 edges | Negative weights |

**Key Observations**:
- 17% of evolved structures contain feedback connections → memory capability emergence
- DELAY nodes frequently appear → internal state retention
- MULTIPLY nodes correlate with increased complexity

---

## 5. Implementation Details

### 5.1 Project Structure

```
eoe_mvp/
├── core/eoe/
│   ├── batched_agents.py      # GPU agent pool
│   ├── genome.py              # DAG brain genome
│   ├── environment.py         # Environment simulator
│   ├── fields/                # Physics field system
│   │   ├── energy.py
│   │   ├── impedance.py
│   │   ├── stigmergy.py
│   │   └── wind.py
│   └── t_maze.py              # Benchmark tasks
├── configs/
│   ├── base.py                # PoolConfig base
│   └── presets/               # Experimental configs
├── benchmarks/
│   └── benchmark_runner.py    # Evaluation framework
└── scripts/
    └── run_v17_modulator.py   # Main experiment runner
```

### 5.2 Running Experiments

```bash
# Standard experiment (v17.2 stable)
PYTHONPATH=. python scripts/run_v17_modulator.py --steps 10000

# Custom configuration
PYTHONPATH=. python -c "
from configs.presets.stable import StableConfig
from scripts.run_v17_modulator import run_experiment
config = StableConfig()
run_experiment(config, steps=5000)
"
```

### 5.3 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| BASE_METABOLISM | 0.008 | Base energy cost per step |
| REPRODUCTION_THRESHOLD | 60.0 | Energy required for reproduction |
| MUTATION_RATE | 0.5 | Structural mutation probability |
| SUPERNODE_ENABLED | True | Enable SuperNode pattern mining |
| ENERGY_SOURCES | 30 | Number of energy food sources |

---

## 6. Discussion & Future Work

### 6.1 Current Limitations

1. **Topology Constraints**: Evolutionary environment too simple for memory回路 emergence
2. **Insufficient Warm-up**: Simple environments cannot guide advanced strategy development

### 6.2 Proposed Directions

**Option A: High-Pressure Dynamic Niches**
- Mobile/vanishing energy sources
- Mobile predators
- KIF storms

*Goal: Force emergence of memory回路 and predictive models*

**Option B: Meta-Learning Evolution**
- Each generation faces different random terrain
- Reward "rapid adaptation" over "specific skills"

*Goal: Evolve meta-learners that "learn how to learn"*

---

## 7. References

- Darwin, C. (1859). *On the Origin of Species*
- Baldwin, J.M. (1896). A New Factor in Evolution. *The American Naturalist*
- Stanley, K.O. & Miikkulainen, R. (2002). Evolving Neural Networks through Augmenting Topologies. *Evolutionary Computation*

---

## Appendix: Version History

| Version | Key Milestone |
|---------|---------------|
| v14 | Baldwin Effect + Evolutionary Ratchet |
| v15 | Nonlinear Metabolism + T-Maze + Intelligent Prey |
| v15.2 | Brain Preloading - Complex intelligence emergence |
| v16 | Deceptive Landscape - Dynamic energy field |
| v16.17 | Population collapse fix - 60 population / 58 complex structures |
| v17.1 | Net-2-Net zero-weight fix |
| v17.2 | Soft carrying capacity + Crowding penalty |
| v18.0 | **Complexity-Driven Co-Evolution (CDC)** - 环境参数作为可演化物种 |

---

## v18.0: Complexity-Driven Co-Evolution (CDC)

### Core Innovation

Instead of treating environment parameters as fixed knobs, we evolve them as a separate species in a dual-loop evolutionary system:

```
外层循环: 演化环境参数 (μ+λ 遗传算法)
    ↓
内层循环: 在演化出的环境中演化大脑
    ↓
适应度: complexity_delta × survival_rate × energy_efficiency
```

### Key Mechanisms

1. **Environment Genome**: 将环境参数打包为可演化基因
   - `base_metabolism`, `neural_cost`, `food_energy`, `food_count`
   - `predation_rate`, `predation_range`, `season_length`, `winter_multiplier`

2. **Brain Torch Passing (火炬传递)**: 每代提取最优大脑模板传递给下一代
   - 解决大脑连续性问题

3. **Energy Efficiency Constraint**: 防止网络臃肿化
   - 只有"有用的复杂"才被奖励

### Experimental Results (40代演化)

| 配置 | 值 |
|------|-----|
| 宇宙数 | 8 |
| 内层步数 | 350 |
| Agent数 | 60 |
| 演化代数 | 40 |

| 指标 | 结果 |
|------|------|
| 最佳适应度 | 0.205 |
| 平均复杂度 | 4.8-5.3 |
| 存活率 | 21-27% |

### Recommended Environment Parameters

```python
# 经演化验证的平衡参数
base_metabolism = 0.006~0.025
neural_cost = 0.0003~0.0025
food_energy = 35~80
food_count = 12~35
global_energy_budget = 1200~3500
predation_rate = 0.4~1.3
predation_range = 2~7
winter_multiplier = 0.08~0.25
season_length = 600~2500
```

### Files

- 主脚本: `scripts/run_v18_coevolution.py`
- 输出: `outputs/v18_coevolution/history.json`

---

*Technical Report generated for EOE Project*
*南京大学人工智能学院*