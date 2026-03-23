# 前馈断层 (Feedforward Disconnect)

## 严重程度: 🔴 系统性天花板

## 发现日期
2026-03-20

## 问题描述

演化保存的拓扑结构和系统默认brain都存在 **sensor到actuator的连接断裂**，导致：
1. Hebbian学习的梯度消失 (pre × post ≈ 0)
2. Agent无法后天学会任何行为
3. 基准测试全部失败

## 根因分析

### 结构问题

**默认Brain结构**:
```
SENSOR(0,1) → PREDICTOR(4,5) → ACTUATOR(2,3)
                    ↓
             PREDICTOR激活=0 ❌
```

**v15保存的结构**:
```
类似问题：感知节点无法有效驱动执行器
```

### 演化环境缺陷

旧演化环境 (`v15-v16`) 的问题：
1. **无后天学习机制**: 没有Hebbian学习，权重不演化
2. **能量获取过于容易**: 不动也能活，演化没有压力
3. **奖励信号稀疏**: Agent无需感知-运动协调就能生存

结果：演化出"看似复杂（节点多）但实际断路"的畸形结构

## 症状表现

1. **Hebbian学习无效**: 权重变化 ≈ 0
2. **节点激活异常**: PREDICTOR节点激活值 ≈ 0
3. **基准测试失败**: Level 1-3 全部超时
4. **Agent行为**: 几乎不动或随机游走

## 验证代码

```python
from core.eoe.agent import Agent

agent = Agent(agent_id=0, x=50, y=50)

# 检查边连接
print('边 (前馈路径):')
for e in agent.genome.edges:
    src_type = agent.genome.nodes[e['source_id']].node_type.name
    dst_type = agent.genome.nodes[e['target_id']].node_type.name
    print(f'  {e["source_id"]}({src_type}) -> {e["target_id"]}({dst_type})')

# 输出示例:
# 5(PREDICTOR) -> 2(ACTUATOR)  # 没有 SENSOR -> ACTUATOR 直接路径！
# 4(PREDICTOR) -> 2(ACTUATOR)
```

## 解决方案

### 短期: 建立基线
1. 创建 sensor→actuator 直连的简单brain
2. 验证Hebbian学习机制有效
3. 证明基准测试系统无Bug

### 长期: 新演化引擎
1. **开启一生学习**: Hebbian权重更新在个体生命周期内
2. **保留拓扑变异**: 只遗传结构，不遗传权重
3. **演化目标**: 寻找"可学习性(Learnability)"强的拓扑

预期效果: 演化会主动选择"sensor与运动节点有丰富潜在路径"的结构

## 影响范围

- ❌ 所有v15保存的complexity_step*.json
- ❌ 所有v16保存的complexity_step*.json  
- ❌ 默认Agent brain
- ✅ 需要新演化生成的结构

## 关联文件

- `benchmarks/benchmark_runner.py` - 热身期实现
- `core/eoe/genome.py` - Reward调制Hebbian
- `core/eoe/environment.py` - 能量追踪

## 状态

🟡 待解决 - 需要新演化实验验证