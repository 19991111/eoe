"""
EOE - Embodied Operator Evolution
===============================

A neuroevolution system where mathematical operators evolve into 
complex brain structures in a 2D chemotaxis environment.

Modules:
- node: Node, NodeType, SuperNode
- genome: OperatorGenome
- agent: Agent
- environment: Environment, ChunkManager
- population: Population
- modules: SubgraphFreezer, SpontaneousActivity, MultiLevelPredictor
"""

from .node import Node, NodeType, SuperNode
from .genome import OperatorGenome
from .agent import Agent
from .environment import Environment, ChunkManager
from .population import Population

__version__ = "9.3"
__all__ = [
    "Node",
    "NodeType", 
    "SuperNode",
    "OperatorGenome",
    "Agent",
    "Environment",
    "ChunkManager",
    "Population",
]