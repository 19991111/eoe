"""
EOE - Embodied Operator Evolution
===============================

A neuroevolution system where mathematical operators evolve into 
complex brain structures in a 2D chemotaxis environment.

Modules:
- node: Node, NodeType, SuperNode
- genome: OperatorGenome
- agent: Agent
- environment: Environment (v12), ChunkManager
- population: Population
- manifest: PhysicsManifest (SSOT)
- brain_manager: BrainManager
"""

from .node import Node, NodeType, SuperNode
from .genome import OperatorGenome
from .agent import Agent
from .environment import Environment
from .population import Population

__version__ = "12.6"
__all__ = [
    "Node",
    "NodeType", 
    "SuperNode",
    "OperatorGenome",
    "Agent",
    "Environment",
    "Population",
]