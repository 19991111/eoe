"""
EOE v9.3 Core Module (Backward Compatibility)
=============================================

This module provides backward compatibility by importing from 
the new modular structure in eoe/ package.

For new code, use:
    from eoe import Population, Agent, Environment
    
For legacy code, use:
    from core import Population, Agent, Environment
"""

# Re-export from new package
from eoe.node import Node, NodeType, SuperNode, SubgraphTracker, MacroOperator
from eoe.genome import OperatorGenome
from eoe.agent import Agent
from eoe.environment import Environment, ChunkManager
from eoe.population import (
    Population, 
    SubgraphFreezer, 
    SpontaneousActivity, 
    MultiLevelPredictor
)

# Version info
__version__ = "9.3"
__all__ = [
    # Node types
    "NodeType",
    "Node", 
    "SuperNode",
    "SubgraphTracker",
    "MacroOperator",
    # Genome
    "OperatorGenome",
    # Agent & Environment
    "Agent",
    "Environment", 
    "ChunkManager",
    # Population
    "Population",
    "SubgraphFreezer",
    "SpontaneousActivity",
    "MultiLevelPredictor",
]