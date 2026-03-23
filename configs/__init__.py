# EOE 配置模块
"""
分层配置系统:
- base: 基础默认值
- presets: 预设配置
- utils: 配置工具
"""

from .base import PoolConfig, AgentState

__all__ = ['PoolConfig', 'AgentState']