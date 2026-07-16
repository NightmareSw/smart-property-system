"""
agents/ —— 物业多 Agent 模块
=============================
对外暴露两个工厂函数，调用方无需关心内部实现。
"""

from src.agents.admin_agent import build_admin_agent
from src.agents.resident_agent import build_resident_agent, set_resident_context
