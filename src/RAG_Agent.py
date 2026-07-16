"""
RAG_Agent.py —— 命令行测试入口
==============================
用法：
    python RAG_Agent.py

这是 Agent 的终端测试脚本，用于在命令行下快速验证 Agent 功能。
启动 Web 服务请运行 app.py 然后访问 http://localhost:8000
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent_builder import build_agent
