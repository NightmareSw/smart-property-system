"""
app.py —— FastAPI 后端服务
==========================
职责：
1. 启动时加载 agent_builder 模块（触发 Embedding 模型和向量库的全局初始化）
2. 提供 POST /chat 接口：接收用户消息，返回 Agent 回复
3. 管理多会话：每个浏览器标签页拥有独立的对话记忆
4. 托管前端页面（test.html）

启动方式：
    python app.py
    或
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

访问方式：
    浏览器打开 http://localhost:8000
"""

import os
import sys
import time

# ---------- 确保 src/ 模块可以被导入 ----------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor

# ---------- 导入 Agent 工厂函数 ----------
from src.agent_builder import build_agent

# ============================================================
# FastAPI 应用实例
# ============================================================
app = FastAPI(
    title="智能HR助手 API",
    description="基于 RAG + Agent 的智能 HR 后端服务",
    version="1.0.0"
)

# ============================================================
# CORS 中间件 —— 允许浏览器跨域请求
# ============================================================
# 如果没有这个，浏览器会因为"同源策略"拦截 fetch 请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # 开发阶段允许所有来源，生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],              # 允许所有 HTTP 方法
    allow_headers=["*"],              # 允许所有请求头
)

# ============================================================
# 数据模型（Pydantic）—— 定义请求和响应的数据结构
# ============================================================

class ChatRequest(BaseModel):
    """客户端发送的聊天请求"""
    input: str = Field(..., description="用户输入的消息")
    session_id: str = Field(default="default", description="会话ID，同个ID共享对话记忆")


class ChatResponse(BaseModel):
    """服务端返回的聊天响应"""
    output: str = Field(..., description="Agent 的回复内容")
    session_id: str = Field(default="default", description="当前会话ID")
    response_time: float = Field(default=0.0, description="处理耗时（秒）")

# ============================================================
# 会话管理 —— 每个 session_id 对应一个独立的 AgentExecutor
# ============================================================
# 用字典存储： key=session_id, value=AgentExecutor
# AgentExecutor 内部绑定了该会话独立的 ConversationBufferMemory
_sessions: dict[str, AgentExecutor] = {}

def get_or_create_session(session_id: str) -> AgentExecutor:
    """
    获取或创建一个会话。

    如果 session_id 已经存在，直接返回对应的 executor（保留之前的对话记忆）。
    如果 session_id 是新的，调用 build_agent() 创建一个全新的 executor。
    """
    if session_id not in _sessions:
        print(f"[app] 创建新会话: {session_id}")
        _sessions[session_id] = build_agent()
    return _sessions[session_id]


# ============================================================
# API 路由
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    首页 —— 返回前端聊天页面（test.html）。
    浏览器访问 http://localhost:8000 就能看到聊天界面。
    """
    html_path = os.path.join(os.path.dirname(__file__), "..", "test.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h2>前端页面 test.html 未找到</h2>", status_code=404)
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口 —— Agent 的核心入口。

    接收用户消息，交给 Agent 处理，返回回复。
    同个 session_id 的多轮对话会保留上下文记忆。

    请求示例:
        POST /chat
        {"input": "王博的教育背景是什么?", "session_id": "user_001"}

    响应示例:
        {"output": "王博毕业于...", "session_id": "user_001", "response_time": 2.34}
    """
    # 1. 获取或创建该会话的 executor
    executor = get_or_create_session(request.session_id)

    # 2. 调用 Agent，记录耗时
    start_time = time.time()
    try:
        result = executor.invoke({"input": request.input})
        output = result.get("output", "（Agent 未返回有效内容）")
    except Exception as e:
        # Agent 调用失败时返回错误提示，不会让整个服务崩溃
        print(f"[app] Agent 调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"Agent 处理失败: {str(e)}")

    elapsed = round(time.time() - start_time, 2)

    # 3. 返回结果
    print(f"[app] 会话={request.session_id} | 耗时={elapsed}s | 输入={request.input[:30]}...")
    return ChatResponse(
        output=output,
        session_id=request.session_id,
        response_time=elapsed
    )


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """
    清除指定会话 —— 清空对话记忆，下次请求会重新创建。
    用于"开始新对话"功能。

    用法: DELETE /sessions/user_001
    """
    if session_id in _sessions:
        del _sessions[session_id]
        print(f"[app] 已清除会话: {session_id}")
        return {"message": f"会话 {session_id} 已清除"}
    return {"message": f"会话 {session_id} 不存在"}


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  智能HR助手 后端服务启动中...")
    print("  浏览器打开: http://localhost:8080")
    print("  API 文档:   http://localhost:8080/docs")
    print("=" * 50)
    # host="127.0.0.1" 仅本机访问；改成 "0.0.0.0" 可让局域网其他设备访问
    # 端口 8080 需与 test.html 中 fetch 的端口一致
    uvicorn.run(app, host="127.0.0.1", port=8080)
