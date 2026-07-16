"""
resident_agent.py —— 住户客服 Agent
===================================
权限：公告 RAG 语义搜索（只读）+ 查询本户物业费（跨 Agent 通信）
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory

from src.DeepSeek_v4_pro import deepseek_v4_pro
import src.property_db as db

_llm = deepseek_v4_pro

# 住户上下文（登录时由 app.py 注入）
_resident_context: dict = {}


def set_resident_context(room_number: str, owner_name: str, password: str):
    """设置当前住户身份上下文（登录成功后由 app.py 调用）"""
    _resident_context["room_number"] = room_number
    _resident_context["owner_name"] = owner_name
    _resident_context["password"] = password


# ============================================================
# 工厂函数
# ============================================================

def build_resident_agent() -> AgentExecutor:
    """构建住户客服 Agent（只读公告 + 可查询自己的物业费）"""

    # --- 工具 1: 语义搜索公告（RAG）---
    @tool
    def search_announcements(query: str) -> str:
        """
        通过语义搜索（RAG）查找相关物业公告。不需要精确关键词，用自然语言提问即可。

        例如：
        - "最近有什么安全方面的通知？" 能匹配到"电梯维护公告"
        - "什么时候要交钱？" 能匹配到"物业费缴纳提醒"
        - "小区近期有什么活动？" 能匹配到"社区夏日活动通知"

        query: 自然语言查询（越具体越好）
        """
        rows = db.search_announcements_rag(query)
        if not rows:
            # RAG 无结果时回退到 SQL Like 搜索
            rows_like = db.search_announcements(query)
            if not rows_like:
                return f"未找到与'{query}'相关的公告。"
            rows = [{"title": r["title"], "content": r["content"],
                     "publish_date": r["publish_date"], "author": r["author"],
                     "id": r["id"], "score": 0} for r in rows_like]

        lines = []
        for r in rows:
            score_info = f" [相关度: {r['score']}]" if r.get("score") else ""
            lines.append(
                f"[ID:{r['id']}] {r['title']}{score_info}\n"
                f"  发布日期: {r['publish_date']} | 作者: {r['author']}\n"
                f"  内容: {r['content']}"
            )
        return "\n\n".join(lines)

    # --- 工具 2: 查询自己的物业费（→ AdminAgent 跨 Agent 通信）---
    @tool
    def query_my_payment() -> str:
        """
        查询当前住户的物业费缴纳情况。
        此工具会自动使用登录时的门牌号和密码向管理员系统发起查询请求。

        返回: 该住户所有物业费记录的原始数据（JSON格式），由客服Agent进行语言加工后输出。
        """
        room = _resident_context.get("room_number", "")
        pwd = _resident_context.get("password", "")

        # 验证身份
        resident = db.authenticate(room, pwd)
        if not resident:
            return "[AdminAgent] 身份验证失败，无法查询物业费。"

        # 向"管理员系统"查询（模拟 Agent 间通信）
        rows = db.get_payment_by_room(room)
        if not rows:
            return f"[AdminAgent] 门牌号 {room} 暂无物业费记录。"

        import json
        return json.dumps(rows, ensure_ascii=False, indent=2)

    tools = [search_announcements, query_my_payment]

    owner = _resident_context.get("owner_name", "住户")

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
你是小区的物业客服助手，正在为住户 **{owner}**（门牌号 {_resident_context.get('room_number', '')}）服务。

你可以使用以下工具：
- 公告查询（search_announcements）：采用 RAG 语义搜索，不需要精确关键词。
  住户用自然语言描述需求（如"最近有什么安全方面的消息？"），工具会自动匹配语义最相关的公告。
- 物业费查询（query_my_payment）：查询当前住户自己的物业费缴纳情况。

重要规则：
1. query_my_payment 返回的是原始JSON数据，你需要将其翻译成住户能看懂的自然语言。
   - "paid" → "已缴纳"
   - "unpaid" → "未缴纳"
   - "overdue" → "已逾期"
   - 按记录逐一说明：哪期、多少钱、截止日期、缴纳状态
2. 你只能查询当前住户自己的数据，不能查询其他住户。
3. 语气亲切、有礼貌，像真正的物业客服一样。
4. 住户问公告时，用 search_announcements 工具并将用户的自然语言原意作为参数传入。
5. 不要编造信息。
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = create_tool_calling_agent(_llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,
                         max_iterations=5, handle_parsing_errors=True)
