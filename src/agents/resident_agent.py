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
from src.DeepSeek_r1_7b import deepseek_r1
from src.resilience import safe_chroma_search
import src.property_db as db

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

def build_resident_agent(llm=None, memory=None) -> AgentExecutor:
    """构建住户客服 Agent（只读公告 + 可查询自己的物业费）。

    llm: 可选，传入自定义 LLM 实例（默认 deepseek_v4_pro），用于降级切换本地模型
    memory: 可选，传入已有 ConversationBufferMemory，用于降级时继承对话历史
    """
    _llm = llm or deepseek_v4_pro

    # --- 工具 1: 语义搜索公告（RAG）---
    @tool
    def search_announcements(query: str) -> str:
        """
        通过语义搜索（RAG）查找相关物业公告。不需要精确关键词，用自然语言提问即可。
        Chroma 不可用时自动降级为关键词搜索。

        query: 自然语言查询（越具体越好）
        """
        # 使用容错搜索：Chroma 失败时自动降级 SQL LIKE
        rows = safe_chroma_search(
            db.search_announcements_rag,
            db.search_announcements,
            query, k=4
        )
        if not rows:
            return f"未找到与'{query}'相关的公告。"

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

    # --- 工具 3: 提交报修 ---
    @tool
    def submit_repair(title: str, description: str) -> str:
        """
        提交一条报修工单。

        title: 报修标题（如"水管漏水"、"空调不制冷"）
        description: 详细描述（如位置、现象、紧急程度等）
        """
        room = _resident_context.get("room_number", "")
        rid = db.add_repair(room, title, description)
        return (
            f"报修工单已提交成功！工单ID={rid}\n"
            f"标题: {title}\n"
            f"我们已收到您的报修，物业工作人员将尽快处理。"
        )

    # --- 工具 4: 查询我的工单 ---
    @tool
    def query_my_repairs() -> str:
        """查询当前住户提交的所有报修工单及处理状态"""
        room = _resident_context.get("room_number", "")
        repairs = db.get_repairs_by_room(room)
        if not repairs:
            return "您当前没有报修工单。"
        status_label = {"pending": "待处理", "processing": "处理中",
                       "completed": "已完成", "cancelled": "已取消"}
        lines = []
        for r in repairs:
            lines.append(
                f"[ID:{r['id']}] {r['title']} | 状态:{status_label.get(r['status'], r['status'])} "
                f"| 提交:{r['created_at']}\n"
                f"  描述: {r['description']}"
                f"{' | 物业回复:' + r['admin_note'] if r.get('admin_note') else ''}"
            )
        return "\n\n".join(lines)

    tools = [search_announcements, query_my_payment, submit_repair, query_my_repairs]

    owner = _resident_context.get("owner_name", "住户")

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
你是小区的物业客服助手，正在为住户 **{owner}**（门牌号 {_resident_context.get('room_number', '')}）服务。

你可以使用以下工具：
- 公告查询（search_announcements）：采用 RAG 语义搜索，不需要精确关键词。
  住户用自然语言描述需求，工具会自动匹配语义最相关的公告。
- 物业费查询（query_my_payment）：查询当前住户自己的物业费缴纳情况。
- 报修提交（submit_repair）：提交报修工单，需要提供标题和详细描述。
- 我的工单（query_my_repairs）：查看已提交的报修工单及处理状态。

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

【情感感知与标记规则】
6. 识别用户消息中的情绪，根据不同情绪调整回应策略：
   - 负面情绪（愤怒/沮丧/不满）：先表达理解和歉意（如"非常理解您的心情""很抱歉给您带来不便"），
     再提供解决方案。语气要格外温和，避免官腔。
   - 紧急情绪（关键词含"漏水""漏电""着火""煤气""电梯坠落""停电""停水"等）：
     首先安抚"请保持冷静"，然后建议住户立即拨打物业24小时值班电话 400-888-9999，
     同时引导其通过 submit_repair 提交紧急工单。
   - 正面情绪（感谢/表扬/满意）：真诚感谢户主的认可，表示会继续努力。
   - 中性情绪：保持标准客服语气，友好热情。
7. 不要在回复中直接说出情绪分析结果（如"我检测到你很愤怒"），而是自然地运用上述策略。
8. 对于反复表达不满的住户，主动表示"我会将您的情况反馈给物业经理，优先为您处理"。
9. 在你的回复末尾，必须添加一行情感标签。根据用户本轮消息的原始情绪，选择 [sentiment:正面]、
   [sentiment:中性] 或 [sentiment:负面] 之一附加在回复最后。这行标签仅供系统后台使用，不要向用户解释它的含义。

【信息熵反问机制】
10. 在调用任何工具之前，先评估用户Query的信息完整度（信息熵）：
    - 指代不明（如"那个""这事""之前那个"）：无法确定具体指向什么
    - 意图缺失（如"帮我查一下""帮我看看"）：未说明要查什么
    - 条件不足（如"我要报修"但未提供标题和描述）：工具需要参数但用户未提供
    - 时间模糊（如"最近的""之前的"）：时间范围不明确
    - 复合意图（如"缴费还有那个公告"）：多个意图杂糅，无法确定优先级
11. 当信息熵过高时，遵循以下策略：
    a. ⛔ 禁止调用任何工具 —— 此时调用工具必然得到不相关结果，浪费资源
    b. 不要给出笼统回复（如"请问有什么可以帮您？"），而是针对缺失信息提出具体的引导性反问：
       - 列出可能选项让用户选择："您是想查公告通知、物业费缴纳情况，还是报修进度呢？"
       - 追问关键参数："请问大概是哪方面的公告？比如停水停电、物业费、还是社区活动？"
       - 确认指代："您是指上次查询的物业费，还是最近的报修工单呢？"
    c. 可结合对话历史判断 —— 如果上一轮用户刚问过物业费，本轮"那上个月的呢？"熵值低，直接查询
    d. 反问数量控制在1-2个，避免连环追问
12. 当触发反问（未调用任何工具）时，在回复末尾附加 [entropy:high] 标签；
    正常回答时附加 [entropy:low] 标签。情感标签和熵标签以空格分隔。
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    if memory is None:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = create_tool_calling_agent(_llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,
                         max_iterations=5, handle_parsing_errors=True)
