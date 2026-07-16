"""
admin_agent.py —— 管理员 Agent
==============================
权限：公告 CRUD + 物业费 CRUD（全量访问）
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory

from src.DeepSeek_v4_pro import deepseek_v4_pro
from src.resilience import llm_breaker, retry_db
import src.property_db as db

_llm = deepseek_v4_pro


# ============================================================
# 工具定义
# ============================================================

@tool
def admin_list_announcements() -> str:
    """查看所有物业公告的列表（含完整内容）"""
    rows = db.get_all_announcements()
    if not rows:
        return "暂无公告。"
    lines = []
    for r in rows:
        lines.append(
            f"[ID:{r['id']}] {r['title']}\n"
            f"  发布日期: {r['publish_date']} | 作者: {r['author']}\n"
            f"  内容: {r['content']}"
        )
    return "\n\n".join(lines)


@tool
def admin_add_announcement(title: str, content: str) -> str:
    """新增一条物业公告。title: 公告标题; content: 公告正文内容"""
    new_id = db.add_announcement(title, content)
    return f"公告新增成功，ID={new_id}"


@tool
def admin_update_announcement(ann_id: int, title: str, content: str) -> str:
    """修改指定 ID 的公告标题和内容。ann_id: 公告ID; title: 新标题; content: 新内容"""
    ok = db.update_announcement(ann_id, title, content)
    return f"公告 ID={ann_id} 更新成功" if ok else f"公告 ID={ann_id} 不存在"


@tool
def admin_delete_announcement(ann_id: int) -> str:
    """删除指定 ID 的公告。ann_id: 公告ID"""
    ok = db.delete_announcement(ann_id)
    return f"公告 ID={ann_id} 已删除" if ok else f"公告 ID={ann_id} 不存在"


@tool
def admin_list_all_payments() -> str:
    """查看全部住户的物业费缴纳记录"""
    rows = db.get_all_payments()
    if not rows:
        return "暂无缴纳记录。"
    lines = []
    for r in rows:
        lines.append(
            f"[ID:{r['id']}] 门牌号:{r['room_number']} | 金额:{r['amount']}元 "
            f"| 截止:{r['due_date']} | 状态:{r['status']}"
            f"{' | 缴纳日期:' + r['paid_date'] if r['paid_date'] else ''}"
        )
    return "\n".join(lines)


@tool
def admin_add_payment(room_number: str, amount: float, due_date: str) -> str:
    """新增一条物业费缴纳记录。room_number: 门牌号(如101); amount: 金额(元); due_date: 截止日期(如2026-09-30)"""
    new_id = db.add_payment(room_number, amount, due_date)
    return f"物业费记录新增成功，ID={new_id}"


@tool
def admin_mark_paid(pay_id: int) -> str:
    """将指定 ID 的物业费标记为'已缴纳'。pay_id: 缴费记录ID"""
    ok = db.update_payment_status(pay_id, "paid")
    return f"记录 ID={pay_id} 已标记为'已缴纳'" if ok else f"记录 ID={pay_id} 不存在"


@tool
def admin_delete_payment(pay_id: int) -> str:
    """删除指定 ID 的物业费记录。pay_id: 缴费记录ID"""
    ok = db.delete_payment(pay_id)
    return f"记录 ID={pay_id} 已删除" if ok else f"记录 ID={pay_id} 不存在"


# ============================================================
# 业主管理工具（管理员专用）
# ============================================================

@tool
def admin_list_owners() -> str:
    """查看所有业主/住户完整信息（含门牌号、姓名、电话、密码）"""
    owners = db.get_all_owners()
    if not owners:
        return "暂无业主信息。"
    lines = []
    for o in owners:
        lines.append(
            f"门牌号:{o['room_number']} | 姓名:{o['owner_name']} "
            f"| 电话:{o.get('phone', '')} | 密码:{o.get('password', '')}"
        )
    return "\n".join(lines)


@tool
def admin_add_owner(room_number: str, owner_name: str, phone: str, password: str) -> str:
    """新增一位业主/住户。room_number: 门牌号(如101); owner_name: 姓名; phone: 电话; password: 登录密码"""
    ok = db.add_owner(room_number, password, owner_name, phone)
    return f"业主 {owner_name}({room_number}室) 新增成功" if ok else f"门牌号 {room_number} 已存在，请勿重复添加"


@tool
def admin_update_owner(room_number: str, owner_name: str, phone: str, password: str) -> str:
    """修改业主/住户信息。room_number: 门牌号; owner_name: 新姓名; phone: 新电话; password: 新密码"""
    ok = db.update_owner(room_number, owner_name=owner_name, phone=phone, password=password)
    return f"业主 {room_number}室 信息已更新" if ok else f"门牌号 {room_number} 不存在"


@tool
def admin_delete_owner(room_number: str) -> str:
    """删除指定门牌号的业主/住户。room_number: 门牌号(如101)"""
    ok = db.delete_owner(room_number)
    return f"业主 {room_number}室 已删除" if ok else f"门牌号 {room_number} 不存在"


# ============================================================
# 公告生成与发布工具
# ============================================================

@tool
def admin_generate_announcement(topic: str, details: str) -> str:
    """
    根据主题和要点生成一篇语言精炼、格式规范的物业公告草稿。
    生成后需要向管理员确认，管理员同意后才能正式发布。

    topic: 公告主题（如"停水通知"、"电梯维护"、"社区活动"）
    details: 公告要点（如时间、地点、原因、影响范围等，越详细越好）
    """
    prompt = f"""请根据以下信息生成一篇正式的物业公告。

要求：
1. 语言精炼、规范、得体，语气正式但不生硬
2. 格式清晰，包含标题、正文、发布日期
3. 结尾统一使用"xxx物业管理中心"
4. 不超过200字

主题：{topic}
要点：{details}

请直接输出公告全文，不要加额外说明："""

    try:
        from langchain_core.messages import HumanMessage

        def _call_llm():
            return _llm.invoke([HumanMessage(content=prompt)])

        def _fallback():
            from langchain_core.messages import AIMessage
            return AIMessage(content=f"【{topic}】\n\n{details}\n\nxxx物业管理中心")

        response = llm_breaker.call(_call_llm, fallback=_fallback)
        announcement = response.content.strip()
        return (
            f"=== 公告草稿 ===\n\n{announcement}\n\n"
            f"---\n【重要】以上为草稿，请先向管理员展示并等待明确确认。"
            f"管理员回复「确认/可以/发布」后才可调用 admin_publish_announcement。"
            f"如需修改，按管理员要求调整后重新展示。"
        )
    except Exception as e:
        return f"公告生成失败: {str(e)}。请重试或手动编写公告。"


@tool
def admin_publish_announcement(title: str, content: str) -> str:
    """
    将确认后的公告正式发布：写入 SQLite 数据库 + Chroma 向量库 + ADVER 文件夹。

    title: 公告标题
    content: 公告正文（需包含"xxx物业管理中心"结尾）
    """
    import os
    # 生成安全的文件名
    safe_title = title.replace("/", "-").replace("\\", "-").replace(":", "：")
    filename = f"{safe_title}.txt"
    # 项目根目录 = src/agents/ 的上两级
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    adver_dir = os.path.join(project_root, "ADVER")
    os.makedirs(adver_dir, exist_ok=True)

    # 写入 ADVER 文件夹
    filepath = os.path.join(adver_dir, filename)
    full_text = f"{title}\n{content}"
    for encoding in ["utf-8", "gbk"]:
        try:
            with open(filepath, "w", encoding=encoding) as f:
                f.write(full_text)
            break
        except UnicodeEncodeError:
            continue

    # 写入数据库 + Chroma
    new_id = db.add_announcement(title, content, source_file=filename)
    db.add_notification("", "新公告发布", f"物业发布了新公告：{title}", "info")
    return f"公告已发布成功！ID={new_id}，文件已保存至 ADVER/{filename}"


# ============================================================
# 数据库级工具 —— 筛选查询 & 统计
# ============================================================

@tool
def admin_db_stats() -> str:
    """查看数据库整体统计信息：各表记录数、Chroma向量数、物业费各状态数量"""
    stats = db.get_db_stats()
    return (
        f"=== 数据库统计 ===\n"
        f"公告: {stats['announcements']} 条（Chroma 向量: {stats['chroma_vectors']} 条）\n"
        f"业主: {stats['owners']} 户\n"
        f"物业费记录: {stats['payments']} 条 "
        f"（已缴纳:{stats['paid']} / 未缴纳:{stats['unpaid']} / 逾期:{stats['overdue']}）"
    )


@tool
def admin_filter_payments(status: str) -> str:
    """
    按状态筛选物业费缴纳记录。
    status: 状态值，可选 'paid'(已缴纳)、'unpaid'(未缴纳)、'overdue'(逾期)
    """
    rows = db.get_payments_by_status(status)
    status_label = {"paid": "已缴纳", "unpaid": "未缴纳", "overdue": "逾期"}.get(status, status)
    if not rows:
        return f"暂无状态为'{status_label}'的物业费记录。"
    lines = []
    for r in rows:
        lines.append(
            f"[ID:{r['id']}] 门牌号:{r['room_number']} | 金额:{r['amount']}元 "
            f"| 截止:{r['due_date']}"
            f"{' | 缴纳日期:' + r['paid_date'] if r.get('paid_date') else ''}"
        )
    return f"=== {status_label}记录 ({len(rows)}条) ===\n" + "\n".join(lines)


@tool
def admin_query_payment_by_room(room_number: str) -> str:
    """
    查询指定门牌号的物业费缴纳记录。
    room_number: 门牌号(如101)
    """
    rows = db.get_payment_by_room(room_number)
    if not rows:
        return f"门牌号 {room_number} 暂无物业费记录。"
    lines = [f"=== {room_number}室 物业费记录 ==="]
    for r in rows:
        status_label = {"paid": "已缴纳", "unpaid": "未缴纳", "overdue": "逾期"}.get(
            r.get("status", ""), r.get("status", ""))
        lines.append(
            f"[ID:{r['id']}] 金额:{r['amount']}元 | 截止:{r['due_date']} "
            f"| 状态:{status_label}"
            f"{' | 缴纳日期:' + r['paid_date'] if r.get('paid_date') else ''}"
        )
    return "\n".join(lines)


@tool
def admin_search_announcements(keyword: str) -> str:
    """
    按关键词搜索公告（标题和内容模糊匹配）。
    keyword: 搜索关键词
    """
    rows = db.search_announcements(keyword)
    if not rows:
        return f"未找到包含'{keyword}'的公告。"
    lines = [f"=== 搜索'{keyword}'结果 ({len(rows)}条) ==="]
    for r in rows:
        lines.append(
            f"[ID:{r['id']}] {r['title']}\n"
            f"  发布日期: {r['publish_date']} | 作者: {r['author']}\n"
            f"  内容: {r['content']}"
        )
    return "\n\n".join(lines)


# ============================================================
# 管理员账号管理工具（仅超级管理员可调用）
# ============================================================

@tool
def admin_list_accounts() -> str:
    """查看所有系统管理员账号列表（含用户名、姓名、角色）"""
    admins = db.get_all_admins()
    if not admins:
        return "暂无管理员账号。"
    lines = []
    for a in admins:
        lines.append(f"用户名:{a['username']} | 姓名:{a['name']} | 角色:{a['role']}")
    return "\n".join(lines)


@tool
def admin_add_account(username: str, password: str, name: str, role: str) -> str:
    """新增一个管理员账号。username: 登录用户名; password: 登录密码; name: 显示姓名; role: 角色(admin/superadmin)"""
    ok = db.add_admin(username, password, name, role)
    return f"管理员 {name}({username}) 新增成功" if ok else f"用户名 {username} 已存在"


@tool
def admin_update_account(username: str, password: str, name: str, role: str) -> str:
    """修改管理员账号信息。username: 要修改的用户名; password: 新密码; name: 新姓名; role: 新角色"""
    ok = db.update_admin(username, password=password, name=name, role=role)
    return f"管理员 {username} 信息已更新" if ok else f"用户名 {username} 不存在"


@tool
def admin_delete_account(username: str) -> str:
    """删除指定用户名的管理员账号。username: 要删除的用户名"""
    admins = db.get_all_admins()
    if len(admins) <= 1:
        return "无法删除：系统中仅剩一位管理员"
    ok = db.delete_admin(username)
    return f"管理员 {username} 已删除" if ok else f"用户名 {username} 不存在"


# ============================================================
# 物业费状态管理工具
# ============================================================

@tool
def admin_set_payment_status(pay_id: int, status: str) -> str:
    """
    手动设置物业费缴纳状态。
    pay_id: 物业费记录ID
    status: 目标状态，可选 'paid'(已缴纳)、'unpaid'(未缴纳)
    """
    valid = {"paid": "已缴纳", "unpaid": "未缴纳"}
    if status not in valid:
        return f"无效状态，可选: {', '.join(valid.keys())}"
    ok = db.update_payment_status(pay_id, status,
        "" if status == "unpaid" else None)
    return f"记录 ID={pay_id} 已设为「{valid[status]}」" if ok else f"记录 ID={pay_id} 不存在"


# ============================================================
# 报修工单管理工具
# ============================================================

@tool
def admin_list_repairs() -> str:
    """查看全部报修工单列表（含状态、门牌号、内容）"""
    repairs = db.get_all_repairs()
    if not repairs:
        return "暂无报修工单。"
    status_label = {"pending": "待处理", "processing": "处理中",
                   "completed": "已完成", "cancelled": "已取消"}
    lines = []
    for r in repairs:
        lines.append(
            f"[ID:{r['id']}] {r['title']} | 门牌号:{r['room_number']} "
            f"| 状态:{status_label.get(r['status'], r['status'])} "
            f"| 提交:{r['created_at']}\n"
            f"  描述: {r['description']}"
            f"{' | 备注:' + r['admin_note'] if r.get('admin_note') else ''}"
        )
    return "\n\n".join(lines)


@tool
def admin_update_repair(repair_id: int, status: str, admin_note: str) -> str:
    """
    更新报修工单状态并添加备注。
    repair_id: 工单ID; status: 新状态(pending/processing/completed/cancelled);
    admin_note: 管理员备注（处理说明等）
    """
    ok = db.update_repair(repair_id, status, admin_note)
    status_label = {"pending": "待处理", "processing": "处理中",
                   "completed": "已完成", "cancelled": "已取消"}
    if ok:
        return f"工单 ID={repair_id} 已更新为「{status_label.get(status, status)}」"
    return f"工单 ID={repair_id} 不存在"


# ============================================================
# 工厂函数
# ============================================================

def build_admin_agent() -> AgentExecutor:
    """构建管理员 Agent（全权限 CRUD）"""
    tools = [
        admin_list_announcements,
        admin_add_announcement,
        admin_update_announcement,
        admin_delete_announcement,
        admin_search_announcements,
        admin_generate_announcement,
        admin_publish_announcement,
        admin_list_all_payments,
        admin_add_payment,
        admin_mark_paid,
        admin_delete_payment,
        admin_filter_payments,
        admin_query_payment_by_room,
        admin_list_owners,
        admin_add_owner,
        admin_update_owner,
        admin_delete_owner,
        admin_db_stats,
        admin_list_accounts,
        admin_add_account,
        admin_update_account,
        admin_delete_account,
        admin_list_repairs,
        admin_update_repair,
        admin_set_payment_status,
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
你是物业管理员助手，拥有数据库的全部操作权限。

你可以使用以下工具：
- 公告管理：查看全部、按关键词搜索、AI生成公告草稿、正式发布、新增、修改、删除公告
- 物业费管理：查看全部、按状态筛选、按门牌号查询、新增记录、标记已缴纳、删除记录、手动设置状态(paid/unpaid)
- 业主管理：查看全部、新增、修改、删除业主信息
- 报修工单：查看全部工单、更新工单状态+备注
- 数据库统计：查看系统整体数据概览
- 管理员账号管理：查看全部账号、新增、修改、删除管理员账号

【公告生成与发布流程（重要·必须遵守）】
1. 当管理员要求写公告时，调用 admin_generate_announcement(topic, details)。
2. 工具返回草稿后，你必须将草稿原文展示给管理员，并问「确认发布吗？」。
3. ⛔ **严禁在管理员确认前调用 admin_publish_announcement！** 生成草稿后立即停止本轮工具调用，等待管理员下一轮回复。
4. 管理员明确回复「确认」「可以」「发布」「行」后，你才能调用 admin_publish_announcement(title, content) 正式发布。
5. 管理员要求修改时，口头调整内容后再次展示，仍需等待确认。
6. 发布后汇报结果：公告ID、保存位置。

其他规则：
- 当用户问"未缴纳的有哪些"、"谁逾期了"时，使用 admin_filter_payments 工具。
- 当用户问"101的缴费情况"时，使用 admin_query_payment_by_room 工具。
- 操作完成后简洁汇报结果，先查询最新数据再回答用户问题。
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = create_tool_calling_agent(_llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,
                         max_iterations=5, handle_parsing_errors=True)
