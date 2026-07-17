"""
property_app.py —— 物业多 Agent 系统服务端
=========================================
路由设计：
  GET  /                  → 登录页面
  POST /login             → 身份认证，返回角色 + token
  POST /chat/admin        → 管理员 Agent 对话接口
  POST /chat/resident     → 住户客服 Agent 对话接口

多 Agent 协作流程：
  住户询问"我的物业费" → ResidentAgent 调用 query_my_payment 工具
  → 工具内部验证身份后调用 property_db.get_payment_by_room()
  → 返回原始数据 → ResidentAgent 的 LLM 加工为自然语言 → 输出给用户

启动方式：
  python property_app.py  → 浏览器打开 http://localhost:9090
"""

import os
import sys
import time
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import shutil
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain.agents import AgentExecutor

import src.property_db as db
from src.agents import build_admin_agent, build_resident_agent, set_resident_context
from src.resilience import format_error, CircuitBreakerOpenError
from src.DeepSeek_r1_7b import deepseek_r1 as _fallback_llm

app = FastAPI(title="物业多Agent系统", version="2.0.0")


# ============================================================
# 全局异常中间件
# ============================================================

@app.middleware("http")
async def error_middleware(request, call_next):
    """捕获所有未处理异常，返回统一格式"""
    try:
        return await call_next(request)
    except Exception as exc:
        err = format_error(exc)
        from fastapi.responses import JSONResponse
        status = 503 if err["error"] == "service_unavailable" else 500
        return JSONResponse(status_code=status, content={"detail": err["message"]})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 会话管理
# ============================================================

# token → AgentExecutor
_admin_sessions: dict[str, AgentExecutor] = {}
_resident_sessions: dict[str, AgentExecutor] = {}

# token → 住户上下文（用于 ResidentAgent 工具中的身份绑定）
_resident_contexts: dict[str, dict] = {}


# ============================================================
# 请求/响应模型
# ============================================================

class LoginRequest(BaseModel):
    role: str             # "admin" 或 "resident"
    # 管理员字段
    username: str = ""
    password: str = ""
    # 住户字段
    room_number: str = ""
    resident_password: str = ""


class ChatRequest(BaseModel):
    token: str
    input: str


class ChatResponse(BaseModel):
    output: str
    role: str
    response_time: float


class OwnerRequest(BaseModel):
    room_number: str
    owner_name: str
    phone: str = ""
    password: str


class OwnerUpdateRequest(BaseModel):
    owner_name: str = None
    phone: str = None
    password: str = None


# 路径常量
ADVER_DIR = str(Path(__file__).parent.parent / "ADVER")


# ============================================================
# 页面路由
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def login_page():
    html_path = os.path.join(os.path.dirname(__file__), "..", "property_login.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ============================================================
# 认证接口
# ============================================================

@app.post("/login")
async def login(req: LoginRequest):
    """
    统一登录入口。
    根据 role 字段分流：
    - admin: 验证 username + password
    - resident: 验证 room_number + resident_password
    成功后返回 token，前端凭 token 调用对应的 chat 接口。
    """
    token = str(uuid.uuid4())

    if req.role == "admin":
        admin_info = db.authenticate_admin(req.username, req.password)
        if admin_info:
            _admin_sessions[token] = build_admin_agent()
            return {
                "success": True,
                "token": token,
                "role": "admin",
                "display_name": admin_info.get("name", "管理员"),
                "admin_role": admin_info.get("role", "admin"),
            }
        raise HTTPException(401, detail="管理员账号或密码错误")

    elif req.role == "resident":
        resident = db.authenticate(req.room_number, req.resident_password)
        if resident:
            # 注入住户上下文（ResidentAgent 的工具会用到）
            set_resident_context(
                room_number=resident["room_number"],
                owner_name=resident["owner_name"],
                password=req.resident_password,
            )
            _resident_sessions[token] = build_resident_agent()
            # 保存明文密码供后续 chat 请求刷新上下文使用
            _resident_contexts[token] = {
                **resident,
                "password": req.resident_password,
            }
            return {
                "success": True,
                "token": token,
                "role": "resident",
                "display_name": f"{resident['owner_name']}({resident['room_number']}室)"
            }
        raise HTTPException(401, detail="门牌号或密码错误")

    raise HTTPException(400, detail="无效的角色类型")


# ============================================================
# 对话接口
# ============================================================

@app.post("/chat/admin", response_model=ChatResponse)
async def chat_admin(req: ChatRequest):
    """管理员 Agent 对话接口（本地模型降级）"""
    executor = _admin_sessions.get(req.token)
    if not executor:
        raise HTTPException(401, detail="未登录或会话已过期")

    start = time.time()
    try:
        result = await asyncio.to_thread(executor.invoke, {"input": req.input})
        output = result.get("output", "（无响应）")
    except Exception as e:
        print(f"[admin_chat] 云端调用失败: {e}，尝试降级到本地模型")
        try:
            fallback_executor = build_admin_agent(
                llm=_fallback_llm, memory=executor.memory)
            result = await asyncio.to_thread(
                fallback_executor.invoke, {"input": req.input})
            output = result.get("output", "")
            output = (
                "⚠️ 云端 AI 服务暂时不可达，已切换至本地备用模型，响应速度和精度可能略有下降。\n\n"
                + output
            )
            _admin_sessions[req.token] = fallback_executor
        except Exception as e2:
            print(f"[admin_chat] 本地降级也失败: {e2}")
            raise HTTPException(500, detail="AI 服务暂时不可用，请稍后重试")

    return ChatResponse(
        output=output,
        role="admin",
        response_time=round(time.time() - start, 2),
    )


@app.post("/chat/resident", response_model=ChatResponse)
async def chat_resident(req: ChatRequest):
    """住户客服 Agent 对话接口（LLM 情感标签 + 本地模型降级）"""
    executor = _resident_sessions.get(req.token)
    if not executor:
        raise HTTPException(401, detail="未登录或会话已过期")

    ctx = _resident_contexts.get(req.token, {})
    room = ctx.get("room_number", "")
    set_resident_context(
        room_number=room,
        owner_name=ctx.get("owner_name", ""),
        password=ctx.get("password", ""),
    )

    start = time.time()
    try:
        result = await asyncio.to_thread(executor.invoke, {"input": req.input})
        raw_output = result.get("output", "（无响应）")
    except Exception as e:
        print(f"[resident_chat] 云端调用失败: {e}，尝试降级到本地模型")
        try:
            # 重建 Agent，注入本地 R1 模型，继承对话记忆
            fallback_executor = build_resident_agent(
                llm=_fallback_llm, memory=executor.memory)
            result = await asyncio.to_thread(
                fallback_executor.invoke, {"input": req.input})
            raw_output = result.get("output", "")
            raw_output = (
                "⚠️ 云端 AI 服务暂时不可达，已切换至本地备用模型，响应速度和精度可能略有下降。\n\n"
                + raw_output
            )
            _resident_sessions[req.token] = fallback_executor
        except Exception as e2:
            print(f"[resident_chat] 本地降级也失败: {e2}")
            raise HTTPException(500, detail="AI 服务暂时不可用，请稍后重试")

    # 解析 LLM 标签: [sentiment:xxx] [entropy:high/low]
    import re
    sentiment_label = "neutral"
    sentiment_score = 0.5
    entropy_high = False

    # 情感标签
    match = re.search(r'\[sentiment:(正面|中性|负面)\]', raw_output)
    if match:
        label_cn = match.group(1)
        sentiment_label = {"正面": "positive", "中性": "neutral", "负面": "negative"}[label_cn]
        sentiment_score = {"positive": 0.85, "neutral": 0.5, "negative": 0.15}[sentiment_label]

    # 信息熵标签
    if re.search(r'\[entropy:high\]', raw_output):
        entropy_high = True

    # 剥离所有系统标签
    output = re.sub(r'\s*\[sentiment:(?:正面|中性|负面)\]\s*', '', raw_output)
    output = re.sub(r'\s*\[entropy:(?:high|low)\]\s*', '', output).rstrip()

    if not output:
        output = raw_output  # 防止全部剥离后为空

    # 日志写入
    try:
        db.add_sentiment_log(room, req.input[:200], sentiment_score, sentiment_label,
                           entropy_high=entropy_high)
    except Exception:
        pass

    return ChatResponse(
        output=output,
        role="resident",
        response_time=round(time.time() - start, 2),
    )


# ============================================================
# 住户注册
# ============================================================

class RegisterRequest(BaseModel):
    room_number: str
    owner_name: str
    phone: str
    password: str


@app.post("/api/register")
async def register_resident(req: RegisterRequest):
    """住户自助注册"""
    ok = db.add_owner(req.room_number, req.password, req.owner_name, req.phone)
    if not ok:
        raise HTTPException(409, detail=f"门牌号 {req.room_number} 已存在，请直接登录或联系管理员")
    db.add_notification("", "新住户注册", f"{req.room_number}室 {req.owner_name} 已注册", "info")
    return {"success": True, "message": "注册成功"}


# ============================================================
# 住户自助接口
# ============================================================

@app.get("/api/my-payments")
async def my_payments(token: str):
    """查询当前住户的物业费记录（直接返回JSON，不走Agent）"""
    ctx = _resident_contexts.get(token)
    if not ctx:
        raise HTTPException(401, detail="未登录")
    rows = db.get_payment_by_room(ctx["room_number"])
    return {"payments": rows, "room_number": ctx["room_number"]}


@app.post("/api/pay/{pay_id}")
async def pay_bill(pay_id: int, token: str):
    """模拟在线支付：标记指定物业费记录为已缴纳"""
    ctx = _resident_contexts.get(token)
    if not ctx:
        raise HTTPException(401, detail="未登录")

    # 验证该记录属于当前住户
    rows = db.get_payment_by_room(ctx["room_number"])
    target = None
    for r in rows:
        if r["id"] == pay_id:
            target = r
            break
    if not target:
        raise HTTPException(404, detail="记录不存在")
    if target["status"] == "paid":
        raise HTTPException(400, detail="该记录已缴纳")

    ok = db.update_payment_status(pay_id, "paid")
    if not ok:
        raise HTTPException(500, detail="支付失败")
    return {"success": True, "message": f"支付成功，金额 {target['amount']} 元"}


# ============================================================
# 消息通知接口
# ============================================================

@app.get("/api/notifications")
async def list_notifications(token: str):
    """获取当前用户的通知列表（住户仅看个人，管理员看全部）"""
    if token in _resident_contexts:
        ctx = _resident_contexts[token]
        notifications = db.get_notifications(ctx["room_number"], include_public=False)
        unread = db.get_unread_count(ctx["room_number"], include_public=False)
    elif token in _admin_sessions:
        notifications = db.get_notifications("", include_public=True)
        unread = db.get_unread_count("", include_public=True)
    else:
        raise HTTPException(401, detail="未登录")
    return {"notifications": notifications, "unread": unread}


@app.get("/api/notifications/unread-count")
async def unread_count(token: str):
    """获取未读通知数量"""
    if token in _resident_contexts:
        room = _resident_contexts[token]["room_number"]
        count = db.get_unread_count(room, include_public=False)
    elif token in _admin_sessions:
        count = db.get_unread_count("", include_public=True)
    else:
        raise HTTPException(401, detail="未登录")
    return {"unread": count}


@app.post("/api/notifications/{nid}/read")
async def mark_read(nid: int, token: str):
    """标记通知为已读"""
    if token not in _resident_contexts and token not in _admin_sessions:
        raise HTTPException(401, detail="未登录")
    db.mark_notification_read(nid)
    return {"success": True}


# ============================================================
# 情感分析统计接口
# ============================================================

@app.get("/api/sentiment/stats")
async def sentiment_stats(token: str, hours: int = 24):
    """管理员查看情感分析统计数据"""
    if token not in _admin_sessions:
        raise HTTPException(401, detail="未登录或权限不足")
    return db.get_sentiment_stats(hours)


# ============================================================
# 报修工单接口
# ============================================================

class RepairRequest(BaseModel):
    title: str
    description: str


class RepairUpdateRequest(BaseModel):
    status: str = None
    admin_note: str = None


@app.get("/api/repairs/my")
async def my_repairs(token: str):
    """住户查看自己的报修工单"""
    ctx = _resident_contexts.get(token)
    if not ctx:
        raise HTTPException(401, detail="未登录")
    repairs = db.get_repairs_by_room(ctx["room_number"])
    return {"repairs": repairs}


@app.post("/api/repairs")
async def submit_repair(req: RepairRequest, token: str):
    """住户提交报修"""
    ctx = _resident_contexts.get(token)
    if not ctx:
        raise HTTPException(401, detail="未登录")
    rid = db.add_repair(ctx["room_number"], req.title, req.description)
    db.add_notification(ctx["room_number"], "报修已提交", f"您的报修「{req.title}」已提交，请等待处理", "repair")
    db.add_notification("", "新报修工单", f"{ctx['room_number']}室提交了报修：{req.title}", "repair")
    return {"success": True, "id": rid}


@app.get("/api/repairs")
async def list_repairs(token: str):
    """管理员查看全部工单"""
    _verify_admin(token)
    return {"repairs": db.get_all_repairs()}


@app.put("/api/repairs/{repair_id}")
async def update_repair_api(repair_id: int, req: RepairUpdateRequest, token: str):
    """管理员更新工单状态"""
    _verify_admin(token)
    ok = db.update_repair(repair_id, req.status, req.admin_note)
    if not ok:
        raise HTTPException(404, detail="工单不存在")

    # 发送通知给对应住户
    repairs = db.get_all_repairs()
    for r in repairs:
        if r["id"] == repair_id:
            status_label = {"pending": "待处理", "processing": "处理中",
                           "completed": "已完成", "cancelled": "已取消"}
            db.add_notification(r["room_number"], "工单状态更新",
                f"您的报修「{r['title']}」状态已更新为：{status_label.get(req.status, req.status or '')}",
                "repair")
            break
    return {"success": True}


# ============================================================
# 物业费管理接口（管理员专用）
# ============================================================

class PublishPaymentRequest(BaseModel):
    room_number: str = "all"   # "all" 或具体门牌号
    amount: float
    year: int
    period: str               # "1月"..."12月" / "第一季度"..."第四季度" / "上半年"/"下半年"/"全年"
    notes: str = ""


def _calc_due_date(year: int, period: str) -> str:
    """根据年份和周期计算截止日期"""
    period_map = {
        "1月": "01-31", "2月": "02-28", "3月": "03-31", "4月": "04-30",
        "5月": "05-31", "6月": "06-30", "7月": "07-31", "8月": "08-31",
        "9月": "09-30", "10月": "10-31", "11月": "11-30", "12月": "12-31",
        "第一季度": "03-31", "第二季度": "06-30",
        "第三季度": "09-30", "第四季度": "12-31",
        "上半年": "06-30", "下半年": "12-31", "全年": "12-31",
    }
    md = period_map.get(period, "12-31")
    return f"{year}-{md}"


@app.post("/api/payments/publish")
async def publish_payment(req: PublishPaymentRequest, token: str):
    """管理员发布物业费（单户或批量全部业主）"""
    _verify_admin(token)

    due_date = _calc_due_date(req.year, req.period)
    note_text = f"{req.year}年{req.period} | {req.notes}" if req.notes else f"{req.year}年{req.period}"

    if req.room_number == "all":
        owners = db.get_all_owners()
        rooms = [o["room_number"] for o in owners]
    else:
        rooms = [req.room_number]

    created = []
    for room in rooms:
        pid = db.add_payment(room, req.amount, due_date, notes=note_text)
        created.append(pid)
        db.add_notification(room, "物业费待缴",
            f"{req.year}年{req.period}物业费 ¥{req.amount} 已发布，请及时缴纳", "payment")

    # 记录操作日志
    log_id = db.add_payment_log("publish", "管理员", len(created), req.amount,
                                note_text, created)

    return {
        "success": True,
        "count": len(created),
        "rooms": rooms,
        "due_date": due_date,
        "log_id": log_id,
    }


@app.get("/api/payments/logs")
async def payment_logs(token: str):
    """获取最近的物业费操作日志"""
    _verify_admin(token)
    return {"logs": db.get_payment_logs()}


class SetPaymentStatusRequest(BaseModel):
    pay_id: int
    status: str


@app.post("/api/payments/set-status")
async def set_payment_status(req: SetPaymentStatusRequest, token: str):
    """管理员手动设置物业费缴纳状态"""
    _verify_admin(token)
    valid = {"paid", "unpaid"}
    if req.status not in valid:
        raise HTTPException(400, detail=f"无效状态，可选: {valid}")
    ok = db.update_payment_status(req.pay_id, req.status,
        "" if req.status == "unpaid" else None)
    if not ok:
        raise HTTPException(404, detail="记录不存在")
    return {"success": True}


@app.post("/api/payments/undo/{log_id}")
async def undo_payment(log_id: int, token: str):
    """撤销指定日志对应的物业费发布操作"""
    _verify_admin(token)
    result = db.undo_payment_log(log_id)
    if result is None:
        raise HTTPException(404, detail="日志不存在")
    if "error" in result:
        raise HTTPException(400, detail=result["error"])
    return {"success": True, **result}


@app.get("/api/payments/grouped")
async def payments_grouped(token: str):
    """按门牌号分组返回物业费数据，用于管理员查询面板"""
    _verify_admin(token)
    all_payments = db.get_all_payments()
    owners = {o["room_number"]: o["owner_name"] for o in db.get_all_owners()}

    grouped = {}
    for p in all_payments:
        room = p["room_number"]
        if room not in grouped:
            grouped[room] = {
                "room_number": room,
                "owner_name": owners.get(room, ""),
                "records": [],
                "total_unpaid": 0,
            }
        grouped[room]["records"].append(p)
        if p["status"] != "paid":
            grouped[room]["total_unpaid"] += p["amount"]

    result = sorted(grouped.values(), key=lambda x: x["room_number"])
    return {"groups": result}


# ============================================================
# 公开接口
# ============================================================

@app.get("/api/announcements")
async def list_announcements_public():
    """查看全部公告列表（公开，无需登录），供住户端浏览"""
    rows = db.get_all_announcements()
    return {"announcements": rows}


@app.get("/api/get_announcement/{ann_id}")
async def get_announcement_detail(ann_id: int):
    """查看单条公告详情（公开）"""
    rows = db.get_all_announcements()
    for r in rows:
        if r["id"] == ann_id:
            return r
    raise HTTPException(404, detail="公告不存在")


# ============================================================
# 管理员权限校验辅助
# ============================================================

def _verify_admin(token: str) -> str:
    """校验管理员 token，失败抛出 401，成功返回管理员标识"""
    if token not in _admin_sessions:
        raise HTTPException(401, detail="未登录或非管理员会话")
    return "admin"


# ============================================================
# ADVER 公告文件管理接口（管理员专用）
# ============================================================

@app.get("/api/adver/files")
async def list_adver_files(token: str):
    """列出 ADVER/ 文件夹中的所有公告文件及其关联的公告ID"""
    _verify_admin(token)
    files = []
    adver_path = Path(ADVER_DIR)
    if not adver_path.exists():
        return {"files": files}

    # 查所有公告的 source_file → id 映射
    conn = __import__('sqlite3').connect(db.DB_PATH)
    conn.row_factory = __import__('sqlite3').Row
    cur = conn.cursor()
    cur.execute("SELECT id, source_file FROM announcements WHERE source_file IS NOT NULL")
    source_map = {row["source_file"]: row["id"] for row in cur.fetchall()}
    conn.close()

    for fpath in sorted(adver_path.glob("*")):
        if fpath.suffix.lower() not in (".txt", ".pdf"):
            continue
        stat = fpath.stat()
        files.append({
            "filename": fpath.name,
            "size_bytes": stat.st_size,
            "modified": __import__('datetime').datetime.fromtimestamp(
                stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "type": fpath.suffix.lower()[1:],
            "ann_id": source_map.get(fpath.name),
        })
    return {"files": files}


@app.post("/api/adver/upload")
async def upload_adver_file(token: str, file: UploadFile = File(...)):
    """上传公告文件（TXT/PDF），自动解析并写入 SQLite + Chroma"""
    _verify_admin(token)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".txt", ".pdf"):
        raise HTTPException(400, detail="仅支持 TXT 或 PDF 文件")

    # 保存文件到 ADVER/
    os.makedirs(ADVER_DIR, exist_ok=True)
    dest_path = os.path.join(ADVER_DIR, file.filename)
    with open(dest_path, "wb") as f:
        content_bytes = await file.read()
        f.write(content_bytes)

    # 如果已有同名文件的公告记录，先删除
    db.delete_announcement_by_source(file.filename)

    # 解析文件并写入数据库
    try:
        title, content = db.parse_uploaded_file(dest_path)
    except Exception as e:
        os.remove(dest_path)
        raise HTTPException(400, detail=f"文件解析失败: {str(e)}")

    new_id = db.add_announcement(title, content, source_file=file.filename)
    db.add_notification("", "新公告发布", f"物业发布了新公告：{title}", "info")
    return {"success": True, "ann_id": new_id, "title": title, "filename": file.filename}


@app.delete("/api/adver/files/{filename:path}")
async def delete_adver_file(filename: str, token: str):
    """删除 ADVER/ 中的公告文件及对应的数据库记录"""
    _verify_admin(token)
    filename = unquote(filename)

    file_path = os.path.join(ADVER_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="文件不存在")

    os.remove(file_path)
    db.delete_announcement_by_source(filename)
    return {"success": True, "deleted": filename}


@app.post("/api/adver/reindex")
async def reindex_adver(token: str):
    """清空全部公告并从 ADVER/ 重建索引"""
    _verify_admin(token)

    db.delete_all_announcements()

    adver_path = Path(ADVER_DIR)
    count = 0
    for fpath in sorted(adver_path.glob("*")):
        if fpath.suffix.lower() not in (".txt", ".pdf"):
            continue
        try:
            title, content = db.parse_uploaded_file(str(fpath))
            db.add_announcement(title, content, source_file=fpath.name)
            count += 1
        except Exception as e:
            print(f"[reindex] 跳过 {fpath.name}: {e}")

    return {"success": True, "count": count}


# ============================================================
# 业主管理接口（管理员专用）
# ============================================================

@app.get("/api/owners")
async def list_owners(token: str):
    """列出全部业主信息"""
    _verify_admin(token)
    owners = db.get_all_owners()
    return {"owners": owners}


@app.post("/api/owners")
async def create_owner(req: OwnerRequest, token: str):
    """新增业主"""
    _verify_admin(token)
    ok = db.add_owner(req.room_number, req.password, req.owner_name, req.phone)
    if not ok:
        raise HTTPException(409, detail=f"门牌号 {req.room_number} 已存在")
    return {"success": True}


@app.put("/api/owners/{room_number}")
async def update_owner_api(room_number: str, req: OwnerUpdateRequest, token: str):
    """修改业主信息"""
    _verify_admin(token)
    kwargs = {}
    if req.owner_name is not None:
        kwargs["owner_name"] = req.owner_name
    if req.phone is not None:
        kwargs["phone"] = req.phone
    if req.password is not None:
        kwargs["password"] = req.password
    if not kwargs:
        raise HTTPException(400, detail="至少需要提供一个要修改的字段")
    ok = db.update_owner(room_number, **kwargs)
    if not ok:
        raise HTTPException(404, detail=f"门牌号 {room_number} 不存在")
    return {"success": True}


@app.delete("/api/owners/{room_number}")
async def delete_owner_api(room_number: str, token: str):
    """删除业主"""
    _verify_admin(token)
    ok = db.delete_owner(room_number)
    if not ok:
        raise HTTPException(404, detail=f"门牌号 {room_number} 不存在")
    return {"success": True}
# ============================================================
# 管理员账号管理接口（管理员专用）
# ============================================================

class AdminRequest(BaseModel):
    username: str
    password: str
    name: str = ""
    role: str = "admin"


class AdminUpdateRequest(BaseModel):
    password: str = None
    name: str = None
    role: str = None


@app.get("/api/admins")
async def list_admins(token: str):
    """列出全部管理员账号"""
    _verify_admin(token)
    admins = db.get_all_admins()
    return {"admins": admins}


@app.post("/api/admins")
async def create_admin(req: AdminRequest, token: str):
    """新增管理员账号"""
    _verify_admin(token)
    ok = db.add_admin(req.username, req.password, req.name, req.role)
    if not ok:
        raise HTTPException(409, detail=f"用户名 {req.username} 已存在")
    return {"success": True}


@app.put("/api/admins/{username}")
async def update_admin_api(username: str, req: AdminUpdateRequest, token: str):
    """修改管理员账号信息"""
    _verify_admin(token)
    kwargs = {}
    if req.password is not None:
        kwargs["password"] = req.password
    if req.name is not None:
        kwargs["name"] = req.name
    if req.role is not None:
        kwargs["role"] = req.role
    if not kwargs:
        raise HTTPException(400, detail="至少需要提供一个要修改的字段")
    ok = db.update_admin(username, **kwargs)
    if not ok:
        raise HTTPException(404, detail=f"用户名 {username} 不存在")
    return {"success": True}


@app.delete("/api/admins/{username}")
async def delete_admin_api(username: str, token: str):
    """删除管理员账号"""
    _verify_admin(token)
    admins = db.get_all_admins()
    if len(admins) <= 1:
        raise HTTPException(400, detail="不能删除最后一位管理员")
    ok = db.delete_admin(username)
    if not ok:
        raise HTTPException(404, detail=f"用户名 {username} 不存在")
    return {"success": True}


if __name__ == "__main__":
    import uvicorn
    print("=" * 55)
    print("  物业多Agent协作系统")
    print("  浏览器打开: http://localhost:9090")
    print()
    print("  管理员账号存于 admin/管理员.xlsx")
    print("  住户账号:   门牌号 101 / 密码 101001")
    print("             门牌号 102 / 密码 102001")
    print("             门牌号 201 / 密码 201001")
    print("             门牌号 302 / 密码 302001")
    print("=" * 55)
    uvicorn.run(app, host="127.0.0.1", port=9090)
