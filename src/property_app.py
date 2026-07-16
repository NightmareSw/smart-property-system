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

app = FastAPI(title="物业多Agent系统", version="1.0.0")

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
    """管理员 Agent 对话接口"""
    executor = _admin_sessions.get(req.token)
    if not executor:
        raise HTTPException(401, detail="未登录或会话已过期")

    start = time.time()
    try:
        result = executor.invoke({"input": req.input})
        output = result.get("output", "（无响应）")
    except Exception as e:
        print(f"[admin_chat] 错误: {e}")
        raise HTTPException(500, detail=str(e))

    return ChatResponse(
        output=output,
        role="admin",
        response_time=round(time.time() - start, 2),
    )


@app.post("/chat/resident", response_model=ChatResponse)
async def chat_resident(req: ChatRequest):
    """住户客服 Agent 对话接口"""
    executor = _resident_sessions.get(req.token)
    if not executor:
        raise HTTPException(401, detail="未登录或会话已过期")

    ctx = _resident_contexts.get(req.token, {})
    # 确保工具中的上下文是最新的（多 token 场景下需要刷新）
    set_resident_context(
        room_number=ctx.get("room_number", ""),
        owner_name=ctx.get("owner_name", ""),
        password=ctx.get("password", ""),
    )

    start = time.time()
    try:
        result = executor.invoke({"input": req.input})
        output = result.get("output", "（无响应）")
    except Exception as e:
        print(f"[resident_chat] 错误: {e}")
        raise HTTPException(500, detail=str(e))

    return ChatResponse(
        output=output,
        role="resident",
        response_time=round(time.time() - start, 2),
    )


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
