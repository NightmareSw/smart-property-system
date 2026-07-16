"""
property_db.py —— 物业数据库层
=============================
双存储架构：
- SQLite:    结构化数据（住户、公告、物业费），支持精确 CRUD
- Chroma:    公告向量库，支持语义搜索（RAG）

公告的增删改操作会同步双写 SQLite + Chroma，保证数据一致性。
"""

import os
import sys
import sqlite3
import shutil
import hashlib
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

DB_PATH = str(Path(__file__).parent.parent / "property.db")
CHROMA_DIR = str(Path(__file__).parent.parent / "chroma_announcements_db")
ADVER_DIR = str(Path(__file__).parent.parent / "ADVER")
OWNER_EXCEL = str(Path(__file__).parent.parent / "OWNER" / "业主信息.xlsx")
PAYMENT_EXCEL = str(Path(__file__).parent.parent / "OWNER" / "物业费.xlsx")
ADMIN_EXCEL = str(Path(__file__).parent.parent / "admin" / "管理员.xlsx")

# --- Embedding 模型（全局单例，所有公告共享一个向量库）---
_embed_model = SentenceTransformerEmbeddings(
    model_name=r"C:\Users\Alienware\.cache\modelscope\models\Qwen--Qwen3-Embedding-0.6B\snapshots\master",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Chroma 向量库实例（模块级单例）
_announcement_vdb: Chroma = None


# ============================================================
# 数据库初始化
# ============================================================

def _hash(plain: str) -> str:
    """简单密码哈希（仅 demo 使用，生产环境请用 bcrypt）"""
    return hashlib.sha256(plain.encode()).hexdigest()


# ============================================================
# 公告向量库（Chroma RAG）—— 语义搜索层
# ============================================================

def _build_announcement_document(ann_id: int, title: str, content: str,
                                  publish_date: str, author: str) -> Document:
    """
    将一条公告转为 Chroma Document。
    文本格式: 标题 + 换行 + 内容，确保标题关键词也被向量化。
    metadata 中保存 SQLite ID，用于删除时定位。
    """
    return Document(
        page_content=f"{title}\n{content}",
        metadata={
            "ann_id": str(ann_id),  # Chroma where 过滤需字符串类型
            "title": title,
            "publish_date": publish_date,
            "author": author,
        }
    )


def _init_chroma():
    """初始化公告 Chroma 向量库（如果已持久化则加载，否则创建）"""
    global _announcement_vdb
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        _announcement_vdb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=_embed_model,
        )
        print("[property_db] 公告向量库已从磁盘加载")
    else:
        _announcement_vdb = Chroma(
            embedding_function=_embed_model,
            persist_directory=CHROMA_DIR,
        )
        print("[property_db] 公告向量库已创建")


def _load_announcements_from_adver() -> list[tuple]:
    """
    从 ADVER 文件夹扫描所有 TXT 文件，每份文件 = 一条公告。

    文件格式（约定）：
      第一行 → 公告标题
      剩余行 → 公告正文
    发布日期取文件的修改时间，作者默认"物业中心"。

    返回: [(title, content, publish_date, author), ...]
    """
    adver_path = Path(ADVER_DIR)
    if not adver_path.exists():
        return []

    results = []
    # 扫描所有 TXT，按文件名排序保证稳定
    for filepath in sorted(adver_path.glob("*.txt")):
        try:
            for encoding in ['utf-8', 'gbk']:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        lines = [l.strip() for l in f.readlines() if l.strip()]
                    if len(lines) >= 2:
                        title = lines[0]
                        content = "\n".join(lines[1:])
                    elif len(lines) == 1:
                        title = lines[0]
                        content = lines[0]
                    else:
                        continue

                    pub_date = datetime.fromtimestamp(
                        filepath.stat().st_mtime
                    ).strftime("%Y-%m-%d")
                    results.append((title, content, pub_date, "物业中心"))
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            print(f"[property_db] 读取公告文件失败 {filepath.name}: {e}")

    return results


def _seed_chroma():
    """将 SQLite 中已有的公告写入 Chroma（仅当 Chroma 为空时执行）"""
    # 检查 Chroma 是否已有数据，避免重复导入导致向量膨胀
    if _announcement_vdb._collection.count() > 0:
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT id, title, content, publish_date, author FROM announcements")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return

    docs = [
        _build_announcement_document(r["id"], r["title"], r["content"],
                                     r["publish_date"], r["author"])
        for r in rows
    ]
    _announcement_vdb.add_documents(docs)
    _announcement_vdb.persist()
    print(f"[property_db] Chroma 种子数据写入完成: {len(docs)} 条公告")


def _chroma_add_announcement(ann_id: int, title: str, content: str,
                              publish_date: str, author: str):
    """向 Chroma 新增一条公告（与 SQLite INSERT 同步调用）"""
    doc = _build_announcement_document(ann_id, title, content, publish_date, author)
    _announcement_vdb.add_documents([doc])
    _announcement_vdb.persist()


def _chroma_delete_announcement(ann_id: int):
    """从 Chroma 删除一条公告（通过 metadata.ann_id 过滤）"""
    try:
        collection = _announcement_vdb._collection
        collection.delete(where={"ann_id": str(ann_id)})
        _announcement_vdb.persist()
    except Exception as e:
        print(f"[property_db] Chroma 删除失败 (id={ann_id}): {e}")


def _chroma_update_announcement(ann_id: int, title: str, content: str,
                                 publish_date: str, author: str):
    """更新 Chroma 中的公告：先删旧文档，再插入新文档"""
    _chroma_delete_announcement(ann_id)
    _chroma_add_announcement(ann_id, title, content, publish_date, author)


def search_announcements_rag(query: str, k: int = 4) -> list[dict]:
    """
    RAG 语义搜索公告 —— 用自然语言查询相关公告。

    与 SQLite LIKE 不同，这里不要求关键词精确匹配。
    "最近有什么安全方面的通知？" 能匹配到"电梯维护公告"。

    Args:
        query: 自然语言查询
        k: 返回最相似的 k 条结果

    Returns:
        公告列表（dict 格式，含相似度分数）
    """
    if _announcement_vdb is None:
        return []

    docs = _announcement_vdb.similarity_search_with_score(query, k=k)
    results = []
    for doc, score in docs:
        results.append({
            "id": int(doc.metadata.get("ann_id", 0)),
            "title": doc.metadata.get("title", ""),
            "content": doc.page_content,
            "publish_date": doc.metadata.get("publish_date", ""),
            "author": doc.metadata.get("author", ""),
            "score": round(float(1.0 - score), 4),  # distance → similarity
        })
    return results


# ============================================================
# 数据库初始化
# ============================================================

def init_db():
    """
    初始化数据库：
    - SQLite: 仅存 announcements 表（公告缓存）
    - Chroma: 公告向量库（RAG 语义搜索）
    - 业主数据: OWNER/业主信息.xlsx
    - 物业费数据: OWNER/物业费.xlsx
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # --- 建表 ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS announcements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            publish_date TEXT NOT NULL,
            author TEXT DEFAULT '物业中心'
        )
    """)

    # --- 迁移：添加 source_file 列（用于关联 ADVER 文件）---
    try:
        cur.execute("ALTER TABLE announcements ADD COLUMN source_file TEXT")
        print("[property_db] source_file 列已添加")
    except sqlite3.OperationalError:
        pass  # 列已存在

    # --- 插入 demo 数据（已存在则跳过）---
    cur.execute("SELECT COUNT(*) FROM announcements")
    if cur.fetchone()[0] == 0:
        # 从 ADVER 文件夹读取公告数据
        announcements = _load_announcements_from_adver()
        if announcements:
            cur.executemany(
                "INSERT INTO announcements (title, content, publish_date, author) VALUES (?,?,?,?)",
                announcements
            )
            print(f"[property_db] 从 ADVER/ 加载了 {len(announcements)} 条公告")
        else:
            print("[property_db] ADVER/ 中无公告文件，跳过")

    conn.commit()
    conn.close()
    print("[property_db] SQLite 初始化完成")

    # --- 初始化公告向量库 ---
    _init_chroma()
    # 首次运行时 SQLite demo 数据刚插入，同步到 Chroma
    _seed_chroma()


# ============================================================
# 业主数据 —— 从 OWNER/业主信息.xlsx 读取
# ============================================================

def get_all_owners() -> list[dict]:
    """
    从 Excel 读取业主数据。

    Excel 格式（约定）：
      门牌号 | 密码 | 姓名 | 电话
      (第一行为表头，从第二行开始读取)

    返回: [{"room_number": "101", "password": "101001", "owner_name": "张三", "phone": "138..."}, ...]
    """
    import openpyxl
    excel_path = Path(OWNER_EXCEL)
    if not excel_path.exists():
        print(f"[property_db] [WARN] 业主 Excel 不存在: {excel_path}")
        return []

    wb = openpyxl.load_workbook(str(excel_path), read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(min_row=2, values_only=True))  # 跳过表头
    wb.close()

    owners = []
    for row in rows:
        if not row[0]:  # 跳过空行
            continue
        owners.append({
            "room_number": str(row[0]).strip(),
            "password": str(row[1]).strip() if row[1] else "",
            "owner_name": str(row[2]).strip() if row[2] else "",
            "phone": str(row[3]).strip() if len(row) > 3 and row[3] else "",
        })
    return owners


# ============================================================
# 业主 CRUD —— 基于 OWNER/业主信息.xlsx
# ============================================================

def _save_owners(rows: list[list]):
    """将业主数据全量写回 Excel"""
    import openpyxl
    path = Path(OWNER_EXCEL)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["门牌号", "密码", "姓名", "电话"])
    for row in rows:
        ws.append(list(row))
    wb.save(str(path))


def add_owner(room_number: str, password: str, owner_name: str, phone: str = "") -> bool:
    """新增业主，门牌号已存在时返回 False"""
    owners = get_all_owners()
    for o in owners:
        if o["room_number"] == str(room_number).strip():
            return False
    rows = [[o["room_number"], o["password"], o["owner_name"], o["phone"]]
            for o in owners]
    rows.append([str(room_number).strip(), password, owner_name, phone])
    _save_owners(rows)
    return True


def update_owner(room_number: str, owner_name: str = None, phone: str = None,
                 password: str = None) -> bool:
    """更新业主信息，未传的字段保持不变。返回 False 表示门牌号不存在"""
    owners = get_all_owners()
    updated = False
    for o in owners:
        if o["room_number"] == str(room_number).strip():
            if owner_name is not None:
                o["owner_name"] = owner_name
            if phone is not None:
                o["phone"] = phone
            if password is not None:
                o["password"] = password
            updated = True
            break
    if not updated:
        return False
    rows = [[o["room_number"], o["password"], o["owner_name"], o["phone"]]
            for o in owners]
    _save_owners(rows)
    return True


def delete_owner(room_number: str) -> bool:
    """删除业主，返回 False 表示门牌号不存在"""
    owners = get_all_owners()
    original_len = len(owners)
    owners = [o for o in owners if o["room_number"] != str(room_number).strip()]
    if len(owners) == original_len:
        return False
    rows = [[o["room_number"], o["password"], o["owner_name"], o["phone"]]
            for o in owners]
    _save_owners(rows)
    return True


# ============================================================
# 认证
# ============================================================

def authenticate(room_number: str, password: str) -> dict | None:
    """
    验证住户登录 —— 从 OWNER/业主信息.xlsx 中比对门牌号和密码。

    每次调用都会重新读取 Excel，因此修改 Excel 后无需重启服务即可生效。
    (如果并发量大，可改为缓存 + 文件修改时间检测)
    """
    owners = get_all_owners()
    for owner in owners:
        if owner["room_number"] == room_number and owner["password"] == password:
            return {
                "room_number": owner["room_number"],
                "owner_name": owner["owner_name"],
                "phone": owner.get("phone", ""),
            }
    return None


# ============================================================
# 管理员数据 —— 从 admin/管理员.xlsx 读取
# ============================================================

def get_all_admins() -> list[dict]:
    """
    从 Excel 读取管理员数据。
    Excel 格式（约定）：
      用户名 | 密码 | 姓名 | 角色
      (第一行为表头，从第二行开始读取)
    返回: [{"username": "admin", "password": "admin123", "name": "系统管理员", "role": "superadmin"}, ...]
    """
    import openpyxl
    excel_path = Path(ADMIN_EXCEL)
    if not excel_path.exists():
        return []
    wb = openpyxl.load_workbook(str(excel_path), read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    wb.close()
    admins = []
    for row in rows:
        if not row[0]:
            continue
        admins.append({
            "username": str(row[0]).strip(),
            "password": str(row[1]).strip() if row[1] else "",
            "name": str(row[2]).strip() if len(row) > 2 and row[2] else "",
            "role": str(row[3]).strip() if len(row) > 3 and row[3] else "admin",
        })
    return admins


def authenticate_admin(username: str, password: str) -> dict | None:
    """验证管理员登录，成功返回管理员信息字典，失败返回 None"""
    admins = get_all_admins()
    for a in admins:
        if a["username"] == username and a["password"] == password:
            return a
    return None


def _save_admins(rows: list[list]):
    """将管理员数据全量写回 Excel"""
    import openpyxl
    path = Path(ADMIN_EXCEL)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["用户名", "密码", "姓名", "角色"])
    for row in rows:
        ws.append(list(row))
    wb.save(str(path))


def add_admin(username: str, password: str, name: str = "", role: str = "admin") -> bool:
    """新增管理员，用户名已存在时返回 False"""
    admins = get_all_admins()
    for a in admins:
        if a["username"] == username:
            return False
    rows = [[a["username"], a["password"], a["name"], a["role"]] for a in admins]
    rows.append([username, password, name, role])
    _save_admins(rows)
    return True


def update_admin(username: str, password: str = None, name: str = None,
                 role: str = None) -> bool:
    """更新管理员信息，未传的字段保持不变。返回 False 表示用户名不存在"""
    admins = get_all_admins()
    updated = False
    for a in admins:
        if a["username"] == username:
            if password is not None:
                a["password"] = password
            if name is not None:
                a["name"] = name
            if role is not None:
                a["role"] = role
            updated = True
            break
    if not updated:
        return False
    rows = [[a["username"], a["password"], a["name"], a["role"]] for a in admins]
    _save_admins(rows)
    return True


def delete_admin(username: str) -> bool:
    """删除管理员，返回 False 表示用户名不存在"""
    admins = get_all_admins()
    original_len = len(admins)
    admins = [a for a in admins if a["username"] != username]
    if len(admins) == original_len:
        return False
    rows = [[a["username"], a["password"], a["name"], a["role"]] for a in admins]
    _save_admins(rows)
    return True


# ============================================================
# 公告 CRUD（AdminAgent 全权限，ResidentAgent 只读）
# ============================================================

def get_all_announcements() -> list[dict]:
    """查询全部公告"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT id, title, content, publish_date, author FROM announcements ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def search_announcements(keyword: str) -> list[dict]:
    """按关键词搜索公告（模糊匹配标题和内容）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, content, publish_date, author FROM announcements "
        "WHERE title LIKE ? OR content LIKE ? ORDER BY publish_date DESC",
        (f"%{keyword}%", f"%{keyword}%")
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def add_announcement(title: str, content: str, author: str = "物业中心",
                     source_file: str = None) -> int:
    """新增公告（SQLite + Chroma 双写），返回新记录的 id"""
    pub_date = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO announcements (title, content, publish_date, author, source_file) "
        "VALUES (?,?,?,?,?)",
        (title, content, pub_date, author, source_file)
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    _chroma_add_announcement(new_id, title, content, pub_date, author)
    return new_id


def update_announcement(ann_id: int, title: str = None, content: str = None) -> bool:
    """更新公告标题和/或内容（SQLite + Chroma 双更新）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM announcements WHERE id=?", (ann_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return False

    new_title = title if title else row["title"]
    new_content = content if content else row["content"]

    if title:
        cur.execute("UPDATE announcements SET title=? WHERE id=?", (title, ann_id))
    if content:
        cur.execute("UPDATE announcements SET content=? WHERE id=?", (content, ann_id))
    conn.commit()
    conn.close()

    # 同步更新 Chroma 向量库
    _chroma_update_announcement(ann_id, new_title, new_content,
                                row["publish_date"], row["author"])
    return True


def delete_announcement(ann_id: int) -> bool:
    """删除公告（SQLite + Chroma 双删）"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM announcements WHERE id=?", (ann_id,))
    if not cur.fetchone():
        conn.close()
        return False
    cur.execute("DELETE FROM announcements WHERE id=?", (ann_id,))
    conn.commit()
    conn.close()
    # 同步从 Chroma 向量库删除
    _chroma_delete_announcement(ann_id)
    return True


def delete_announcement_by_source(source_file: str) -> bool:
    """按 ADVER 源文件名删除公告（SQLite + Chroma 双删）"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM announcements WHERE source_file=?", (source_file,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return False
    ann_id = row[0]
    cur.execute("DELETE FROM announcements WHERE id=?", (ann_id,))
    conn.commit()
    conn.close()
    _chroma_delete_announcement(ann_id)
    return True


def delete_all_announcements():
    """清空全部公告（SQLite + Chroma），用于重建索引"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM announcements")
    conn.commit()
    conn.close()
    try:
        _announcement_vdb._collection.delete(where={})
        _announcement_vdb.persist()
        print("[property_db] Chroma 公告集合已清空")
    except Exception as e:
        print(f"[property_db] Chroma 清空失败: {e}")


def parse_uploaded_file(filepath: str) -> tuple:
    """
    解析上传的文件（TXT 或 PDF），返回 (title, content)。

    TXT: 第一非空行 = 标题，剩余行 = 正文
    PDF: 文件名（无扩展名）= 标题，全部提取文本 = 正文
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        for encoding in ["utf-8", "gbk"]:
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                if len(lines) >= 2:
                    return lines[0], "\n".join(lines[1:])
                elif len(lines) == 1:
                    return lines[0], lines[0]
                else:
                    raise ValueError("文件内容为空")
            except UnicodeDecodeError:
                continue
        raise ValueError("无法解码TXT文件")

    elif ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        if not pages:
            raise ValueError("PDF文件为空或无法解析")
        title = os.path.splitext(os.path.basename(filepath))[0]
        content = ""
        for i, page in enumerate(pages):
            content += f"\n--- 第{i+1}页 ---\n{page.page_content}"
        return title, content.strip()

    else:
        raise ValueError(f"不支持的文件类型: {ext}")


# ============================================================
# 物业费 CRUD —— 基于 OWNER/物业费.xlsx（AdminAgent 全权限）
# ============================================================

def _load_payments_xl() -> list[list]:
    """
    读取物业费 Excel，返回每一行的原始列表（含表头行索引信息）。

    Excel 格式：门牌号 | 金额 | 截止日期 | 状态 | 缴纳日期
    第一行为表头，从第二行开始是数据。
    """
    import openpyxl
    path = Path(PAYMENT_EXCEL)
    if not path.exists():
        return []
    wb = openpyxl.load_workbook(str(path))
    ws = wb.active
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    wb.close()
    return rows


def _save_payments_xl(header: list, rows: list[list]):
    """将数据写回物业费 Excel（全量覆写）"""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(header)
    for row in rows:
        ws.append(list(row))
    wb.save(str(Path(PAYMENT_EXCEL)))


def get_all_payments() -> list[dict]:
    """查询全部物业费记录"""
    rows = _load_payments_xl()
    results = []
    for i, row in enumerate(rows):
        if not row[0]:
            continue
        results.append({
            "id": i + 1,                     # 行号即 ID
            "room_number": str(row[0]).strip(),
            "amount": float(row[1]) if row[1] else 0.0,
            "due_date": str(row[2]).strip() if row[2] else "",
            "status": str(row[3]).strip() if row[3] else "unpaid",
            "paid_date": str(row[4]).strip() if len(row) > 4 and row[4] else "",
        })
    return results


def get_payment_by_room(room_number: str) -> list[dict]:
    """
    按门牌号查询物业费记录。

    这是 ResidentAgent → AdminAgent 跨 Agent 通信的核心函数。
    """
    all_rows = get_all_payments()
    return [r for r in all_rows if r["room_number"] == str(room_number).strip()]


def get_payments_by_status(status: str) -> list[dict]:
    """按缴费状态筛选物业费记录（paid / unpaid / overdue）"""
    all_rows = get_all_payments()
    return [r for r in all_rows if r.get("status", "") == status]


def get_db_stats() -> dict:
    """获取数据库统计信息"""
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM announcements")
    ann_count = cur.fetchone()[0]
    conn.close()

    payments = get_all_payments()
    owners = get_all_owners()

    chroma_count = 0
    try:
        if _announcement_vdb:
            chroma_count = _announcement_vdb._collection.count()
    except Exception:
        pass

    return {
        "announcements": ann_count,
        "chroma_vectors": chroma_count,
        "payments": len(payments),
        "paid": sum(1 for p in payments if p.get("status") == "paid"),
        "unpaid": sum(1 for p in payments if p.get("status") == "unpaid"),
        "overdue": sum(1 for p in payments if p.get("status") == "overdue"),
        "owners": len(owners),
    }


def add_payment(room_number: str, amount: float, due_date: str, status: str = "unpaid") -> int:
    """新增物业费记录（追加到 Excel 末尾）"""
    import openpyxl
    path = Path(PAYMENT_EXCEL)
    wb = openpyxl.load_workbook(str(path))
    ws = wb.active
    new_row_idx = ws.max_row + 1
    ws.append([str(room_number), amount, due_date, status, ""])
    wb.save(str(path))
    return new_row_idx - 1  # 减去表头行


def update_payment_status(pay_id: int, status: str, paid_date: str = None) -> bool:
    """更新缴费状态（按行号定位）"""
    rows = _load_payments_xl()
    idx = pay_id - 1  # ID = 行号，转为0-based
    if idx < 0 or idx >= len(rows):
        return False
    row = list(rows[idx])
    row[3] = status
    row[4] = paid_date or datetime.now().strftime("%Y-%m-%d")
    header = ["门牌号", "金额", "截止日期", "状态", "缴纳日期"]
    all_rows = [list(r) for r in rows]
    all_rows[idx] = row
    _save_payments_xl(header, all_rows)
    return True


def delete_payment(pay_id: int) -> bool:
    """删除物业费记录（按行号删除）"""
    rows = _load_payments_xl()
    idx = pay_id - 1
    if idx < 0 or idx >= len(rows):
        return False
    all_rows = [list(r) for r in rows]
    del all_rows[idx]
    header = ["门牌号", "金额", "截止日期", "状态", "缴纳日期"]
    _save_payments_xl(header, all_rows)
    return True


# ============================================================
# 启动时自动初始化（import 即执行）
# ============================================================
if __name__ == "__main__":
    # 直接运行时重建：清理旧数据，重新初始化
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    print("旧数据已清理，重新初始化...")
    init_db()
    print("DB created & seeded.")
else:
    # import 时只初始化一次（SQLite 建表幂等，Chroma 加载已有数据）
    init_db()
