# 智慧物业系统 — 高并发与容错机制技术文档

## 一、SQLite WAL 模式 + 连接管理器

### 问题

SQLite 默认 `journal_mode=delete`，写操作期间整个数据库被锁，所有读操作阻塞。多用户场景下频繁出现 `SQLITE_BUSY`。

### 方案

启用 WAL (Write-Ahead Logging) 模式，读写分离：

```python
# src/property_db.py — _get_conn()
def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")        # 读写并发
    conn.execute("PRAGMA busy_timeout=5000")        # 忙时等5秒
    conn.execute("PRAGMA synchronous=NORMAL")       # 安全与性能平衡
    conn.execute("PRAGMA cache_size=-8000")         # 8MB 缓存
    return conn
```

| 参数 | 作用 |
|------|------|
| `journal_mode=WAL` | 写操作写入 WAL 文件，读操作不受阻 |
| `busy_timeout=5000` | 遇到锁时等待最多5秒而非立即报错 |
| `synchronous=NORMAL` | 关键时机 fsync，非每次写入 |
| `cache_size=-8000` | 负数表示 KB，8MB 页缓存减少磁盘 IO |

### 效果

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| 2 用户同时查物业费 | 第二个阻塞等写入完成 | 并发执行 |
| 高并发写入 | SQLITE_BUSY 直接报错 | 5 秒内自动重试 |

---

## 二、异步 Agent 调用

### 问题

`executor.invoke()` 是同步调用，LLM 耗时 2-10 秒，直接阻塞 FastAPI 事件循环，期间所有其他请求无法处理。

### 方案

```python
# src/property_app.py — /chat/admin 端点
@app.post("/chat/admin")
async def chat_admin(req: ChatRequest):
    # ...
    result = await asyncio.to_thread(executor.invoke, {"input": req.input})
    #                              ^^^^^^^^^^^^
    #                              将阻塞调用放入线程池，释放事件循环
```

`asyncio.to_thread()` 将同步 Agent 调用转移到独立线程，FastAPI 事件循环立即释放处理下一个请求。

### 效果

```
之前: 请求A(LLM 10s) ──────────────────► 响应A
      请求B(API)         等待... 等待... ──► 响应B

之后: 请求A(LLM 10s) ──────────────────► 响应A
      请求B(API)      ──► 响应B (立即)
```

---

## 三、DB 操作自动重试

### 问题

高并发下 WAL 模式虽大幅改善，极端情况仍可能遇到锁冲突。

### 方案

```python
# src/resilience.py — @retry_db 装饰器
def retry_db(max_retries: int = 3, delay: float = 0.3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() or "busy" in str(e).lower():
                        if attempt < max_retries - 1:
                            time.sleep(delay * (attempt + 1))  # 指数退避
                            continue
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        continue
                    raise
            raise last_err
        return wrapper
    return decorator
```

关键写操作加装饰器：

```python
# src/property_db.py
@retry_db(max_retries=3)
def add_announcement(...): ...

@retry_db(max_retries=3)
def add_payment(...): ...

@retry_db(max_retries=3)
def add_repair(...): ...
```

重试策略：第1次等0.3s → 第2次等0.6s → 第3次等0.9s → 最终失败。

---

## 四、LLM 调用熔断器

### 问题

DeepSeek API 故障时，每个请求都会等待超时（数十秒），大量请求堆积耗尽线程池。

### 方案

三态熔断器：

```python
# src/resilience.py — CircuitBreaker 类
class CircuitBreaker:
    """
    状态机:
      CLOSED → (连续失败 >= 3) → OPEN
      OPEN → (等待30秒) → HALF_OPEN
      HALF_OPEN → (成功) → CLOSED | (失败) → OPEN
    """
    def __init__(self, failure_threshold=3, recovery_time=30.0):
        self.state = "CLOSED"
        # ...

    def call(self, func, *args, fallback=None, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time < self.recovery_time:
                return fallback() if fallback else raise CircuitBreakerOpenError()
            self.state = "HALF_OPEN"  # 超时，试探恢复

        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"   # 试探成功，恢复
            return result
        except Exception:
            self.failure_count += 1
            if self.failure_count >= 3:
                self.state = "OPEN"     # 连续失败，熔断
            return fallback() if fallback else raise
```

### 集成到公告生成

```python
# src/agents/admin_agent.py — admin_generate_announcement
def _call_llm():
    return _llm.invoke([HumanMessage(content=prompt)])

def _fallback():
    # 降级：直接用用户输入拼一条简易公告
    return AIMessage(content=f"【{topic}】\n\n{details}\n\nxxx物业管理中心")

response = llm_breaker.call(_call_llm, fallback=_fallback)
```

### 效果

```
正常:  请求 → LLM → 精美公告
故障1: 请求 → LLM → 失败 (计数1)
故障2: 请求 → LLM → 失败 (计数2)
故障3: 请求 → LLM → 失败 (计数3 → 熔断!)
故障4: 请求 → 跳过LLM → 返回简易公告 (毫秒级)
...30秒后...
试探:  请求 → LLM → 成功 → 恢复
```

---

## 五、ChromaDB 降级搜索

### 问题

ChromaDB 向量库故障时，住户公告搜索完全不可用。

### 方案

```python
# src/resilience.py — safe_chroma_search()
def safe_chroma_search(chroma_func, fallback_func, query, k=4):
    try:
        results = chroma_func(query, k)
        if results:
            return results
    except Exception as e:
        print(f"[resilience] Chroma 搜索失败，降级 SQL LIKE: {e}")

    # 降级：SQL LIKE 关键词匹配
    if fallback_func:
        rows = fallback_func(query)
        return [{"id": r["id"], "title": r["title"], ...} for r in rows]
    return []
```

### 集成

```python
# src/agents/resident_agent.py — search_announcements 工具
rows = safe_chroma_search(
    db.search_announcements_rag,     # 主路径: Chroma 向量搜索
    db.search_announcements,         # 降级:   SQL LIKE 模糊搜索
    query, k=4
)
```

Chroma 宕机时自动切换 SQL LIKE，用户无感知。

---

## 六、全局异常中间件

### 问题

未捕获异常直接暴露给用户（traceback），体验差且泄露内部信息。

### 方案

```python
# src/property_app.py
@app.middleware("http")
async def error_middleware(request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        err = format_error(exc)
        status = 503 if err["error"] == "service_unavailable" else 500
        return JSONResponse(status_code=status, content={"detail": err["message"]})
```

错误分类：

```python
# src/resilience.py — format_error()
def format_error(exc: Exception) -> dict:
    if isinstance(exc, CircuitBreakerOpenError):
        return {"error": "service_unavailable",
                "message": "AI 服务暂时不可用，请稍后重试"}
    if isinstance(exc, sqlite3.OperationalError):
        return {"error": "database_error",
                "message": "数据库繁忙，请稍后重试"}
    if isinstance(exc, sqlite3.IntegrityError):
        return {"error": "conflict",
                "message": "数据冲突，该记录可能已存在"}
    return {"error": "internal_error",
            "message": f"服务内部错误"}
```

---

## 防护矩阵总览

| 层级 | 机制 | 保护对象 | 降级策略 |
|------|------|----------|----------|
| 数据库 | WAL 模式 | SQLite | 读写并发，锁等待5s |
| 数据库 | @retry_db | 写操作 | 3次指数退避重试 |
| LLM | CircuitBreaker | deepseek API | 熔断后返回简易公告 |
| Chroma | safe_chroma_search | 向量搜索 | 降级 SQL LIKE |
| Agent | asyncio.to_thread | 事件循环 | 线程池隔离 |
| 全局 | error_middleware | 所有异常 | 友好错误 + HTTP状态码 |
