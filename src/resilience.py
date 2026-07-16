"""
resilience.py —— 容错、熔断、降级机制
===================================
- DB 操作自动重试（解决 WAL 锁冲突）
- LLM 调用熔断器（连续失败后快速失败 + 自动恢复）
- Chroma 降级（失败时回退 SQL LIKE 搜索）
"""
import time
import functools
import sqlite3
from threading import Lock


# ============================================================
# 1. DB 操作重试装饰器
# ============================================================

def retry_db(max_retries: int = 3, delay: float = 0.3):
    """数据库操作自动重试，处理 SQLITE_BUSY 等临时错误"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    last_err = e
                    if "locked" in str(e).lower() or "busy" in str(e).lower():
                        if attempt < max_retries - 1:
                            time.sleep(delay * (attempt + 1))
                            continue
                except Exception as e:
                    last_err = e
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        continue
                    raise
            raise last_err
        return wrapper
    return decorator


# ============================================================
# 2. LLM 熔断器
# ============================================================

class CircuitBreaker:
    """
    熔断器状态机:
      CLOSED → (连续失败 >= threshold) → OPEN
      OPEN → (等待 recovery_time) → HALF_OPEN
      HALF_OPEN → (成功) → CLOSED | (失败) → OPEN
    """

    def __init__(self, failure_threshold: int = 3, recovery_time: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED / OPEN / HALF_OPEN
        self.lock = Lock()

    def call(self, func, *args, fallback=None, **kwargs):
        """
        通过熔断器调用函数。
        - CLOSED: 正常调用，记录失败
        - OPEN: 直接返回 fallback（如有）或抛异常
        - HALF_OPEN: 试探调用，成功则恢复
        """
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time >= self.recovery_time:
                    self.state = "HALF_OPEN"
                else:
                    if fallback is not None:
                        return fallback()
                    raise CircuitBreakerOpenError(
                        f"熔断器开启中，{self.recovery_time - (time.time() - self.last_failure_time):.0f}s 后恢复")

        try:
            result = func(*args, **kwargs)
            with self.lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
            return result
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
            if fallback is not None:
                return fallback()
            raise e


class CircuitBreakerOpenError(Exception):
    pass


# LLM 熔断器实例（全局共享）
llm_breaker = CircuitBreaker(failure_threshold=3, recovery_time=30.0)


# ============================================================
# 3. Chroma 降级包装器
# ============================================================

def safe_chroma_search(chroma_func, fallback_func, query: str, k: int = 4) -> list[dict]:
    """
    Chroma 语义搜索，失败时自动降级到 SQL LIKE。
    返回: [{"id":..., "title":..., "content":..., ...}, ...]
    """
    try:
        results = chroma_func(query, k)
        if results:
            return results
    except Exception as e:
        print(f"[resilience] Chroma 搜索失败，降级 SQL LIKE: {e}")

    # 降级
    if fallback_func:
        rows = fallback_func(query)
        if rows:
            return [{"id": r["id"], "title": r["title"], "content": r["content"],
                     "publish_date": r["publish_date"], "author": r["author"],
                     "score": 0} for r in rows]
    return []


# ============================================================
# 4. 全局异常响应格式化
# ============================================================

def format_error(exc: Exception) -> dict:
    """将异常转为用户友好的错误响应"""
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
            "message": f"服务内部错误: {str(exc)[:200]}"}
