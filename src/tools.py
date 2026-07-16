from datetime import datetime

from langchain_core.tools import tool


@tool
def Add(a: float, b: float) -> float:
    """计算两个数的和。"""
    return a + b


@tool
def Minus(a: float, b: float) -> float:
    """计算两个数的差。"""
    return a - b


@tool
def Multiply(a: float, b: float) -> float:
    """计算两个数的乘积。"""
    return a * b


@tool
def Divide(a: float, b: float) -> float:
    """计算两个数的商。"""
    return a / b


@tool
def GetCurrentTime() -> str:
    """获取当前的日期和时间，返回格式为 'YYYY-MM-DD HH:MM:SS'。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


tools = [Add, Minus, Multiply, Divide, GetCurrentTime]