from langchain_openai import ChatOpenAI
from src.env import DEEPSEEK_R1_BASE_URL, DEEPSEEK_R1_MODEL

# Ollama 提供 OpenAI 兼容 API 端点（/v1），使用 ChatOpenAI 可获得原生 tool calling 支持
deepseek_r1 = ChatOpenAI(
    base_url=f"{DEEPSEEK_R1_BASE_URL}/v1",
    api_key="ollama",  # Ollama 无需真实 key，但不能为空
    model=DEEPSEEK_R1_MODEL,
    temperature=0.0
)
