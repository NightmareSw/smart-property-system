from openai.types.shared_params import reasoning

from src.env import DEEPSEEK_V4_PRO_API_KEY,DEEPSEEK_V4_PRO_MODEL,DEEPSEEK_V4_PRO_BASE_URL
from langchain_openai import ChatOpenAI

deepseek_v4_pro = ChatOpenAI(
    base_url=DEEPSEEK_V4_PRO_BASE_URL,
    api_key=DEEPSEEK_V4_PRO_API_KEY,
    model=DEEPSEEK_V4_PRO_MODEL,
    temperature=0.0

)

