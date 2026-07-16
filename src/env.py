# 配置模型环境
import os
from pathlib import Path
from dotenv import load_dotenv

# 使用绝对路径加载 .env，确保无论从哪个目录运行都能找到
_load_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_load_path, override=True)

DEEPSEEK_V4_PRO_API_KEY=os.getenv("DEEPSEEK_V4_PRO_API_KEY")
DEEPSEEK_V4_PRO_BASE_URL=os.getenv("DEEPSEEK_V4_PRO_BASE_URL")
DEEPSEEK_V4_PRO_MODEL=os.getenv("DEEPSEEK_V4_PRO_MODEL")
DEEPSEEK_R1_BASE_URL=os.getenv("DEEPSEEK_R1_BASE_URL")
DEEPSEEK_R1_MODEL=os.getenv("DEEPSEEK_R1_MODEL")


