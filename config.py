import os
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# 환경변수들 정의
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
OPENAI_STREAMING = os.getenv("OPENAI_STREAMING", "false").lower() == "true"
