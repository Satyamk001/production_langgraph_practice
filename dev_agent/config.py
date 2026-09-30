from dotenv import load_dotenv
from pathlib import Path
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model(
    model="gemini-3.8-flash",
    model_provider="google_genai",
    temperature=0,
    max_retries=5,
)

WORKSPACE = Path("workspace")
WORKSPACE.mkdir(exist_ok=True)
