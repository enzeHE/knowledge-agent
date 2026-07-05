from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Claude API
    claude_api_key: str
    claude_api_base: str
    claude_model: str = "claude-3-5-sonnet-20241022"

    # 数据库
    mysql_url: str
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_docs"

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "knowledge-agent"
    langchain_callbacks_background: str = "false"

    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    @property
    def llm_model(self):
        return self.claude_model

    @property
    def llm_api_key(self):
        return self.claude_api_key

    @property
    def llm_base_url(self):
        return self.claude_api_base

    class Config:
        env_file = ".env"


settings = Settings()
