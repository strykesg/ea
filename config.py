import os
from dataclasses import dataclass

@dataclass
class Config:
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    SESSION_SECRET: str = os.getenv("SESSION_SECRET", "dev-secret-key")
    
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    GENERATION_MODEL: str = "mistralai/mistral-7b-instruct"
    REFINEMENT_MODEL: str = "openai/gpt-4"
    
    MAX_WORKERS: int = 10
    MAX_CONCURRENT_REQUESTS: int = 20
    RETRY_ATTEMPTS: int = 3
    RETRY_DELAY: float = 1.0
    
    TARGET_RECORDS: int = 5000
    OUTPUT_FILE: str = "output/training_data.jsonl"
    
    HOST: str = "0.0.0.0"
    PORT: int = 5000

config = Config()
