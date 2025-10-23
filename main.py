import uvicorn
from src.api.app import create_app
from config import config

if __name__ == "__main__":
    app = create_app()
    
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )
