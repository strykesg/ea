from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from src.core.pipeline import DataPipeline
from src.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = DataPipeline()
    await pipeline.initialize()
    app.state.pipeline = pipeline
    yield
    await pipeline.shutdown()

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Training Data Generator",
        description="High-concurrency pipeline for generating AI training datasets",
        version="1.0.0",
        lifespan=lifespan
    )
    
    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.include_router(router)
    
    return app
