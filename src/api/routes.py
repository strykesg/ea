from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import asyncio
import json

router = APIRouter()
templates = Jinja2Templates(directory="templates")

async def event_stream(pipeline):
    while True:
        try:
            status = await pipeline.get_status()
            data = json.dumps(status)
            yield f"data: {data}\n\n"
            await asyncio.sleep(1)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            break

@router.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@router.get("/api/status")
async def get_status(request: Request):
    pipeline = request.app.state.pipeline
    status = await pipeline.get_status()
    return JSONResponse(status)

@router.post("/api/start")
async def start_generation(request: Request):
    pipeline = request.app.state.pipeline
    body = await request.json()
    num_records = body.get("num_records", None)
    
    asyncio.create_task(pipeline.start_generation(num_records))
    
    return JSONResponse({"status": "started"})

@router.post("/api/stop")
async def stop_generation(request: Request):
    pipeline = request.app.state.pipeline
    result = await pipeline.stop_generation()
    return JSONResponse(result)

@router.get("/api/stream")
async def stream_status(request: Request):
    pipeline = request.app.state.pipeline
    return StreamingResponse(
        event_stream(pipeline),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
