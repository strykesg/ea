from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
import asyncio
import json
import os

router = APIRouter()
templates = Jinja2Templates(directory="templates")

async def event_stream(pipeline):
    while True:
        try:
            status = await pipeline.get_status()
            activity = await pipeline.state_manager.get_recent_activity(20)
            data = json.dumps({
                **status,
                "activity": activity
            })
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
    
    if pipeline.is_running:
        return JSONResponse({"error": "Pipeline already running"}, status_code=400)
    
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

@router.get("/api/activity")
async def get_activity(request: Request):
    pipeline = request.app.state.pipeline
    activity = await pipeline.state_manager.get_recent_activity(50)
    return JSONResponse({"activity": activity})

@router.get("/api/download")
async def download_jsonl(request: Request):
    pipeline = request.app.state.pipeline
    output_file = pipeline.output_file
    
    if not os.path.exists(output_file):
        return JSONResponse({"error": "No training data available"}, status_code=404)
    
    return FileResponse(
        path=output_file,
        media_type="application/jsonl",
        filename="training_data.jsonl",
        headers={
            "Content-Disposition": "attachment; filename=training_data.jsonl"
        }
    )

@router.post("/api/reset")
async def reset_dataset(request: Request):
    pipeline = request.app.state.pipeline
    
    if pipeline.is_running:
        return JSONResponse({"error": "Cannot reset while generation is running"}, status_code=400)
    
    result = await pipeline.reset_dataset()
    return JSONResponse(result)
