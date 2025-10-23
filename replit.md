# AI Training Data Generator

## Overview
High-concurrency data generation pipeline for creating AI training datasets in JSONL format compatible with Axolotl. The system uses a two-step LLM process via OpenRouter API: fast generation (Mistral Nemo) followed by powerful refinement (DeepSeek R1).

## Architecture
- **FastAPI** backend with async workers
- **In-memory** state management with asyncio locks
- **OpenRouter API** integration for LLM calls
- **Real-time dashboard** with Server-Sent Events (SSE)
- **JSONL output** format for Axolotl compatibility

## Project Structure
```
├── config.py                  # Configuration settings
├── main.py                    # Application entry point
├── src/
│   ├── core/
│   │   ├── llm_client.py     # OpenRouter API client
│   │   ├── pipeline.py       # Data generation pipeline
│   │   └── state_manager.py  # State management
│   └── api/
│       ├── app.py            # FastAPI application
│       └── routes.py         # API endpoints
├── templates/
│   └── dashboard.html        # Web dashboard UI
├── output/                   # Generated JSONL files
└── static/                   # Static assets

## Configuration

### Environment Variables (Replit Secrets)
Set these environment variables to configure the pipeline:

**Required:**
- `OPENROUTER_API_KEY`: Your OpenRouter API key (required for LLM calls)

**Optional:**
- `GENERATOR_MODEL`: Fast LLM for initial generation (default: mistralai/mistral-nemo)
- `REFINER_MODEL`: Powerful LLM for refinement (default: deepseek/deepseek-r1-0528-qwen3-8b)
- `MAX_WORKERS`: Number of concurrent workers for speed control (default: 10)
- `MAX_CONCURRENT_REQUESTS`: API request concurrency limit (default: 20)
- `SESSION_SECRET`: Secret key for sessions (default: auto-generated)

### Other Configurable Settings (config.py)
- `TARGET_RECORDS`: Default number of records to generate (default: 5000)

## Features
✅ High-concurrency async pipeline with worker pools
✅ Two-step LLM process (generate + refine)
✅ Real-time progress monitoring via web dashboard
✅ **Live activity feed showing questions as they're generated**
✅ **Configurable worker count via MAX_WORKERS environment variable**
✅ Automatic retry with exponential backoff
✅ In-memory state management for progress tracking
✅ JSONL output format for Axolotl
✅ **VM deployment configuration for production**

## Usage
1. Start the application (workflow runs automatically)
2. Open the web dashboard at `https://<your-repl>.replit.dev`
3. Set the number of records to generate
4. Click "Start Generation"
5. Monitor real-time progress in the dashboard
6. Find generated data in `output/training_data.jsonl`

## API Endpoints
- `GET /` - Web dashboard with live activity feed
- `GET /api/status` - Get current pipeline status
- `GET /api/activity` - Get recent generation activity
- `POST /api/start` - Start generation (body: `{"num_records": 5000}`)
- `POST /api/stop` - Stop generation
- `GET /api/stream` - Server-Sent Events stream for real-time updates

## Deployment

### Option 1: Replit Deployment
The app is configured for VM deployment on Replit, which is ideal for this stateful, long-running pipeline:
- Maintains state in server memory
- Keeps workers running continuously
- Perfect for high-concurrency data generation
- Click "Deploy" in Replit to publish with a live URL

### Option 2: Dokploy Deployment
For self-hosted deployment on your own VPS using Dokploy:
- Production-ready Dockerfile included
- Full deployment guide in `DEPLOY_DOKPLOY.md`
- Supports custom domains with automatic SSL
- Resource control and scaling
- See [Dokploy Deployment Guide](./DEPLOY_DOKPLOY.md) for detailed instructions

## Files

- `main.py` - Application entry point
- `config.py` - Configuration with environment variables
- `src/core/llm_client.py` - OpenRouter API client
- `src/core/pipeline.py` - Data generation pipeline
- `src/core/state_manager.py` - In-memory state management
- `src/api/app.py` - FastAPI application
- `src/api/routes.py` - API endpoints
- `templates/dashboard.html` - Web dashboard
- `Dockerfile` - Production Docker image
- `docker-compose.yml` - Local Docker testing
- `DEPLOY_DOKPLOY.md` - Dokploy deployment guide

## Recent Changes
- 2025-10-23: Initial project setup with FastAPI and OpenRouter integration
- 2025-10-23: Implemented in-memory state manager (removed Redis dependency)
- 2025-10-23: Created real-time web dashboard with SSE
- 2025-10-23: Added two-step LLM pipeline with concurrency control
- 2025-10-23: Made LLM models configurable via environment variables (GENERATOR_MODEL, REFINER_MODEL)
- 2025-10-23: Updated defaults to mistralai/mistral-nemo and deepseek/deepseek-r1-0528-qwen3-8b
- 2025-10-23: Added live activity feed to show questions as they're generated in real-time
- 2025-10-23: Made MAX_WORKERS configurable via environment variable for speed control
- 2025-10-23: Configured VM deployment for production use (Replit)
- 2025-10-23: Added Dockerfile and Docker Compose for Dokploy deployment
