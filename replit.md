# Elite Trading AI Training Data Generator

## Overview
High-concurrency data generation pipeline for creating elite-level trading AI training datasets in JSONL format for Axolotl finetuning. The system uses dynamic prompt engineering with expert personas to generate diverse financial questions, then a two-step LLM process via OpenRouter API: fast question generation (Mistral Nemo) followed by powerful answer refinement (DeepSeek R1).

The pipeline generates questions across multiple expert personas (Quantitative Analyst, Value Investor, Macroeconomic Strategist, Technical Trader, Risk Manager, Behavioral Finance Expert) and question categories (predictive, explanatory, comparative, counterfactual) to ensure maximum dataset diversity.

## Architecture
- **FastAPI** backend with async workers
- **SQLite database** with async write queue for persistent storage
- **In-memory** state management with asyncio locks for real-time metrics
- **OpenRouter API** integration for LLM calls
- **Real-time dashboard** with Server-Sent Events (SSE)
- **JSONL output** format for Axolotl compatibility
- **Resume capability** - skips already-generated records on restart

## Project Structure
```
├── config.py                  # Configuration settings
├── main.py                    # Application entry point
├── src/
│   ├── core/
│   │   ├── llm_client.py     # OpenRouter API client with dynamic prompting
│   │   ├── pipeline.py       # Data generation pipeline
│   │   ├── database.py       # SQLite database manager with write queue
│   │   └── state_manager.py  # In-memory state management
│   └── api/
│       ├── app.py            # FastAPI application
│       └── routes.py         # API endpoints
├── templates/
│   └── dashboard.html        # Web dashboard UI
├── data/
│   └── training_records.db   # SQLite database (persists across restarts)
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
✅ **Dynamic prompt engineering with 6 expert personas** (Quant Analyst, Value Investor, Macro Strategist, etc.)
✅ **4 question categories** (Predictive, Explanatory, Comparative, Counterfactual)
✅ **Diverse trading topics** (crypto, forex, equities, derivatives, DeFi, etc.)
✅ **Axolotl-compatible chat_template format** (system/user/assistant messages)
✅ **SQLite persistence** - dataset survives restarts and deployments
✅ **Resume capability** - automatically skips already-generated records
✅ **One-click reset** - delete all records and start fresh
✅ High-concurrency async pipeline with worker pools
✅ Two-step LLM process (generate questions + refine answers)
✅ Real-time progress monitoring via Server-Sent Events (SSE)
✅ **Live activity feed showing questions as they're generated**
✅ **One-click JSONL download** when generation completes
✅ **Configurable worker count via MAX_WORKERS environment variable**
✅ Automatic retry with exponential backoff
✅ Async write queue to prevent SQLite lock contention
✅ **VM deployment configuration for production**

## Usage
1. Start the application (workflow runs automatically)
2. Open the web dashboard at `https://<your-repl>.replit.dev`
3. Set the number of records to generate (default: 5000)
4. Click "Start Generation"
5. Monitor real-time progress in the dashboard
6. **If interrupted, restart - it will resume from where it left off**
7. Download JSONL file when complete (button appears automatically)
8. Find generated data in `output/training_data.jsonl` and `data/training_records.db`

### Set and Forget Mode
- Set your desired record count and click "Start Generation"
- **Close the browser** - generation continues in the background
- **Replit will keep it running** - dataset persists across restarts
- Come back anytime to check progress or download results
- Use the **Reset button** when you want to start a new dataset

## API Endpoints
- `GET /` - Web dashboard with live activity feed
- `GET /api/status` - Get current pipeline status (includes total_persisted count)
- `GET /api/activity` - Get recent generation activity
- `POST /api/start` - Start generation (body: `{"num_records": 5000}`)
- `POST /api/stop` - Stop generation
- `POST /api/reset` - Clear all records from database and JSONL file
- `GET /api/download` - Download training_data.jsonl file
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
- `src/core/llm_client.py` - OpenRouter API client with dynamic prompting
- `src/core/pipeline.py` - Data generation pipeline with persistence
- `src/core/database.py` - SQLite manager with async write queue
- `src/core/state_manager.py` - In-memory state management
- `src/api/app.py` - FastAPI application
- `src/api/routes.py` - API endpoints (including reset)
- `templates/dashboard.html` - Web dashboard with reset button
- `data/training_records.db` - SQLite database (auto-created, persists data)
- `Dockerfile` - Production Docker image
- `docker-compose.yml` - Local Docker testing
- `DEPLOY_DOKPLOY.md` - Dokploy deployment guide

## Output Format
The system generates training data in the Axolotl-compatible `chat_template` format. Each JSONL line contains:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an elite [persona] with deep expertise in financial markets..."
    },
    {
      "role": "user",
      "content": "[Generated trading question requiring multi-dimensional analysis]"
    },
    {
      "role": "assistant",
      "content": "[Comprehensive expert answer synthesizing technical, fundamental, and macro factors]"
    }
  ]
}
```

This format is optimized for Axolotl finetuning with `type: chat_template` in your config YAML.

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
- 2025-10-23: **Implemented dynamic prompt engineering with 6 expert personas and 4 question categories**
- 2025-10-23: **Changed output to Axolotl chat_template format (system/user/assistant messages)**
- 2025-10-23: **Added JSONL download button in dashboard (appears when generation completes)**
- 2025-10-23: **Dashboard now uses EventSource (SSE) for real-time updates instead of polling**
- 2025-10-23: **Added SQLite database for persistent storage across restarts and deployments**
- 2025-10-23: **Implemented resume capability - automatically skips already-generated records**
- 2025-10-23: **Added reset button to delete all records and start fresh**
- 2025-10-23: **Added async write queue to prevent SQLite lock contention**
