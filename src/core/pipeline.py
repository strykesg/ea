import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.core.llm_client import LLMClient
from src.core.state_manager import StateManager
from config import config

class DataPipeline:
    def __init__(self):
        self.state_manager = StateManager()
        self.llm_client: Optional[LLMClient] = None
        self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        self.output_file = config.OUTPUT_FILE
        self.is_running = False
        self.was_stopped = False
    
    async def initialize(self):
        await self.state_manager.connect()
        await self.state_manager.initialize_pipeline(config.TARGET_RECORDS)
    
    async def shutdown(self):
        await self.state_manager.disconnect()
    
    async def process_record(self, record_id: int) -> Optional[Dict[str, Any]]:
        async with self.semaphore:
            try:
                assert self.llm_client is not None
                generated = await self.llm_client.generate_record(record_id)
                await self.state_manager.increment("generated")
                
                refined = await self.llm_client.refine_record(generated)
                await self.state_manager.increment("refined")
                
                record = {
                    "id": record_id,
                    "instruction": refined.get("instruction", ""),
                    "input": refined.get("input", ""),
                    "output": refined.get("output", ""),
                    "generated_at": datetime.utcnow().isoformat()
                }
                
                await self.state_manager.mark_completed(record_id, record)
                await self.state_manager.increment("completed")
                
                return record
            except Exception as e:
                await self.state_manager.increment("failed")
                print(f"Error processing record {record_id}: {e}")
                return None
    
    async def worker(self, worker_id: int):
        while self.is_running:
            record_id = await self.state_manager.get_from_queue()
            if record_id is None:
                await asyncio.sleep(0.1)
                continue
            
            record = await self.process_record(record_id)
            if record:
                await self.write_record(record)
    
    async def write_record(self, record: Dict[str, Any]):
        try:
            with open(self.output_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"Error writing record: {e}")
    
    async def start_generation(self, num_records: Optional[int] = None):
        if self.is_running:
            return {"error": "Pipeline already running"}
        
        if num_records is None:
            num_records = config.TARGET_RECORDS
        
        self.is_running = True
        self.was_stopped = False
        
        with open(self.output_file, "w") as f:
            f.write("")
        
        await self.state_manager.initialize_pipeline(num_records)
        await self.state_manager.update_state(status="running")
        
        for i in range(num_records):
            await self.state_manager.add_to_queue(i + 1)
        
        async with LLMClient() as client:
            self.llm_client = client
            
            workers = [
                asyncio.create_task(self.worker(i))
                for i in range(config.MAX_WORKERS)
            ]
            
            while self.is_running:
                state = await self.state_manager.get_state()
                if state["completed"] + state["failed"] >= num_records:
                    break
                await asyncio.sleep(1)
            
            self.is_running = False
            await asyncio.sleep(0.5)
            
            for task in workers:
                task.cancel()
            
            await asyncio.gather(*workers, return_exceptions=True)
        
        if not self.was_stopped:
            state = await self.state_manager.get_state()
            if state["completed"] + state["failed"] >= num_records:
                await self.state_manager.update_state(status="completed")
                return {"status": "completed"}
        
        current_state = await self.state_manager.get_state()
        return {"status": current_state.get("status", "stopped")}
    
    async def stop_generation(self):
        self.is_running = False
        self.was_stopped = True
        await self.state_manager.update_state(status="stopped")
        return {"status": "stopped"}
    
    async def get_status(self) -> Dict[str, Any]:
        state = await self.state_manager.get_state()
        return state
