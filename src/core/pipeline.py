import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.core.llm_client import LLMClient
from src.core.state_manager import StateManager
from src.core.database import DatabaseManager
from config import config

class DataPipeline:
    def __init__(self):
        self.state_manager = StateManager()
        self.db_manager = DatabaseManager()
        self.llm_client: Optional[LLMClient] = None
        self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        self.output_file = config.OUTPUT_FILE
        self.is_running = False
        self.was_stopped = False
    
    async def initialize(self):
        await self.state_manager.connect()
        await self.state_manager.initialize_pipeline(config.TARGET_RECORDS)
        await self.db_manager.start_writer()
    
    async def shutdown(self):
        await self.db_manager.stop_writer()
        await self.state_manager.disconnect()
    
    async def process_record(self, record_id: int) -> Optional[Dict[str, Any]]:
        async with self.semaphore:
            try:
                assert self.llm_client is not None
                
                await self.state_manager.add_activity({
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "generating",
                    "record_id": record_id,
                    "message": f"Generating record #{record_id}..."
                })
                
                generated = await self.llm_client.generate_record(record_id)
                await self.state_manager.increment("generated")
                
                await self.state_manager.add_activity({
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "generated",
                    "record_id": record_id,
                    "question": generated.get("question", "")[:100],
                    "message": f"Generated #{record_id}: {generated.get('question', '')[:80]}..."
                })
                
                await self.state_manager.add_activity({
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "refining",
                    "record_id": record_id,
                    "message": f"Refining record #{record_id}..."
                })
                
                refined = await self.llm_client.refine_record(generated)
                await self.state_manager.increment("refined")
                
                record = {
                    "messages": [
                        {
                            "role": "system",
                            "content": refined.get("system_prompt", "You are an expert financial analyst and trader.")
                        },
                        {
                            "role": "user",
                            "content": refined.get("question", "")
                        },
                        {
                            "role": "assistant",
                            "content": refined.get("answer", "")
                        }
                    ]
                }
                
                await self.state_manager.mark_completed(record_id, record)
                await self.state_manager.increment("completed")
                
                await self.state_manager.add_activity({
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "completed",
                    "record_id": record_id,
                    "question": refined.get("question", "")[:100],
                    "message": f"Completed #{record_id}: {refined.get('question', '')[:80]}..."
                })
                
                return record
            except Exception as e:
                await self.state_manager.increment("failed")
                await self.state_manager.add_activity({
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "error",
                    "record_id": record_id,
                    "message": f"Failed #{record_id}: {str(e)}"
                })
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
                await self.save_record(record_id, record)
    
    async def save_record(self, record_id: int, record: Dict[str, Any]):
        try:
            await self.db_manager.save_record(record_id, record)
        except Exception as e:
            print(f"Error saving record to database: {e}")
    
    async def start_generation(self, num_records: Optional[int] = None):
        if self.is_running:
            return {"error": "Pipeline already running"}
        
        if num_records is None:
            num_records = config.TARGET_RECORDS
        
        self.is_running = True
        self.was_stopped = False
        
        await self.state_manager.initialize_pipeline(num_records)
        await self.state_manager.update_state(status="running")
        
        await self.db_manager.flush()
        existing_ids = self.db_manager.get_existing_ids()
        
        for i in range(num_records):
            record_id = i + 1
            if record_id not in existing_ids:
                await self.state_manager.add_to_queue(record_id)
            else:
                await self.state_manager.increment("completed")
                await self.state_manager.mark_completed(record_id, {})
        
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
                await self.db_manager.flush()
                await self.state_manager.update_state(status="completed")
                await self.export_to_jsonl()
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
        await self.db_manager.flush()
        state["total_persisted"] = self.db_manager.get_total_count()
        return state
    
    async def export_to_jsonl(self):
        try:
            await self.db_manager.flush()
            count = self.db_manager.export_to_jsonl(self.output_file)
            await self.state_manager.add_activity({
                "timestamp": datetime.utcnow().isoformat(),
                "type": "completed",
                "record_id": 0,
                "message": f"Exported {count} records to {self.output_file}"
            })
            return count
        except Exception as e:
            print(f"Error exporting to JSONL: {e}")
            return 0
    
    async def reset_dataset(self):
        try:
            await self.db_manager.stop_writer()
            self.db_manager.clear_all()
            
            with open(self.output_file, "w") as f:
                f.write("")
            
            await self.db_manager.start_writer()
            
            await self.state_manager.initialize_pipeline(config.TARGET_RECORDS)
            await self.state_manager.add_activity({
                "timestamp": datetime.utcnow().isoformat(),
                "type": "completed",
                "record_id": 0,
                "message": "Dataset reset - all records cleared"
            })
            
            return {"status": "reset", "message": "Dataset cleared successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
