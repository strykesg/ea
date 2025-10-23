import json
import asyncio
from typing import Dict, Any, Optional
from collections import deque

class StateManager:
    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.queue: deque = deque()
        self.completed: Dict[int, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
    
    async def connect(self):
        pass
    
    async def disconnect(self):
        pass
    
    async def initialize_pipeline(self, total_records: int):
        async with self.lock:
            self.state = {
                "total_records": total_records,
                "generated": 0,
                "refined": 0,
                "completed": 0,
                "failed": 0,
                "status": "idle"
            }
            self.queue.clear()
            self.completed.clear()
    
    async def get_state(self) -> Dict[str, Any]:
        async with self.lock:
            return self.state.copy()
    
    async def update_state(self, **kwargs):
        async with self.lock:
            self.state.update(kwargs)
    
    async def increment(self, field: str, amount: int = 1):
        async with self.lock:
            self.state[field] = self.state.get(field, 0) + amount
    
    async def add_to_queue(self, record_id: int):
        async with self.lock:
            self.queue.append(record_id)
    
    async def get_from_queue(self) -> Optional[int]:
        async with self.lock:
            if self.queue:
                return self.queue.popleft()
            return None
    
    async def mark_completed(self, record_id: int, data: Dict[str, Any]):
        async with self.lock:
            self.completed[record_id] = data
    
    async def get_completed_count(self) -> int:
        async with self.lock:
            return len(self.completed)
