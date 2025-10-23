import asyncio
import sqlite3
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path: str = "data/training_records.db"):
        self.db_path = db_path
        self.write_queue: asyncio.Queue = asyncio.Queue()
        self.writer_task: Optional[asyncio.Task] = None
        
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_records (
                id INTEGER PRIMARY KEY,
                record_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at 
            ON training_records(created_at)
        """)
        
        conn.commit()
        conn.close()
    
    async def start_writer(self):
        if self.writer_task is None or self.writer_task.done():
            self.writer_task = asyncio.create_task(self._writer_worker())
    
    async def stop_writer(self):
        if self.writer_task and not self.writer_task.done():
            await self.write_queue.put(None)
            await self.writer_task
    
    async def _writer_worker(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        cursor = conn.cursor()
        
        while True:
            try:
                item = await self.write_queue.get()
                
                if item is None:
                    break
                
                record_id, record_data = item
                
                cursor.execute(
                    "INSERT OR REPLACE INTO training_records (id, record_data) VALUES (?, ?)",
                    (record_id, json.dumps(record_data))
                )
                conn.commit()
                
            except Exception as e:
                print(f"Database write error: {e}")
            finally:
                self.write_queue.task_done()
        
        conn.close()
    
    async def save_record(self, record_id: int, record_data: Dict[str, Any]):
        await self.write_queue.put((record_id, record_data))
    
    async def flush(self):
        await self.write_queue.join()
    
    def get_record(self, record_id: int) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        cursor = conn.cursor()
        
        cursor.execute("SELECT record_data FROM training_records WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        return None
    
    def get_all_records(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        cursor = conn.cursor()
        
        cursor.execute("SELECT record_data FROM training_records ORDER BY id")
        rows = cursor.fetchall()
        conn.close()
        
        return [json.loads(row[0]) for row in rows]
    
    def get_total_count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM training_records")
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def get_existing_ids(self) -> set:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM training_records")
        ids = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        return ids
    
    def clear_all(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM training_records")
        conn.commit()
        conn.close()
    
    def export_to_jsonl(self, output_file: str):
        records = self.get_all_records()
        
        with open(output_file, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        
        return len(records)
