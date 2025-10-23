import aiohttp
import asyncio
import json
from typing import Dict, Any, Optional
from config import config

class LLMClient:
    def __init__(self):
        self.api_key = config.OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.session: Optional[aiohttp.ClientSession] = None
        self.retry_attempts = config.RETRY_ATTEMPTS
        self.retry_delay = config.RETRY_DELAY
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def call_llm(self, prompt: str, model: str, temperature: float = 0.7) -> str:
        if not self.session:
            raise RuntimeError("LLMClient must be used as async context manager")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://replit.com",
            "X-Title": "AI Training Data Generator"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        
        for attempt in range(self.retry_attempts):
            try:
                async with self.session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    elif response.status == 429:
                        wait_time = self.retry_delay * (2 ** attempt)
                        await asyncio.sleep(wait_time)
                    else:
                        text = await response.text()
                        raise Exception(f"API Error {response.status}: {text}")
            except asyncio.TimeoutError:
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise
        
        raise Exception(f"Failed after {self.retry_attempts} attempts")
    
    async def generate_record(self, index: int) -> Dict[str, Any]:
        generation_prompt = f"""Generate a creative and diverse AI training record #{index}.
Create a realistic conversation or instruction-response pair.
Make it unique, educational, and high-quality.
Return only JSON with keys: "instruction", "input", "output"."""
        
        raw_data = await self.call_llm(generation_prompt, config.GENERATION_MODEL, temperature=0.9)
        
        try:
            generated = json.loads(raw_data)
        except json.JSONDecodeError:
            generated = {
                "instruction": "Generated content",
                "input": "",
                "output": raw_data
            }
        
        return generated
    
    async def refine_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        refinement_prompt = f"""Refine and improve this AI training record.
Enhance clarity, correctness, and educational value.
Ensure high quality and proper formatting.
Return only improved JSON with same structure.

Original record:
{json.dumps(record, indent=2)}"""
        
        refined_data = await self.call_llm(refinement_prompt, config.REFINEMENT_MODEL, temperature=0.3)
        
        try:
            refined = json.loads(refined_data)
            if "instruction" in refined and "output" in refined:
                return refined
        except json.JSONDecodeError:
            pass
        
        return record
