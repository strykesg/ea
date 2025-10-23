import aiohttp
import asyncio
import json
import random
from typing import Dict, Any, Optional, List
from config import config

class LLMClient:
    EXPERT_PERSONAS = [
        "Quantitative Analyst",
        "Value Investor",
        "Macroeconomic Strategist",
        "Technical Trader",
        "Risk Manager",
        "Behavioral Finance Expert"
    ]
    
    QUESTION_CATEGORIES = {
        "predictive": "What is the likely outcome or price action if [specific event/condition]?",
        "explanatory": "Explain the second-order effects and mechanisms of [phenomenon/event].",
        "comparative": "Compare and contrast the long-term viability, mechanics, or implications of [option A] versus [option B].",
        "counterfactual": "What would have been the optimal strategy or outcome during [historical event] given [constraints]?"
    }
    
    TRADING_TOPICS = [
        "Bitcoin and cryptocurrency markets",
        "Federal Reserve policy and interest rates",
        "Emerging market equities and currencies",
        "Volatility indices and derivatives",
        "Commodity markets (oil, gold, agriculture)",
        "Tech stock valuations and growth metrics",
        "ETF flows and market microstructure",
        "Credit spreads and corporate bonds",
        "Currency carry trades and forex markets",
        "DeFi protocols and tokenomics"
    ]
    
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
    
    async def call_llm(self, prompt: str, model: str, temperature: float = 0.7, system_prompt: Optional[str] = None) -> str:
        if not self.session:
            raise RuntimeError("LLMClient must be used as async context manager")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://replit.com",
            "X-Title": "AI Training Data Generator"
        }
        
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
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
    
    def _get_dynamic_prompt(self, index: int) -> tuple[str, str, str]:
        persona = random.choice(self.EXPERT_PERSONAS)
        category = random.choice(list(self.QUESTION_CATEGORIES.keys()))
        topic = random.choice(self.TRADING_TOPICS)
        
        system_prompt = f"""You are an elite {persona} with deep expertise in financial markets, trading strategies, and economic analysis. Your role is to generate sophisticated, thought-provoking questions that require multi-dimensional analysis and synthesis of diverse data sources."""
        
        category_template = self.QUESTION_CATEGORIES[category]
        
        generation_prompt = f"""Generate a single, highly specific {category} question about {topic}.

The question must:
1. Require synthesis of multiple data types (technical indicators, fundamental data, macroeconomic factors, sentiment)
2. Be concrete and actionable for a professional trader
3. Reflect real-world trading scenarios and decision-making complexity
4. Challenge the responder to think across multiple time frames and market factors

Category guidance: {category_template}

Return ONLY a valid JSON object with this exact structure:
{{
  "question": "your generated question here"
}}

Do not include any explanation, just the JSON object."""
        
        return system_prompt, generation_prompt, persona
    
    async def generate_record(self, index: int) -> Dict[str, Any]:
        system_prompt, generation_prompt, persona = self._get_dynamic_prompt(index)
        
        raw_data = await self.call_llm(
            generation_prompt, 
            config.GENERATION_MODEL, 
            temperature=0.9,
            system_prompt=system_prompt
        )
        
        try:
            generated = json.loads(raw_data)
            question = generated.get("question", "")
        except json.JSONDecodeError:
            question = raw_data.strip()
        
        return {
            "question": question,
            "persona": persona,
            "system_prompt": system_prompt
        }
    
    async def refine_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        question = record.get("question", "")
        persona = record.get("persona", "Expert Trader")
        
        refinement_system = f"""You are an elite {persona} with decades of experience in financial markets. You provide comprehensive, deeply analytical responses that synthesize technical analysis, fundamental data, macroeconomic factors, and market sentiment."""
        
        refinement_prompt = f"""Provide a comprehensive, expert-level answer to this trading question:

{question}

Your answer must:
1. Synthesize multiple analytical frameworks (technical, fundamental, macro, behavioral)
2. Reference specific indicators, metrics, or data points where relevant
3. Consider multiple time frames and market scenarios
4. Provide actionable insights backed by reasoning
5. Acknowledge uncertainty and risk factors
6. Be detailed yet precise (aim for 150-300 words)

Return ONLY a valid JSON object:
{{
  "answer": "your comprehensive answer here"
}}"""
        
        refined_data = await self.call_llm(
            refinement_prompt, 
            config.REFINEMENT_MODEL, 
            temperature=0.3,
            system_prompt=refinement_system
        )
        
        try:
            refined = json.loads(refined_data)
            answer = refined.get("answer", "")
        except json.JSONDecodeError:
            answer = refined_data.strip()
        
        return {
            "question": question,
            "answer": answer,
            "persona": persona,
            "system_prompt": refinement_system
        }
