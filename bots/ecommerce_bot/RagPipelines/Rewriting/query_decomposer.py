import re
from typing import List, Dict
import json
class QueryDecomposer:
    """Split complex multi-part queries into sub-queries"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
    def rule_based_split(self, query: str) -> List[str]:
        """Simple rule-based decomposition"""
        sub_queries = []
        
        # Split on question marks
        if '?' in query:
            parts = query.split('?')
            for part in parts[:-1]:  # Last part might be empty
                if part.strip():
                    sub_queries.append(part.strip() + '?')
        
        # Split on "and" for compound questions
        if ' and ' in query.lower():
            parts = re.split(r'\s+and\s+', query)
            if len(parts) > 1:
                sub_queries.extend(parts)
        
        # Split on commas that separate clauses
        if ',' in query and query.count(',') <= 3:
            parts = query.split(',')
            if len(parts) > 1:
                sub_queries.extend([p.strip() for p in parts])
        
        return list(set(sub_queries)) or [query]
    
    async def llm_based_decomposition(self, query: str) -> Dict[str, List[str]]:
        """Use LLM for intelligent decomposition with intent classification"""
        
        prompt = f"""Decompose this complex question into independent sub-queries:

Question: "{query}"

Return JSON format:
{{
    "main_intent": "overall goal",
    "sub_queries": [
        "specific question 1",
        "specific question 2",
        "specific question 3"
    ],
    "requires_aggregation": true/false
}}

Rules:
- Each sub-query should be self-contained
- Preserve all original information
- Don't add new information
- Keep queries concise (max 10 words each)"""
        
        response = self.llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except:
            return {"main_intent": query, "sub_queries": [query], "requires_aggregation": False}
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract different entity types from query"""
        entities = {
            "products": [],
            "brands": [],
            "prices": [],
            "specs": [],
            "timeframes": []
        }
        
        # Price extraction
        price_pattern = r'\$\d+(?:\.\d{2})?|\d+\s*(?:dollars|USD|usd)'
        entities["prices"] = re.findall(price_pattern, query, re.IGNORECASE)
        
        # Timeframe extraction
        time_patterns = r'(last|past|previous|next|upcoming)\s+(week|month|year|day)'
        entities["timeframes"] = re.findall(time_patterns, query, re.IGNORECASE)
        
        # Product types (simple list - expand as needed)
        products = ['laptop', 'phone', 'headphone', 'monitor', 'keyboard', 'mouse']
        for product in products:
            if product in query.lower():
                entities["products"].append(product)
        
        return entities

# # Usage
# decomposer = QueryDecomposer(llm_client=groq_client)
# complex_query = "What is the price of iPhone 15 and how does it compare to Samsung Galaxy S24?"
# sub_queries = decomposer.rule_based_split(complex_query)
# # Returns: ["What is the price of iPhone 15", "how does it compare to Samsung Galaxy S24?"]