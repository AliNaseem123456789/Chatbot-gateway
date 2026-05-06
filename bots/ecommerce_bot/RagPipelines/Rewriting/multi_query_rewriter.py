from typing import List, Set, Dict
import asyncio
import json
class MultiQueryRewriter:
    """Generate multiple query variations for better recall"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def generate_variations(self, query: str, num_variations: int = 3) -> List[str]:
        """Generate different formulations of the same intent"""
        
        prompt = f"""Generate {num_variations} different search queries for: "{query}"

Requirements:
- Each query should have the same core intent
- Vary vocabulary, structure, and level of detail
- Make each query natural and searchable
- Include both specific and general formulations

Return as JSON array: ["query1", "query2", "query3"]"""
        
        response = self.llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Higher temperature for diversity
            max_tokens=300
        )
        
        try:
            variations = json.loads(response.choices[0].message.content)
            return [query] + variations[:num_variations]
        except:
            return [query]
    
    async def multi_query_search(self, query: str, search_function, 
                                num_variations: int = 3) -> List[Dict]:
        """Execute search with multiple query variations and combine results"""
        
        variations = await self.generate_variations(query, num_variations)
        
        # Search with each variation
        all_results = []
        for variant in variations:
            results = await search_function(variant, limit=3)
            all_results.extend(results)
        
        # Deduplicate by ID
        unique_results = {}
        for result in all_results:
            result_id = result.get('id') or result.get('product_id')
            if result_id and result_id not in unique_results:
                unique_results[result_id] = result
        
        # Rerank combined results (optional)
        reranked = await self.rerank_results(query, list(unique_results.values()))
        
        return reranked
    
    async def rerank_results(self, original_query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results based on relevance to original query"""
        
        if len(results) <= 5:
            return results
        
        # Create a prompt for reranking
        items_text = "\n".join([
            f"{i+1}. {r.get('name', 'Unknown')} - {r.get('description', '')[:100]}"
            for i, r in enumerate(results[:15])
        ])
        
        prompt = f"""Original query: "{original_query}"

Candidates:
{items_text}

Rank the candidates by relevance (most relevant first).
Return ONLY numbers separated by commas (e.g., "3,1,5,2,4"):"""
        
        response = self.llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        
        try:
            ranking = [int(x.strip()) - 1 for x in response.choices[0].message.content.split(',')]
            reranked = [results[i] for i in ranking if i < len(results)]
            return reranked
        except:
            return results

# Usage
# multi_rewriter = MultiQueryRewriter(llm_client=groq_client)
# results = await multi_rewriter.multi_query_search(
#     query="best budget wireless headphones",
#     search_function=your_hybrid_search,
#     num_variations=3
# )