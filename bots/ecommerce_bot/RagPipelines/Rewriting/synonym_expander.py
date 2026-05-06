import json
from typing import List, Set

class SynonymExpander:
    """Expand query with domain-specific synonyms"""
    
    def __init__(self, custom_synonyms: dict = None):
        self.synonym_dict = custom_synonyms or {
            'cheap': ['budget', 'affordable', 'inexpensive', 'economical', 'value'],
            'fast': ['quick', 'rapid', 'speedy', 'high-performance', 'efficient'],
            'good': ['quality', 'excellent', 'great', 'superior', 'reliable'],
            'buy': ['purchase', 'acquire', 'get', 'order', 'shop for'],
            'computer': ['laptop', 'desktop', 'pc', 'machine', 'system'],
            'phone': ['smartphone', 'mobile', 'cellular', 'device'],
            'big': ['large', 'spacious', 'roomy', 'generous'],
            'small': ['compact', 'mini', 'portable', 'lightweight']
        }
    
    def expand(self, query: str, max_synonyms: int = 2) -> List[str]:
        """Generate query variations with synonyms"""
        words = query.lower().split()
        expansions = [query]  # Original query
        
        for i, word in enumerate(words):
            if word in self.synonym_dict:
                for syn in self.synonym_dict[word][:max_synonyms]:
                    new_words = words.copy()
                    new_words[i] = syn
                    expansions.append(' '.join(new_words))
        
        return list(set(expansions))  # Deduplicate
    
    def expand_with_llm(self, query: str, llm_client, max_variations: int = 3) -> List[str]:
        """Use LLM for intelligent synonym expansion"""
        prompt = f"""Generate {max_variations} alternative search queries for: "{query}"
        
        Rules:
        - Keep the same intent
        - Vary vocabulary while preserving meaning
        - Use domain-appropriate synonyms
        
        Return as JSON array: ["query1", "query2", "query3"]"""
        
        response = llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200
        )
        
        try:
            variations = json.loads(response.choices[0].message.content)
            return [query] + variations[:max_variations]
        except:
            return [query]

# Usage
expander = SynonymExpander()
queries = expander.expand("cheap fast computer")
# Returns: ["cheap fast computer", "budget fast computer", "affordable fast computer", 
#           "cheap quick computer", "cheap rapid computer"]