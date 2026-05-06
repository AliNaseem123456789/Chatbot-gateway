from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
# Import all query rewriting components
from .stopword_remover import StopwordRemover
from .synonym_expander import SynonymExpander
from .query_decomposer import QueryDecomposer
from .conversational_rewriter import ConversationalRewriter
from .hyde_rewriter import HyDERewriter
from .multi_query_rewriter import MultiQueryRewriter
from .constraint_extractor import ConstraintExtractor
from .date_normalizer import DateNormalizer
 # If you have separate file

class QueryType(Enum):
    SIMPLE = "simple"
    CONVERSATIONAL = "conversational"
    COMPLEX = "complex"
    TEMPORAL = "temporal"
    CONSTRAINED = "constrained"
    AMBIGUOUS = "ambiguous"

@dataclass
class RewrittenQuery:
    original: str
    rewritten: str
    technique_used: str
    sub_queries: List[str]
    constraints: Dict[str, Any]
    metadata: Dict[str, Any]

class QueryRewritingPipeline:
    """Complete pipeline orchestrating all rewriting techniques"""
    
    def __init__(self, llm_client, category_map: Dict[str, List[str]] = None):
        self.llm_client = llm_client
        self.stopword_remover = StopwordRemover()
        self.synonym_expander = SynonymExpander()
        self.decomposer = QueryDecomposer(llm_client)
        self.conversational_rewriter = ConversationalRewriter(llm_client)
        self.hyde_rewriter = HyDERewriter(llm_client)
        self.multi_rewriter = MultiQueryRewriter(llm_client)
        self.constraint_extractor = ConstraintExtractor()
        self.date_normalizer = DateNormalizer()
        self.category_map = category_map or {}
    
    def classify_query(self, query: str, history: List[Dict] = None) -> QueryType:
        """Classify query to determine which techniques to apply"""
        
        query_lower = query.lower()
        
        # Check for conversational dependencies
        if history and len(history) > 1:
            pronouns = ['it', 'they', 'them', 'that', 'those', 'these']
            if any(p in query_lower.split() for p in pronouns):
                return QueryType.CONVERSATIONAL
        
        # Check for complex structure
        if (' and ' in query_lower or query_lower.count('?') > 1 or
            len(query.split()) > 15):
            return QueryType.COMPLEX
        
        # Check for temporal references
        temporal_words = ['today', 'yesterday', 'week', 'month', 'year', 'recent']
        if any(word in query_lower for word in temporal_words):
            return QueryType.TEMPORAL
        
        # Check for constraints
        if any(op in query_lower for op in ['under', 'over', 'between', 'stars']):
            return QueryType.CONSTRAINED
        
        # Check for ambiguous terms
        ambiguous_terms = ['apple', 'java', 'spring', 'windows', 'tablet']
        if any(term in query_lower for term in ambiguous_terms):
            return QueryType.AMBIGUOUS
        
        return QueryType.SIMPLE
    
    async def rewrite(self, query: str, 
                     conversation_history: List[Dict] = None,
                     domain: str = "general") -> RewrittenQuery:
        """Main entry point for query rewriting"""
        
        query_type = self.classify_query(query, conversation_history)
        rewritten = query
        techniques_used = []
        sub_queries = [query]
        constraints = {}
        metadata = {"original_type": query_type.value}
        
        # Step 1: Conversational rewrite (if needed)
        if query_type == QueryType.CONVERSATIONAL and conversation_history:
            rewritten = await self.conversational_rewriter.rewrite_with_llm(
                query, conversation_history
            )
            techniques_used.append("conversational")
        
        # Step 2: Constraint extraction
        extracted_constraints = self.constraint_extractor.extract_constraints(rewritten)
        if extracted_constraints:
            constraints.update(extracted_constraints)
            rewritten = self.constraint_extractor.clean_query_text(rewritten, constraints)
            techniques_used.append("constraint_extraction")
        
        # Step 3: Date normalization
        normalized_query, date_range = self.date_normalizer.normalize(rewritten)
        if date_range:
            rewritten = normalized_query
            constraints['date_range'] = date_range
            techniques_used.append("date_normalization")
        
        # Step 4: Decomposition for complex queries
        if query_type == QueryType.COMPLEX:
            decomposed = await self.decomposer.llm_based_decomposition(rewritten)
            if len(decomposed.get('sub_queries', [])) > 1:
                sub_queries = decomposed['sub_queries']
                techniques_used.append("decomposition")
                metadata['main_intent'] = decomposed.get('main_intent')
        
        # Step 5: Stopword removal for simple queries
        if query_type == QueryType.SIMPLE:
            cleaned = self.stopword_remover.remove_stopwords(rewritten)
            if cleaned != rewritten:
                rewritten = cleaned
                techniques_used.append("stopword_removal")
        
        # Step 6: Handle ambiguity
        if query_type == QueryType.AMBIGUOUS:
            disambiguated = await self.disambiguate_query(rewritten)
            if disambiguated != rewritten:
                rewritten = disambiguated
                techniques_used.append("disambiguation")
        
        # Step 7: Generate final search queries
        final_search_queries = sub_queries if len(sub_queries) > 1 else [rewritten]
        
        return RewrittenQuery(
            original=query,
            rewritten=rewritten,
            technique_used=", ".join(techniques_used),
            sub_queries=final_search_queries,
            constraints=constraints,
            metadata=metadata
        )
    
    async def disambiguate_query(self, query: str) -> str:
        """Resolve ambiguous terms using context"""
        
        prompt = f"""Disambiguate this ambiguous query: "{query}"

Common ambiguities in this context:
- "apple" → electronic device brand OR fruit
- "java" → programming language OR coffee
- "spring" → season OR framework
- "windows" → OS OR glass panes

Return the disambiguated version based on typical e-commerce context.
If no ambiguity found, return original query.
Output only the rewritten query:"""
        
        response = self.llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
    
    async def rewrite_for_retrieval(self, query: str,
                                   conversation_history: List[Dict] = None,
                                   use_hyde: bool = False,
                                   use_multi_query: bool = False) -> List[str]:
        """Generate retrieval-optimized queries"""
        
        # Standard rewriting
        rewritten = await self.rewrite(query, conversation_history)
        
        retrieval_queries = rewritten.sub_queries.copy()
        
        # Optional: HyDE for better matching
        if use_hyde:
            hyde_doc = await self.hyde_rewriter.generate_hypothetical_document(
                rewritten.rewritten, domain="ecommerce"
            )
            retrieval_queries.append(hyde_doc)
        
        # Optional: Multi-query expansion
        if use_multi_query:
            variations = await self.multi_rewriter.generate_variations(
                rewritten.rewritten, num_variations=2
            )
            retrieval_queries.extend(variations)
        
        return list(set(retrieval_queries))  # Deduplicate

# # Complete usage example
# async def main():
#     # Initialize
#     pipeline = QueryRewritingPipeline(llm_client=groq_client)
    
#     # Example 1: Conversational
#     history = [
#         {"role": "user", "content": "Show me gaming laptops"},
#         {"role": "assistant", "content": "Here are our top gaming laptops..."}
#     ]
#     result = await pipeline.rewrite("How much is the second one?", history)
#     print(f"Rewritten: {result.rewritten}")
#     print(f"Techniques: {result.technique_used}")
#     print(f"Sub-queries: {result.sub_queries}")
    
#     # Example 2: Constrained query
#     result = await pipeline.rewrite("wireless headphones under $100 with 4 stars")
#     print(f"Constraints: {result.constraints}")
#     print(f"Clean query: {result.rewritten}")
    
#     # Example 3: Full retrieval pipeline
#     retrieval_queries = await pipeline.rewrite_for_retrieval(
#         query="best budget gaming laptop last month",
#         conversation_history=None,
#         use_hyde=True,
#         use_multi_query=True
#     )
#     print(f"Retrieval queries: {retrieval_queries}")

__all__ = ['QueryRewritingPipeline', 'RewrittenQuery', 'QueryType']