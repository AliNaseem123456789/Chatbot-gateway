from typing import List, Dict
class HyDERewriter:
    """Generate hypothetical answers for better retrieval"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def generate_hypothetical_document(self, query: str, 
                                            domain: str = "general") -> str:
        """Generate a fake document that would answer the query"""
        
        domain_prompts = {
            "ecommerce": "Write a product description including specifications, features, and use cases.",
            "medical": "Write a medical explanation using proper terminology and clinical details.",
            "legal": "Write a legal clause or statute excerpt with formal legal language.",
            "technical": "Write technical documentation with precise specifications and parameters.",
            "general": "Write an informative paragraph answering the question thoroughly."
        }
        
        prompt = f"""Question: "{query}"

Task: Generate a hypothetical document that would contain the answer to this question.
{domain_prompts.get(domain, domain_prompts["general"])}

Write in natural, detailed prose as if from a real document.
Length: 100-200 words.
Do NOT mention this is hypothetical. Write directly as if answering.
"""
        
        response = self.llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    async def hyde_retrieval(self, query: str, retriever_function, 
                           domain: str = "general", top_k: int = 5) -> List[Dict]:
        """Complete HyDE pipeline: generate doc → embed → retrieve"""
        
        # Step 1: Generate hypothetical document
        hypothetical_doc = await self.generate_hypothetical_document(query, domain)
        
        # Step 2: Use this document for retrieval (not the original query)
        # The retriever function should accept text and return relevant docs
        retrieved_docs = await retriever_function(hypothetical_doc, limit=top_k)
        
        return retrieved_docs

# # Usage for e-commerce bot
# hyde = HyDERewriter(llm_client=groq_client)

# # Without HyDE: "gaming laptop that doesn't overheat"
# # With HyDE generates: "The ASUS ROG Strix features advanced liquid metal cooling 
# # technology that maintains optimal temperatures under load, preventing thermal 
# # throttling during extended gaming sessions. The vapor chamber design..."

# async def search_with_hyde(query):
#     return await hyde.hyde_retrieval(
#         query=query,
#         retriever_function=your_hybrid_search,
#         domain="ecommerce",
#         top_k=5
#     )