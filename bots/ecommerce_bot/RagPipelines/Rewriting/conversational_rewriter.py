from typing import List, Dict, Optional

class ConversationalRewriter:
    """Resolve coreferences and ellipsis using conversation history"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def detect_pronouns(self, query: str) -> List[str]:
        """Identify pronouns that need resolution"""
        pronouns = ['it', 'they', 'them', 'that', 'those', 
                   'these', 'this', 'there', 'their', 'its']
        found = [p for p in pronouns if f' {p} ' in f' {query.lower()} ']
        return found
    
    def extract_recent_context(self, history: List[Dict], max_turns: int = 3) -> str:
        """Extract recent conversation context"""
        recent = history[-max_turns:] if len(history) > max_turns else history
        context = []
        
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")[:200]  # Limit length
            context.append(f"{role}: {content}")
        
        return "\n".join(context)
    
    async def rewrite_with_llm(self, query: str, history: List[Dict]) -> str:
        """Use LLM to resolve conversational dependencies"""
        
        if not history or len(history) < 2:
            return query
        
        # Quick check - if no pronouns/ellipsis, return early
        if not self.detect_pronouns(query) and len(query.split()) > 5:
            if not any(word in query.lower() for word in ['it', 'that', 'they']):
                return query
        
        context = self.extract_recent_context(history)
        
        prompt = f"""Conversation history:
{context}

Current user query: "{query}"

Task: Rewrite the current query to be self-contained by:
1. Replacing pronouns (it, they, that, these, those) with their referents
2. Filling in missing context from previous turns
3. Making the query understandable without the history

Rules:
- Don't change the core intent
- Don't add information not implied by history
- Keep the same question format
- Return ONLY the rewritten query, nothing else

Rewritten query:"""
        
        response = self.llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150
        )
        
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else query
    
    def rule_based_fallback(self, query: str, last_assistant_response: str) -> str:
        """Simple rule-based fallback when LLM unavailable"""
        
        replacements = {
            'it': last_assistant_response[:30] if last_assistant_response else 'it',
            'that': last_assistant_response[:30] if last_assistant_response else 'that',
            'this': last_assistant_response[:30] if last_assistant_response else 'this',
            'the second one': 'the previously mentioned item',
            'the first one': 'the first mentioned item'
        }
        
        rewritten = query
        for pronoun, replacement in replacements.items():
            if pronoun in rewritten.lower():
                rewritten = rewritten.lower().replace(pronoun, replacement)
        
        return rewritten

# Usage
# rewriter = ConversationalRewriter(llm_client=groq_client)
# history = [
#     {"role": "user", "content": "Tell me about MacBook Pro M3"},
#     {"role": "assistant", "content": "The MacBook Pro M3 features a 14-inch display..."}
# ]
# new_query = "How much does it cost?"
# rewritten = await rewriter.rewrite_with_llm(new_query, history)
# # Output: "How much does the MacBook Pro M3 cost?"