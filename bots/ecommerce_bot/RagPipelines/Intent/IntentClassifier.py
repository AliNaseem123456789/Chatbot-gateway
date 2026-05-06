# RagPipelines/intent_classifier.py
import re
from typing import Dict, Any, Optional
from enum import Enum


class IntentType(Enum):
    GREETING = "greeting"
    SMALL_TALK = "small_talk"
    PRODUCT_SEARCH = "product_search"
    POLICY_QUESTION = "policy_question"
    COMPLEX_QUERY = "complex_query"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    HELP = "help"
    FAREWELL = "farewell"
    AMBIGUOUS = "ambiguous"

class IntentClassifier:
    """Two-tier intent classifier for query routing"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
        # Tier 1: Rule-based patterns
        self.greeting_patterns = [
            r'^(hi|hello|hey|greetings|sup|yo)($|\s)',
            r'good (morning|afternoon|evening)',
            r'howdy'
        ]
        
        self.small_talk_patterns = [
            r'how are you',
            r'how do you do',
            r'what\'s up',
            r'how\'s it going',
            r'nice to meet you',
            r'thanks|thank you',
            r'good (bot|robot|ai)',
            r'you are (cool|great|awesome|helpful)'
        ]
        
        self.farewell_patterns = [
            r'(bye|goodbye|see you|farewell|catch you later)',
            r'(exit|quit|end)($|\s)',
            r'thank you.*(bye|goodbye)',
            r'have a (good|great|nice) day'
        ]
        
        self.help_patterns = [
            r'^(help|what can you do|how do you work)',
            r'what (can|do) you (do|help with)',
            r'capabilities|features'
        ]
        
        self.product_search_patterns = [
            r'(show|find|get|search for|looking for|want)\s+(me\s+)?(a|an|some)?\s*(\w+)',
            r'price of',
            r'cost of',
            r'available',
            r'in stock',
            r'under \$\d+',
            r'cheap|budget|affordable'
        ]
        
        self.policy_patterns = [
            r'(shipping|return|warranty|payment|cancel|refund|policy)',
            r'how long (does|will)',
            r'delivery time',
            r'free shipping'
        ]
        
        self.comparison_patterns = [
            r'compare|versus|vs\.|better than|worse than',
            r'difference between',
            r'which one (is|should)'
        ]
        
        self.recommendation_patterns = [
            r'recommend|suggest|best|top|good',
            r'what (should|do) you recommend',
            r'which (one|product)'
        ]
        
        self.complex_patterns = [
            r' and .* and ',
            r'\?.*\?',  # multiple question marks
            r'(explain|describe|tell me about).*(and|also|additionally)',
            r'what.*and.*how'
        ]
    
    def classify_rule_based(self, query: str) -> IntentType:
        """Tier 1: Fast rule-based classification"""
        query_lower = query.lower().strip()
        
        # Check patterns in order of priority
        if any(re.match(p, query_lower, re.IGNORECASE) for p in self.greeting_patterns):
            return IntentType.GREETING
        
        if any(re.match(p, query_lower, re.IGNORECASE) for p in self.farewell_patterns):
            return IntentType.FAREWELL
        
        if any(re.search(p, query_lower, re.IGNORECASE) for p in self.small_talk_patterns):
            return IntentType.SMALL_TALK
        
        if any(re.search(p, query_lower, re.IGNORECASE) for p in self.help_patterns):
            return IntentType.HELP
        
        if any(re.search(p, query_lower, re.IGNORECASE) for p in self.comparison_patterns):
            return IntentType.COMPARISON
        
        if any(re.search(p, query_lower, re.IGNORECASE) for p in self.recommendation_patterns):
            return IntentType.RECOMMENDATION
        
        if any(re.search(p, query_lower, re.IGNORECASE) for p in self.complex_patterns):
            return IntentType.COMPLEX_QUERY
        
        if any(re.search(p, query_lower, re.IGNORECASE) for p in self.policy_patterns):
            return IntentType.POLICY_QUESTION
        
        if any(re.search(p, query_lower, re.IGNORECASE) for p in self.product_search_patterns):
            return IntentType.PRODUCT_SEARCH
        
        return IntentType.AMBIGUOUS
    
    async def classify_with_llm(self, query: str) -> IntentType:
        """Tier 2: LLM-based classification for ambiguous queries"""
        if not self.llm_client:
            return IntentType.PRODUCT_SEARCH  # Default fallback
        
        prompt = f"""Classify this user query into ONE category:

Query: "{query}"

Categories:
- greeting: Hello, hi, hey
- small_talk: How are you, thanks, you're helpful
- product_search: Find products, show me laptops, price check
- policy_question: Shipping, returns, warranty, payment
- comparison: Compare X vs Y, difference between
- recommendation: Best, recommend, suggest, top
- complex_query: Multi-part questions with AND/OR
- help: What can you do, how do you work
- farewell: Bye, goodbye, see you

Return ONLY the category name (e.g., "product_search"):"""
        
        response = self.llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Map to IntentType
        intent_map = {
            "greeting": IntentType.GREETING,
            "small_talk": IntentType.SMALL_TALK,
            "product_search": IntentType.PRODUCT_SEARCH,
            "policy_question": IntentType.POLICY_QUESTION,
            "comparison": IntentType.COMPARISON,
            "recommendation": IntentType.RECOMMENDATION,
            "complex_query": IntentType.COMPLEX_QUERY,
            "help": IntentType.HELP,
            "farewell": IntentType.FAREWELL,
        }
        
        return intent_map.get(result, IntentType.PRODUCT_SEARCH)
    
    async def classify(self, query: str, use_llm_for_ambiguous: bool = True) -> tuple[IntentType, str]:
        """Main classification method with two-tier approach"""
        
        # Tier 1: Fast rule-based
        intent = self.classify_rule_based(query)
        
        # Tier 2: If ambiguous and LLM enabled, try LLM
        if intent == IntentType.AMBIGUOUS and use_llm_for_ambiguous:
            intent = await self.classify_with_llm(query)
            return intent, "llm"
        
        return intent, "rule-based"
    
    def should_rewrite_query(self, intent: IntentType) -> bool:
        """Determine if query should go through rewriting pipeline"""
        # Intents that benefit from rewriting
        rewrite_intents = {
            IntentType.PRODUCT_SEARCH,
            IntentType.COMPLEX_QUERY,
            IntentType.COMPARISON,
            IntentType.RECOMMENDATION,
            IntentType.AMBIGUOUS
        }
        
        # Intents that skip rewriting (direct response)
        skip_intents = {
            IntentType.GREETING,
            IntentType.SMALL_TALK,
            IntentType.FAREWELL,
            IntentType.HELP
        }
        
        return intent in rewrite_intents
    
    def get_direct_response(self, intent: IntentType) -> Optional[str]:
        """Get canned response for simple intents"""
        responses = {
            IntentType.GREETING: "Hello!  I'm ShopAssist. How can I help you find products today?",
            IntentType.SMALL_TALK: "I'm doing great, thanks for asking! Ready to help you find some awesome electronics!",
            IntentType.FAREWELL: "Goodbye!  Feel free to come back if you need any help with electronics!",
            IntentType.HELP: "I can help you find products, answer policy questions, compare items, and give recommendations. Just ask! 🛒"
        }
        return responses.get(intent)