# Add this class to your ecommerce_bot.py (after imports)
from typing import Dict,List
import time
from collections import defaultdict
from datetime import datetime, timedelta

class RAGMetrics:
    """Track RAG system performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "successful_retrievals": 0,  # Products found > 0
            "failed_retrievals": 0,       # Products found = 0
            "total_response_time": 0,
            "avg_response_time": 0,
            "intent_distribution": defaultdict(int),
            "techniques_used": defaultdict(int),
            "products_per_query": [],
            "hallucinations_detected": 0,
            "query_lengths": [],
            "hourly_queries": defaultdict(int),
            "last_24h": [],
        }
        self.start_time = datetime.now()
    
    def record_query(self, intent: str, techniques: str, products_found: int, response_time_ms: float, query: str):
        """Record metrics for each query"""
        self.metrics["total_queries"] += 1
        
        if products_found > 0:
            self.metrics["successful_retrievals"] += 1
        else:
            self.metrics["failed_retrievals"] += 1
        
        self.metrics["total_response_time"] += response_time_ms
        self.metrics["avg_response_time"] = self.metrics["total_response_time"] / self.metrics["total_queries"]
        
        self.metrics["intent_distribution"][intent] += 1
        
        if techniques and techniques != "none":
            for technique in techniques.split(", "):
                if technique:
                    self.metrics["techniques_used"][technique] += 1
        
        self.metrics["products_per_query"].append(products_found)
        self.metrics["query_lengths"].append(len(query.split()))
        
        hour = datetime.now().strftime("%Y-%m-%d %H:00")
        self.metrics["hourly_queries"][hour] += 1
        
        # Keep last 24h
        self.metrics["last_24h"].append({
            "timestamp": datetime.now().isoformat(),
            "query": query[:50],
            "products_found": products_found,
            "response_time_ms": response_time_ms,
            "intent": intent
        })
        # Keep only last 1000
        if len(self.metrics["last_24h"]) > 1000:
            self.metrics["last_24h"] = self.metrics["last_24h"][-1000:]
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        success_rate = (self.metrics["successful_retrievals"] / self.metrics["total_queries"] * 100) if self.metrics["total_queries"] > 0 else 0
        
        avg_products = sum(self.metrics["products_per_query"]) / len(self.metrics["products_per_query"]) if self.metrics["products_per_query"] else 0
        
        avg_query_length = sum(self.metrics["query_lengths"]) / len(self.metrics["query_lengths"]) if self.metrics["query_lengths"] else 0
        
        return {
            "total_queries": self.metrics["total_queries"],
            "success_rate": round(success_rate, 2),
            "avg_response_time_ms": round(self.metrics["avg_response_time"], 2),
            "avg_products_per_query": round(avg_products, 2),
            "avg_query_length": round(avg_query_length, 2),
            "intent_distribution": dict(self.metrics["intent_distribution"]),
            "most_used_techniques": dict(sorted(self.metrics["techniques_used"].items(), key=lambda x: x[1], reverse=True)[:5]),
            "uptime_hours": round((datetime.now() - self.start_time).total_seconds() / 3600, 2)
        }
    
    def get_retrieval_accuracy(self, test_queries: List[Dict]) -> Dict:
        """Calculate retrieval accuracy using test queries"""
        correct = 0
        total = 0
        results = []
        
        for test in test_queries:
            # This would be called separately with actual search results
            pass
        
        return {"accuracy": 0, "results": results}

# Add to your EcommerceBot __init__
