import re
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

class ConstraintExtractor:
    """Extract structured filters from natural language queries"""
    
    def __init__(self):
        self.patterns = {
            'price_max': [
                r'under\s+\$?(\d+(?:\.\d+)?)',
                r'less than\s+\$?(\d+(?:\.\d+)?)',
                r'below\s+\$?(\d+(?:\.\d+)?)',
                r'<\s*\$?(\d+(?:\.\d+)?)'
            ],
            'price_min': [
                r'over\s+\$?(\d+(?:\.\d+)?)',
                r'more than\s+\$?(\d+(?:\.\d+)?)',
                r'above\s+\$?(\d+(?:\.\d+)?)',
                r'>\s*\$?(\d+(?:\.\d+)?)'
            ],
            'price_range': [
                r'between\s+\$?(\d+(?:\.\d+)?)\s+and\s+\$?(\d+(?:\.\d+)?)',
                r'from\s+\$?(\d+(?:\.\d+)?)\s+to\s+\$?(\d+(?:\.\d+)?)'
            ],
            'rating': [
                r'(\d+(?:\.\d+)?)\s*stars?(?:\s+and\s+above)?',
                r'rated\s+(\d+(?:\.\d+)?)\s*\+',
                r'minimum\s+(\d+(?:\.\d+)?)\s*stars?'
            ],
            'stock': [
                r'in stock',
                r'available',
                r'out of stock',
                r'backorder'
            ]
        }
    
    def extract_constraints(self, query: str) -> Dict[str, Any]:
        """Extract all constraints from query"""
        constraints = {}
        query_lower = query.lower()
        
        # Price extraction
        # Max price
        for pattern in self.patterns['price_max']:
            match = re.search(pattern, query_lower)
            if match:
                constraints['price_max'] = float(match.group(1))
                break
        
        # Min price
        for pattern in self.patterns['price_min']:
            match = re.search(pattern, query_lower)
            if match:
                constraints['price_min'] = float(match.group(1))
                break
        
        # Price range
        for pattern in self.patterns['price_range']:
            match = re.search(pattern, query_lower)
            if match:
                constraints['price_min'] = float(match.group(1))
                constraints['price_max'] = float(match.group(2))
                break
        
        # Rating
        for pattern in self.patterns['rating']:
            match = re.search(pattern, query_lower)
            if match:
                constraints['min_rating'] = float(match.group(1))
                break
        
        # Stock availability
        if 'in stock' in query_lower or 'available' in query_lower:
            constraints['in_stock'] = True
        elif 'out of stock' in query_lower:
            constraints['in_stock'] = False
        
        return constraints
    
    def extract_temporal(self, query: str) -> Dict[str, Any]:
        """Extract time-based constraints"""
        temporal = {}
        query_lower = query.lower()
        
        today = datetime.now()
        
        if 'today' in query_lower:
            temporal['date_start'] = today.strftime('%Y-%m-%d')
            temporal['date_end'] = today.strftime('%Y-%m-%d')
        elif 'yesterday' in query_lower:
            yesterday = today - timedelta(days=1)
            temporal['date_start'] = yesterday.strftime('%Y-%m-%d')
            temporal['date_end'] = yesterday.strftime('%Y-%m-%d')
        elif 'last week' in query_lower:
            last_week = today - timedelta(days=7)
            temporal['date_start'] = last_week.strftime('%Y-%m-%d')
            temporal['date_end'] = today.strftime('%Y-%m-%d')
        elif 'last month' in query_lower:
            last_month = today - timedelta(days=30)
            temporal['date_start'] = last_month.strftime('%Y-%m-%d')
            temporal['date_end'] = today.strftime('%Y-%m-%d')
        
        return temporal
    
    def extract_categories(self, query: str, category_map: Dict[str, List[str]]) -> List[str]:
        """Extract product categories using mapping"""
        found_categories = []
        query_lower = query.lower()
        
        for category, keywords in category_map.items():
            for keyword in keywords:
                if keyword in query_lower:
                    found_categories.append(category)
                    break
        
        return found_categories
    
    def clean_query_text(self, query: str, constraints: Dict) -> str:
        """Remove constraint phrases from query text for semantic search"""
        cleaned = query
        
        # Remove price constraints
        if 'price_max' in constraints:
            patterns = [f"under ${int(constraints['price_max'])}", 
                       f"less than {int(constraints['price_max'])}"]
            for p in patterns:
                cleaned = cleaned.replace(p, '')
        
        # Remove rating constraints
        if 'min_rating' in constraints:
            cleaned = cleaned.replace(f"{int(constraints['min_rating'])} stars", '')
        
        # Remove temporal constraints
        time_phrases = ['today', 'yesterday', 'last week', 'last month', 'recent']
        for phrase in time_phrases:
            cleaned = cleaned.replace(phrase, '')
        
        return ' '.join(cleaned.split())

# Usage
extractor = ConstraintExtractor()
query = "I need a size 10 pair of shoes under 50 dollars."

constraints = extractor.extract_constraints(query)
# Output: {'price_max': 100.0, 'min_rating': 4.0, 'in_stock': True}

cleaned_query = extractor.clean_query_text(query, constraints)
# Output: "wireless headphones"
print(cleaned_query)
print(constraints)