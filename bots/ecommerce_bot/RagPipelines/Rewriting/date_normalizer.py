from datetime import datetime, timedelta
import re
from typing import Tuple, Optional

class DateNormalizer:
    """Convert relative dates to absolute ranges"""
    
    def __init__(self, reference_date: datetime = None):
        self.reference_date = reference_date or datetime.now()
    
    def normalize(self, query: str) -> Tuple[str, Optional[Tuple[str, str]]]:
        """Extract and normalize date references in query"""
        
        date_patterns = {
            'today': (0, 0),
            'yesterday': (1, 1),
            'tomorrow': (-1, -1),
            'this_week': (0, 7),
            'last_week': (7, 14),
            'next_week': (-7, 0),
            'this_month': (0, 30),
            'last_month': (30, 60),
            'next_month': (-30, 0),
            'this_year': (0, 365),
            'last_year': (365, 730),
        }
        
        query_lower = query.lower()
        date_range = None
        normalized_query = query
        
        for pattern, (start_delta, end_delta) in date_patterns.items():
            if pattern.replace('_', ' ') in query_lower:
                start_date = self.reference_date - timedelta(days=start_delta)
                end_date = self.reference_date - timedelta(days=end_delta)
                
                # Ensure start < end
                if start_date > end_date:
                    start_date, end_date = end_date, start_date
                
                date_range = (
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                # Replace relative date with absolute
                normalized_query = re.sub(
                    pattern.replace('_', ' '),
                    f"between {date_range[0]} and {date_range[1]}",
                    normalized_query
                )
                break
        
        return normalized_query, date_range
    
    def extract_date_from_query(self, query: str) -> Optional[datetime]:
        """Extract explicit dates from query"""
        
        # Pattern for dates like "May 5, 2026" or "2026-05-05"
        date_patterns = [
            r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
            r'(\w+)\s+(\d{1,2}),?\s+(\d{4})'  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 3:
                        if '-' in pattern:  # YYYY-MM-DD
                            return datetime(int(groups[0]), int(groups[1]), int(groups[2]))
                        elif '/' in pattern:  # MM/DD/YYYY
                            return datetime(int(groups[2]), int(groups[0]), int(groups[1]))
                        else:  # Month DD, YYYY
                            month_map = {
                                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                                'september': 9, 'october': 10, 'november': 11, 'december': 12
                            }
                            month = month_map.get(groups[0].lower(), 1)
                            return datetime(int(groups[2]), month, int(groups[1]))
                except:
                    pass
        
        return None

# Usage
normalizer = DateNormalizer()
query = "show me products released last month"
normalized, date_range = normalizer.normalize(query)
# normalized: "show me products released between 2026-04-06 and 2026-05-06"
# date_range: ('2026-04-06', '2026-05-06')