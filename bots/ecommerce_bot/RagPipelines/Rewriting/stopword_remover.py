class StopwordRemover:
    """Remove noise words that don't carry semantic meaning"""
    
    def __init__(self):
        self.stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'he', 'him', 'his',
            'she', 'her', 'hers', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'a', 'an', 'and', 'are',
            'as', 'at', 'be', 'by', 'for', 'from', 'has', 'have', 'he',
            'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this',
            'to', 'was', 'were', 'will', 'with', 'would', 'could',
            'should', 'might', 'must', 'what', 'which', 'who', 'whom'
        }
    
    def remove_stopwords(self, query: str) -> str:
        words = query.lower().split()
        filtered = [w for w in words if w not in self.stopwords]
        return ' '.join(filtered)
    
    def normalize_voice(self, query: str) -> str:
        """Special handling for voice transcription artifacts"""
        fillers = {'um', 'uh', 'like', 'actually', 'basically', 
                   'literally', 'so', 'well', 'you know'}
        words = query.lower().split()
        cleaned = [w for w in words if w not in fillers]
        return ' '.join(cleaned)

# Usage
remover = StopwordRemover()
raw = ""
result = remover.remove_stopwords(raw)
# Output: "buy cheap laptop college work"
print(result)