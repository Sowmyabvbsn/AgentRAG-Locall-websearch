import re
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class CitationManager:
    """Manages citations and external link recommendations"""
    
    def __init__(self):
        self.ddgs = DDGS()
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for citation search"""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Extract words and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        key_terms = [word for word in words if word not in stop_words]
        
        # Get most frequent terms (limit to top 10)
        from collections import Counter
        term_counts = Counter(key_terms)
        return [term for term, count in term_counts.most_common(10)]
    
    def search_related_content(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search for related content using DuckDuckGo"""
        try:
            results = self.ddgs.text(query, max_results=max_results)
            citations = []
            
            for result in results:
                citation = {
                    'title': result.get('title', 'No title'),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', 'No description'),
                    'source': self._extract_domain(result.get('href', ''))
                }
                citations.append(citation)
            
            return citations
            
        except Exception as e:
            logger.error(f"Error searching for citations: {e}")
            return []
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return "Unknown source"
    
    def generate_citations_for_response(self, response_text: str, original_query: str) -> Dict[str, Any]:
        """Generate citations and related links for a response"""
        # Extract key terms from both response and query
        response_terms = self.extract_key_terms(response_text)
        query_terms = self.extract_key_terms(original_query)
        
        # Combine and prioritize terms
        all_terms = list(set(query_terms + response_terms))
        
        citations = {
            'primary_sources': [],
            'related_links': [],
            'suggested_searches': []
        }
        
        # Search for primary sources based on main query
        primary_citations = self.search_related_content(original_query, max_results=3)
        citations['primary_sources'] = primary_citations
        
        # Search for related content based on key terms
        if all_terms:
            related_query = " ".join(all_terms[:5])  # Use top 5 terms
            related_citations = self.search_related_content(related_query, max_results=4)
            citations['related_links'] = related_citations
        
        # Generate suggested searches
        citations['suggested_searches'] = self._generate_suggested_searches(all_terms, original_query)
        
        return citations
    
    def _generate_suggested_searches(self, terms: List[str], original_query: str) -> List[str]:
        """Generate suggested search queries"""
        suggestions = []
        
        # Add variations of the original query
        suggestions.append(f"{original_query} explained")
        suggestions.append(f"{original_query} examples")
        suggestions.append(f"how to {original_query}")
        
        # Add term-based suggestions
        if len(terms) >= 2:
            suggestions.append(f"{terms[0]} and {terms[1]}")
            suggestions.append(f"{terms[0]} vs {terms[1]}")
        
        # Add specific domain suggestions
        if terms:
            suggestions.append(f"{terms[0]} research papers")
            suggestions.append(f"{terms[0]} case studies")
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def format_citations_for_display(self, citations: Dict[str, Any]) -> str:
        """Format citations for display in the response"""
        formatted = "\n\n## 📚 Sources and Further Reading\n\n"
        
        # Primary sources
        if citations['primary_sources']:
            formatted += "### 🎯 Primary Sources:\n"
            for i, citation in enumerate(citations['primary_sources'], 1):
                formatted += f"{i}. **[{citation['title']}]({citation['url']})**\n"
                formatted += f"   *{citation['source']}* - {citation['snippet'][:100]}...\n\n"
        
        # Related links
        if citations['related_links']:
            formatted += "### 🔗 Related Resources:\n"
            for i, citation in enumerate(citations['related_links'], 1):
                formatted += f"{i}. **[{citation['title']}]({citation['url']})**\n"
                formatted += f"   *{citation['source']}* - {citation['snippet'][:100]}...\n\n"
        
        # Suggested searches
        if citations['suggested_searches']:
            formatted += "### 🔍 Suggested Searches:\n"
            for search in citations['suggested_searches']:
                formatted += f"- {search}\n"
        
        return formatted