"""
Enhanced Decision Engine
Improved logic for determining RAG and search needs
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import re
from src.config import get_settings
from src.utils import get_logger
from src.api import AzureOpenAIClient

logger = get_logger()


class DecisionEngine:
    """Enhanced decision-making for email assistant operations"""
    
    def __init__(self):
        self.settings = get_settings()
        self.azure_client = AzureOpenAIClient()
        
        # Define query patterns
        self.email_search_patterns = [
            r'\b(email|mail|message)s?\b',
            r'\b(find|search|show|list|get|retrieve)\b',
            r'\bfrom\s+[\w\s@\.]+',
            r'\bto\s+[\w\s@\.]+',
            r'\bsubject\b',
            r'\battachment',
            r'\b(inbox|sent|draft|unread)\b',
            r'\b(today|yesterday|last\s+\w+|recent)\b',
            r'\b(\d+\s+days?\s+ago)\b',
            r'\bplacement\s+drive',
            r'\bcompany\s+visit',
            r'\binterview',
            r'\bmeeting'
        ]
        
        self.time_patterns = [
            r'\b(today|yesterday)\b',
            r'\blast\s+(\d+)\s+(day|week|month)s?\b',
            r'\b(\d+)\s+(day|week|month)s?\s+ago\b',
            r'\bin\s+the\s+last\s+\d+\s+(day|week|month)s?\b',
            r'\brecent\b'
        ]
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Comprehensive query analysis
        
        Args:
            query: User query
            
        Returns:
            Analysis results with decisions
        """
        try:
            query_lower = query.lower()
            
            # Pattern-based analysis
            needs_email_search = self._check_email_search_patterns(query_lower)
            has_time_constraint = self._check_time_patterns(query_lower)
            
            # Extract entities
            extracted_time = self._extract_time_constraint(query)
            extracted_sender = self._extract_sender(query)
            extracted_keywords = self._extract_keywords(query)
            
            # LLM-based semantic analysis
            semantic_analysis = await self._semantic_analysis(query)
            
            # Combine all signals
            needs_rag = needs_email_search or semantic_analysis.get('requires_email_data', False)
            needs_search = needs_rag  # If RAG needed, search is needed
            
            # Determine search scope
            search_scope = self._determine_search_scope(
                has_time_constraint=has_time_constraint,
                extracted_time=extracted_time,
                query_lower=query_lower
            )
            
            analysis = {
                'needs_rag': needs_rag,
                'needs_search': needs_search,
                'confidence': self._calculate_confidence(
                    needs_email_search, 
                    semantic_analysis
                ),
                'time_constraint': extracted_time,
                'sender_filter': extracted_sender,
                'keywords': extracted_keywords,
                'search_scope': search_scope,
                'semantic_intent': semantic_analysis.get('intent', 'unknown'),
                'reasoning': self._build_reasoning(
                    needs_email_search,
                    has_time_constraint,
                    semantic_analysis
                )
            }
            
            logger.info(f"Query analysis: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # Default fallback
            return {
                'needs_rag': True,
                'needs_search': True,
                'confidence': 0.5,
                'reasoning': 'Fallback to safe defaults due to analysis error'
            }
    
    def _check_email_search_patterns(self, query_lower: str) -> bool:
        """Check if query matches email search patterns"""
        for pattern in self.email_search_patterns:
            if re.search(pattern, query_lower):
                logger.debug(f"Matched pattern: {pattern}")
                return True
        return False
    
    def _check_time_patterns(self, query_lower: str) -> bool:
        """Check if query has time constraints"""
        for pattern in self.time_patterns:
            if re.search(pattern, query_lower):
                return True
        return False
    
    def _extract_time_constraint(self, query: str) -> Dict[str, Any]:
        """Extract time constraint from query"""
        query_lower = query.lower()
        now = datetime.now()
        
        # Today
        if 'today' in query_lower:
            return {
                'after': now.replace(hour=0, minute=0, second=0).isoformat(),
                'before': None,
                'description': 'today'
            }
        
        # Yesterday
        if 'yesterday' in query_lower:
            yesterday = now - timedelta(days=1)
            return {
                'after': yesterday.replace(hour=0, minute=0, second=0).isoformat(),
                'before': now.replace(hour=0, minute=0, second=0).isoformat(),
                'description': 'yesterday'
            }
        
        # Last N days
        match = re.search(r'last\s+(\d+)\s+days?', query_lower)
        if match:
            days = int(match.group(1))
            after_date = now - timedelta(days=days)
            return {
                'after': after_date.isoformat(),
                'before': None,
                'description': f'last {days} days'
            }
        
        # N days ago
        match = re.search(r'(\d+)\s+days?\s+ago', query_lower)
        if match:
            days = int(match.group(1))
            after_date = now - timedelta(days=days)
            return {
                'after': after_date.isoformat(),
                'before': None,
                'description': f'{days} days ago'
            }
        
        # Recent (default to last 7 days)
        if 'recent' in query_lower:
            after_date = now - timedelta(days=7)
            return {
                'after': after_date.isoformat(),
                'before': None,
                'description': 'recent (7 days)'
            }
        
        return None
    
    def _extract_sender(self, query: str) -> str:
        """Extract sender from query"""
        # Look for email addresses
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', query)
        if email_match:
            return email_match.group(0)
        
        # Look for "from <name>"
        from_match = re.search(r'from\s+([\w\s]+?)(?:\s+about|\s+regarding|\s+with|$)', query, re.IGNORECASE)
        if from_match:
            return from_match.group(1).strip()
        
        return None
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                     'from', 'about', 'show', 'get', 'find', 'list', 'me', 'my', 'i', 'we'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:10]  # Top 10 keywords
    
    async def _semantic_analysis(self, query: str) -> Dict[str, Any]:
        """Use LLM for semantic analysis of query"""
        try:
            analysis_prompt = f"""Analyze this user query and determine:
1. Does it require searching through emails? (yes/no)
2. What is the user's intent? (search_emails, calendar_query, task_management, general_question)
3. What specific information are they looking for?

Query: {query}

Respond in JSON format:
{{
    "requires_email_data": true/false,
    "intent": "search_emails" or "calendar_query" or "task_management" or "general_question",
    "specific_info": "brief description"
}}"""
            
            messages = [
                {"role": "system", "content": "You are a query analysis expert. Always respond with valid JSON."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            response = await self.azure_client.generate_response(
                messages=messages,
                max_tokens=200
            )
            
            import json
            analysis = json.loads(response)
            return analysis
            
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            return {'requires_email_data': True, 'intent': 'search_emails'}
    
    def _determine_search_scope(
        self,
        has_time_constraint: bool,
        extracted_time: Dict[str, Any],
        query_lower: str
    ) -> str:
        """Determine how broad the search should be"""
        if extracted_time:
            return "time_constrained"
        elif has_time_constraint:
            return "recent"
        elif any(word in query_lower for word in ['all', 'every', 'entire']):
            return "broad"
        else:
            return "normal"
    
    def _calculate_confidence(
        self,
        pattern_match: bool,
        semantic_analysis: Dict[str, Any]
    ) -> float:
        """Calculate confidence in decision"""
        confidence = 0.5  # Base confidence
        
        if pattern_match:
            confidence += 0.3
        
        if semantic_analysis.get('requires_email_data'):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _build_reasoning(
        self,
        needs_email_search: bool,
        has_time_constraint: bool,
        semantic_analysis: Dict[str, Any]
    ) -> str:
        """Build human-readable reasoning"""
        reasons = []
        
        if needs_email_search:
            reasons.append("Query matches email search patterns")
        
        if has_time_constraint:
            reasons.append("Query has time constraints")
        
        if semantic_analysis.get('requires_email_data'):
            reasons.append("Semantic analysis indicates email data needed")
        
        intent = semantic_analysis.get('intent', 'unknown')
        reasons.append(f"Intent classified as: {intent}")
        
        return "; ".join(reasons) if reasons else "Using default search behavior"

