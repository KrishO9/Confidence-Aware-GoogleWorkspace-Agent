"""
Search Strategy Planner
Generates smart search strategies based on user queries
Inspired by TinyAgent and LLMCompiler paradigms
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from src.config import get_settings
from src.utils import get_logger
from src.api import AzureOpenAIClient

logger = get_logger()


class SearchStrategyPlanner:
    """Plans and executes smart search strategies"""
    
    def __init__(self):
        self.settings = get_settings()
        self.azure_client = AzureOpenAIClient()
    
    async def plan_search_strategy(self, query: str) -> Dict[str, Any]:
        """
        Generate a smart search strategy for the query
        
        Args:
            query: User query
            
        Returns:
            Search strategy with multiple approaches
        """
        try:
            planning_prompt = f"""You are a search strategy planner for an email assistant.
Given a user query, create a comprehensive search strategy that combines multiple approaches.

User Query: {query}

Generate a search strategy in JSON format with these components:
1. semantic_search: The semantic search query to use
2. date_filters: Date range filters (if applicable)
3. sender_filters: Sender/recipient filters (if applicable)
4. keyword_filters: Specific keywords to search for
5. gmail_query: Gmail-specific query syntax
6. priority_order: List of approaches in order of priority
7. parallel_searches: Which searches can be run in parallel

Example format:
{{
    "semantic_search": "project deadline discussions",
    "date_filters": {{
        "after": "2024-01-01",
        "before": null
    }},
    "sender_filters": ["john@example.com"],
    "keyword_filters": ["deadline", "project", "milestone"],
    "gmail_query": "from:john@example.com subject:(project OR deadline) after:2024/01/01",
    "priority_order": ["semantic", "gmail", "date"],
    "parallel_searches": ["semantic", "date"],
    "explanation": "Search for project-related emails from John with deadline mentions"
}}

Now create the strategy for the query above:"""
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at creating efficient search strategies. Always respond with valid JSON."
                },
                {
                    "role": "user",
                    "content": planning_prompt
                }
            ]
            
            response = await self.azure_client.generate_response(
                messages=messages,
                max_tokens=1000
            )
            
            # Parse JSON response
            strategy = json.loads(response)
            
            # Validate and enhance strategy
            strategy = self._enhance_strategy(strategy, query)
            
            logger.info(f"Generated search strategy: {strategy.get('explanation', '')}")
            return strategy
            
        except Exception as e:
            logger.error(f"Error planning search strategy: {e}")
            # Fallback strategy
            return self._fallback_strategy(query)
    
    def _enhance_strategy(self, strategy: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Enhance and validate search strategy
        
        Args:
            strategy: Generated strategy
            query: Original query
            
        Returns:
            Enhanced strategy
        """
        # Ensure all required fields exist
        defaults = {
            "semantic_search": query,
            "date_filters": {},
            "sender_filters": [],
            "keyword_filters": [],
            "gmail_query": "",
            "priority_order": ["semantic", "gmail"],
            "parallel_searches": ["semantic"],
            "explanation": "Search for relevant emails"
        }
        
        for key, default_value in defaults.items():
            if key not in strategy:
                strategy[key] = default_value
        
        # Parse date filters if they're strings
        if isinstance(strategy.get("date_filters"), dict):
            date_filters = strategy["date_filters"]
            
            # Convert relative dates to absolute
            if "after" in date_filters and isinstance(date_filters["after"], str):
                date_filters["after"] = self._parse_relative_date(date_filters["after"])
            
            if "before" in date_filters and isinstance(date_filters["before"], str):
                date_filters["before"] = self._parse_relative_date(date_filters["before"])
        
        return strategy
    
    def _parse_relative_date(self, date_str: str) -> Optional[str]:
        """
        Parse relative date expressions
        
        Args:
            date_str: Date string (e.g., "today", "last week", "2024-01-01")
            
        Returns:
            ISO format date string
        """
        try:
            date_lower = date_str.lower().strip()
            now = datetime.now()
            
            if date_lower == "today":
                return now.strftime("%Y-%m-%d")
            elif date_lower == "yesterday":
                return (now - timedelta(days=1)).strftime("%Y-%m-%d")
            elif "last week" in date_lower:
                return (now - timedelta(days=7)).strftime("%Y-%m-%d")
            elif "last month" in date_lower:
                return (now - timedelta(days=30)).strftime("%Y-%m-%d")
            elif "days ago" in date_lower:
                # Extract number
                import re
                match = re.search(r'(\d+)\s*days?\s*ago', date_lower)
                if match:
                    days = int(match.group(1))
                    return (now - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Try parsing as ISO date
            from dateutil import parser
            parsed_date = parser.parse(date_str)
            return parsed_date.strftime("%Y-%m-%d")
            
        except Exception as e:
            logger.warning(f"Could not parse date: {date_str}, error: {e}")
            return None
    
    def _fallback_strategy(self, query: str) -> Dict[str, Any]:
        """
        Fallback strategy when planning fails
        
        Args:
            query: User query
            
        Returns:
            Basic search strategy
        """
        return {
            "semantic_search": query,
            "date_filters": {},
            "sender_filters": [],
            "keyword_filters": [],
            "gmail_query": query,
            "priority_order": ["semantic"],
            "parallel_searches": ["semantic"],
            "explanation": "Basic semantic search"
        }
    
    async def generate_search_variations(
        self,
        query: str,
        num_variations: int = 3
    ) -> List[str]:
        """
        Generate query variations for better recall
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations
        """
        try:
            variation_prompt = f"""Generate {num_variations} variations of the following search query.
Each variation should be semantically similar but use different words or phrasings.

Original Query: {query}

Respond with a JSON list of query variations:"""
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a query reformulation expert."
                },
                {
                    "role": "user",
                    "content": variation_prompt
                }
            ]
            
            response = await self.azure_client.generate_response(
                messages=messages,
                max_tokens=500
            )
            
            variations = json.loads(response)
            
            if isinstance(variations, list):
                return variations[:num_variations]
            
            return [query]
            
        except Exception as e:
            logger.error(f"Error generating variations: {e}")
            return [query]
    
    def build_gmail_query(self, strategy: Dict[str, Any]) -> str:
        """
        Build Gmail query string from strategy
        
        Args:
            strategy: Search strategy
            
        Returns:
            Gmail query string
        """
        parts = []
        
        # Add sender filters
        if strategy.get("sender_filters"):
            for sender in strategy["sender_filters"]:
                parts.append(f"from:{sender}")
        
        # Add date filters
        date_filters = strategy.get("date_filters", {})
        if date_filters.get("after"):
            date_str = date_filters["after"].replace("-", "/")
            parts.append(f"after:{date_str}")
        
        if date_filters.get("before"):
            date_str = date_filters["before"].replace("-", "/")
            parts.append(f"before:{date_str}")
        
        # Add keyword filters
        if strategy.get("keyword_filters"):
            keywords = " OR ".join(strategy["keyword_filters"])
            if len(strategy["keyword_filters"]) > 1:
                parts.append(f"({keywords})")
            else:
                parts.append(keywords)
        
        # Combine parts
        if parts:
            return " ".join(parts)
        
        # Fallback to semantic search query
        return strategy.get("semantic_search", "")

