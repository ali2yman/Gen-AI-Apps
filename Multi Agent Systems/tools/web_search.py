from langchain_core.tools import tool
import requests
import hashlib
import json
import os
import time
from typing import Dict, Any, Optional


class WebSearchTool:
    """Advanced web search tool with caching capabilities."""

    def __init__(self, cache_dir=".cache"):
        """
        Initialize the web search tool.
        Args:
            cache_dir: Directory to store cached search results
        """
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_path(self, query: str) -> str:
        """
        Generate a cache file path for a query.
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"search_{query_hash}.json")

    def _get_cached_result(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result for a query if available and not expired.
        """
        cache_path = self._get_cache_path(query)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            cache_time = cached_data.get('timestamp', 0)
            if time.time() - cache_time < 86400:
                return cached_data.get('results')
        return None

    def _cache_result(self, query: str, results: Dict[str, Any]) -> None:
        """
        Cache search results for future use.
        """
        cache_path = self._get_cache_path(query)
        cache_data = {
            'timestamp': time.time(),
            'query': query,
            'results': results
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

    def _format_results(self, results: Dict[str, Any], query: str) -> str:
        """
        Format search results for readability.
        """
        items = results.get("items", [])
        if not items:
            return f"No results found for: {query}"

        formatted = f"Search results for '{query}':\n\n"
        for i, item in enumerate(items, 1):
            formatted += f"Result {i}:\n"
            formatted += f"Title: {item.get('title', 'No title')}\n"
            formatted += f"Summary: {item.get('snippet', 'No summary available')}\n"
            formatted += f"URL: {item.get('link', 'No link available')}\n\n"
        return formatted

    def search(self, query: str) -> str:
        """
        Search the web for information about a specific query.
        """
        try:
            cached_results = self._get_cached_result(query)
            if cached_results:
                print(f"Using cached results for: {query}")
                return self._format_results(cached_results, query)

            print(f"Performing web search for: {query}")

            # Simulated search results
            results = {
                "items": [
                    {
                        "title": f"Example Result 1 for {query}",
                        "snippet": f"This is a detailed information about {query} with relevant facts and figures.",
                        "link": f"https://example.com/1?q={query.replace(' ', '+')}"
                    },
                    {
                        "title": f"Example Result 2 for {query}",
                        "snippet": f"Additional information about {query} including recent developments and analysis.",
                        "link": f"https://example.com/2?q={query.replace(' ', '+')}"
                    },
                    {
                        "title": f"Example Result 3 for {query}",
                        "snippet": f"Comprehensive guide to {query} with step-by-step instructions and explanations.",
                        "link": f"https://example.com/3?q={query.replace(' ', '+')}"
                    }
                ]
            }

            self._cache_result(query, results)
            return self._format_results(results, query)

        except Exception as e:
            return f"Error performing search: {str(e)}"


# Register as a LangChain tool
@tool("web_search")
def web_search_tool(query: str) -> str:
    """
    Search the web with caching and simulated results.
    """
    return WebSearchTool().search(query)
