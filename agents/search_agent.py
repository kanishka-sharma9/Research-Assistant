"""Search Agent - Handles paper search using ArXiv, Tavily, and web search."""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
from dotenv import load_dotenv
from groq import Groq

# LangChain community tools
from langchain_community.tools import ArxivQueryRun, TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper
from tavily import TavilyClient
import arxiv

# Load environment variables
load_dotenv()


class SearchAgent:
    """Agent responsible for searching and retrieving research papers."""
    
    def __init__(self):
        """Initialize the Search Agent with necessary API keys."""
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required. Please set it in your .env file.")
        
        self.groq_client = Groq(api_key=self.groq_api_key)
        
        # Initialize search tools
        self._initialize_search_tools()
        
        # Cache for storing results
        self.search_cache = {}
        
    def _initialize_search_tools(self):
        """Initialize all search tools."""
        # ArXiv tool setup
        self.arxiv_wrapper = ArxivAPIWrapper(
            top_k_results=10,
            doc_content_chars_max=4000
        )
        self.arxiv_tool = ArxivQueryRun(api_wrapper=self.arxiv_wrapper)
        
        # Direct arxiv client for more control
        self.arxiv_client = arxiv.Client()
        
        # Tavily search setup (if API key available)
        if self.tavily_api_key:
            self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
            self.tavily_tool = TavilySearchResults(
                api_key=self.tavily_api_key,
                max_results=10,
                search_depth="advanced",
                include_domains=[],
                exclude_domains=[]
            )
        else:
            self.tavily_client = None
            self.tavily_tool = None
            print("Warning: Tavily API key not provided. Web search will be limited.")
    
    async def search(
        self, 
        query: str, 
        sources: List[str] = ["arxiv", "web"],
        max_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute search across specified sources.
        
        Args:
            query: Search query string
            sources: List of sources to search ["arxiv", "web", "scholar"]
            max_results: Maximum number of results per source
            filters: Optional filters (year range, etc.)
            
        Returns:
            List of search results with metadata
        """
        all_results = []
        
        # Check cache first
        cache_key = f"{query}_{sources}_{max_results}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # Search each source
        tasks = []
        if "arxiv" in sources:
            tasks.append(self._search_arxiv(query, max_results, filters))
        if "web" in sources and self.tavily_tool:
            tasks.append(self._search_web(query, max_results, filters))
        if "scholar" in sources:
            tasks.append(self._search_scholar(query, max_results, filters))
        
        # Execute searches concurrently
        if tasks:
            results_lists = await asyncio.gather(*tasks, return_exceptions=True)
            for results in results_lists:
                if isinstance(results, list):
                    all_results.extend(results)
                elif isinstance(results, Exception):
                    print(f"Search error: {results}")
        
        # Deduplicate results
        all_results = self._deduplicate_results(all_results)
        
        # Cache results
        self.search_cache[cache_key] = all_results
        
        return all_results
    
    async def _search_arxiv(
        self, 
        query: str, 
        max_results: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search ArXiv for papers.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            filters: Optional filters
            
        Returns:
            List of paper dictionaries
        """
        results = []
        
        try:
            # Use arxiv client for detailed search
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            
            for paper in self.arxiv_client.results(search):
                # Extract all relevant information
                result = {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary,
                    "url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "published": paper.published.isoformat() if paper.published else None,
                    "updated": paper.updated.isoformat() if paper.updated else None,
                    "categories": paper.categories,
                    "primary_category": paper.primary_category,
                    "source": "arxiv",
                    "arxiv_id": paper.get_short_id(),
                    "comment": paper.comment,
                    "journal_ref": paper.journal_ref,
                    "doi": paper.doi,
                    "relevance_score": 0.0  # Will be calculated later
                }
                
                # Apply filters if provided
                if filters:
                    if not self._apply_filters(result, filters):
                        continue
                
                results.append(result)
                
        except Exception as e:
            print(f"ArXiv search error: {e}")
            # Fallback to LangChain tool
            try:
                langchain_results = self.arxiv_tool.run(query)
                # Parse the string results into structured format
                if langchain_results:
                    parsed = self._parse_arxiv_langchain_results(langchain_results, max_results)
                    results.extend(parsed)
            except Exception as e2:
                print(f"ArXiv fallback search error: {e2}")
        
        return results
    
    async def _search_web(
        self, 
        query: str, 
        max_results: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search web using Tavily.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            filters: Optional filters
            
        Returns:
            List of web search results
        """
        results = []
        
        if not self.tavily_client:
            return results
        
        try:
            # Add academic search terms to improve results
            academic_query = f"{query} research paper academic study"
            
            # Use Tavily for web search
            search_results = self.tavily_client.search(
                query=academic_query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False,
                include_images=False
            )
            
            for item in search_results.get("results", []):
                result = {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                    "published": item.get("published_date"),
                    "source": "web",
                    "relevance_score": item.get("score", 0.0)
                }
                
                # Apply filters if provided
                if filters:
                    if not self._apply_filters(result, filters):
                        continue
                
                results.append(result)
                
            # Include the answer if available
            if search_results.get("answer"):
                results.insert(0, {
                    "title": "AI-Generated Summary",
                    "content": search_results["answer"],
                    "source": "web_summary",
                    "relevance_score": 1.0
                })
                
        except Exception as e:
            print(f"Web search error: {e}")
            # Fallback to basic web search using aiohttp if needed
            results = await self._fallback_web_search(query, max_results)
        
        return results
    
    async def _search_scholar(
        self, 
        query: str, 
        max_results: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search Google Scholar (using web scraping or API if available).
        
        Args:
            query: Search query
            max_results: Maximum results to return
            filters: Optional filters
            
        Returns:
            List of scholar search results
        """
        results = []
        
        # Note: Google Scholar doesn't have an official API
        # This is a placeholder for potential integration with:
        # 1. Scholarly library (with proxy/delays to avoid blocking)
        # 2. SerpAPI (paid service)
        # 3. Web scraping with appropriate delays
        
        try:
            # For now, use Tavily to search Google Scholar specifically
            if self.tavily_client:
                scholar_query = f"site:scholar.google.com {query}"
                search_results = self.tavily_client.search(
                    query=scholar_query,
                    max_results=max_results,
                    search_depth="advanced"
                )
                
                for item in search_results.get("results", []):
                    result = {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                        "source": "scholar",
                        "relevance_score": item.get("score", 0.0)
                    }
                    
                    if filters:
                        if not self._apply_filters(result, filters):
                            continue
                    
                    results.append(result)
                    
        except Exception as e:
            print(f"Scholar search error: {e}")
        
        return results
    
    async def _fallback_web_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Fallback web search using direct HTTP requests.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of basic web search results
        """
        results = []
        
        # Use DuckDuckGo HTML version as fallback (no API key required)
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'q': f"{query} research paper",
                    'format': 'json',
                    'no_redirect': '1',
                    'no_html': '1'
                }
                
                async with session.get(
                    'https://api.duckduckgo.com/',
                    params=params,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Parse instant answer
                        if data.get('AbstractText'):
                            results.append({
                                "title": data.get('Heading', 'Summary'),
                                "content": data['AbstractText'],
                                "url": data.get('AbstractURL', ''),
                                "source": "web_fallback",
                                "relevance_score": 0.8
                            })
                        
                        # Parse related topics
                        for topic in data.get('RelatedTopics', [])[:max_results]:
                            if isinstance(topic, dict) and topic.get('Text'):
                                results.append({
                                    "title": topic.get('Text', '')[:100],
                                    "content": topic.get('Text', ''),
                                    "url": topic.get('FirstURL', ''),
                                    "source": "web_fallback",
                                    "relevance_score": 0.5
                                })
                                
        except Exception as e:
            print(f"Fallback web search error: {e}")
        
        return results
    
    def _parse_arxiv_langchain_results(self, text_results: str, max_results: int) -> List[Dict[str, Any]]:
        """Parse text results from LangChain ArXiv tool.
        
        Args:
            text_results: String results from LangChain
            max_results: Maximum number of results
            
        Returns:
            List of parsed paper dictionaries
        """
        results = []
        
        # Basic parsing of text results
        # The format is typically: "Title: ... Authors: ... Summary: ..."
        papers = text_results.split("\n\n")[:max_results]
        
        for paper_text in papers:
            if not paper_text.strip():
                continue
                
            result = {
                "source": "arxiv",
                "relevance_score": 0.0
            }
            
            lines = paper_text.split("\n")
            for line in lines:
                if line.startswith("Title:"):
                    result["title"] = line.replace("Title:", "").strip()
                elif line.startswith("Authors:"):
                    result["authors"] = [a.strip() for a in line.replace("Authors:", "").split(",")]
                elif line.startswith("Summary:"):
                    result["abstract"] = line.replace("Summary:", "").strip()
                elif line.startswith("Published:"):
                    result["published"] = line.replace("Published:", "").strip()
                elif "arxiv.org" in line:
                    result["url"] = line.strip()
            
            if result.get("title"):
                results.append(result)
        
        return results
    
    def _apply_filters(self, result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply filters to a search result.
        
        Args:
            result: Search result dictionary
            filters: Filters to apply
            
        Returns:
            True if result passes filters, False otherwise
        """
        # Year filter
        if "year_from" in filters or "year_to" in filters:
            published = result.get("published")
            if published:
                try:
                    year = int(published[:4]) if isinstance(published, str) else published.year
                    if "year_from" in filters and year < filters["year_from"]:
                        return False
                    if "year_to" in filters and year > filters["year_to"]:
                        return False
                except:
                    pass
        
        # Category filter for ArXiv
        if "categories" in filters and result.get("source") == "arxiv":
            paper_categories = result.get("categories", [])
            if not any(cat in filters["categories"] for cat in paper_categories):
                return False
        
        # Keyword filter
        if "must_include" in filters:
            text = f"{result.get('title', '')} {result.get('abstract', '')} {result.get('content', '')}".lower()
            for keyword in filters["must_include"]:
                if keyword.lower() not in text:
                    return False
        
        if "must_exclude" in filters:
            text = f"{result.get('title', '')} {result.get('abstract', '')} {result.get('content', '')}".lower()
            for keyword in filters["must_exclude"]:
                if keyword.lower() in text:
                    return False
        
        return True
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on title similarity.
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated list of results
        """
        seen_titles = set()
        unique_results = []
        
        for result in results:
            title = result.get("title", "").lower().strip()
            if not title or title in seen_titles:
                continue
            
            # Also check for very similar titles (differ by only a few characters)
            is_duplicate = False
            for seen_title in seen_titles:
                if self._similar_titles(title, seen_title):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_titles.add(title)
                unique_results.append(result)
        
        return unique_results
    
    def _similar_titles(self, title1: str, title2: str, threshold: float = 0.9) -> bool:
        """Check if two titles are similar using simple character comparison.
        
        Args:
            title1: First title
            title2: Second title
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if titles are similar
        """
        if not title1 or not title2:
            return False
        
        # Simple character-based similarity
        longer = max(len(title1), len(title2))
        if longer == 0:
            return True
        
        # Count matching characters
        matches = sum(c1 == c2 for c1, c2 in zip(title1, title2))
        similarity = matches / longer
        
        return similarity >= threshold
    
    async def analyze_paper_relevance(
        self, 
        papers: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Analyze and score paper relevance using Groq.
        
        Args:
            papers: List of papers to analyze
            query: Original search query
            
        Returns:
            Papers with updated relevance scores
        """
        if not papers:
            return papers
        
        # Batch papers for analysis
        batch_size = 5
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            
            prompt = f"""
            Score the relevance of these papers to the research query: "{query}"
            
            Papers:
            {json.dumps([{
                'title': p.get('title'),
                'abstract': p.get('abstract', p.get('content', ''))[:500]
            } for p in batch], indent=2)}
            
            For each paper, provide a relevance score from 0-1 and a brief justification.
            Return as JSON: [{{"index": 0, "score": 0.95, "reason": "..."}}]
            """
            
            try:
                response = self.groq_client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": "You are an expert at assessing research paper relevance."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
                
                scores = json.loads(response.choices[0].message.content)
                if "scores" in scores:
                    scores = scores["scores"]
                
                for score_data in scores:
                    idx = score_data.get("index", 0)
                    if idx < len(batch):
                        batch[idx]["relevance_score"] = score_data.get("score", 0.5)
                        batch[idx]["relevance_reason"] = score_data.get("reason", "")
                        
            except Exception as e:
                print(f"Relevance scoring error: {e}")
                # Default scoring based on title/query match
                for paper in batch:
                    title = paper.get("title", "").lower()
                    query_terms = query.lower().split()
                    matches = sum(1 for term in query_terms if term in title)
                    paper["relevance_score"] = min(matches / len(query_terms), 1.0) if query_terms else 0.5
        
        return papers
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of all searches performed.
        
        Returns:
            Dictionary with search statistics
        """
        total_results = sum(len(results) for results in self.search_cache.values())
        
        return {
            "total_searches": len(self.search_cache),
            "total_results": total_results,
            "sources_used": list(set(
                source for results in self.search_cache.values() 
                for r in results 
                for source in [r.get("source")]
                if source
            )),
            "cache_size": len(self.search_cache),
            "timestamp": datetime.now().isoformat()
        }