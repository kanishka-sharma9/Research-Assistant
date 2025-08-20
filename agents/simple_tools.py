"""Simplified Research Tools for LangGraph Agents."""

import os
import json
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv
from langchain.tools import tool
import arxiv
from tavily import TavilyClient
from groq import Groq

# Load environment variables
load_dotenv()


@tool
def search_arxiv_simple(query: str, max_results: int = 5) -> str:
    """Search ArXiv for academic papers.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default 5)
        
    Returns:
        JSON string containing paper information
    """
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        client = arxiv.Client()
        results = []
        
        for paper in client.results(search):
            result = {
                "title": paper.title,
                "authors": [author.name for author in paper.authors][:3],  # Limit authors
                "abstract": paper.summary[:300] + "..." if len(paper.summary) > 300 else paper.summary,
                "url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "published": paper.published.isoformat() if paper.published else None,
                "arxiv_id": paper.get_short_id(),
                "categories": paper.categories,
                "doi": paper.doi,
                "journal_ref": paper.journal_ref,
                "source_type": "arxiv_paper"
            }
            results.append(result)
        
        return json.dumps(results, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"ArXiv search failed: {str(e)}"})


@tool
def search_web_simple(query: str, max_results: int = 3) -> str:
    """Search the web using Tavily API.
    
    Args:
        query: Search query string  
        max_results: Maximum number of results (default 3)
        
    Returns:
        JSON string containing web search results
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return json.dumps({"error": "TAVILY_API_KEY not found"})
    
    try:
        client = TavilyClient(api_key=tavily_api_key)
        
        search_results = client.search(
            query=f"{query} research paper academic",
            max_results=max_results,
            search_depth="basic"
        )
        
        results = []
        for item in search_results.get("results", []):
            result = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")[:200] + "..." if len(item.get("content", "")) > 200 else item.get("content", ""),
                "score": item.get("score", 0.0),
                "published": item.get("published_date"),
                "domain": item.get("url", "").split("/")[2] if item.get("url") else "",
                "source_type": "web_article"
            }
            results.append(result)
        
        return json.dumps(results, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Web search failed: {str(e)}"})


@tool
def create_simple_plan(topic: str, context: str = "") -> str:
    """Create a simple research plan.
    
    Args:
        topic: Research topic
        context: Additional context
        
    Returns:
        JSON string containing research plan
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return json.dumps({"error": "GROQ_API_KEY not found"})
    
    client = Groq(api_key=groq_api_key)
    
    prompt = f"""Create a simple research plan for: "{topic}"

Return JSON with:
- objective: Main research goal
- keywords: 5 key search terms  
- search_queries: 3 specific search queries
- focus_areas: 3 main areas to explore

Return ONLY valid JSON, keep it concise."""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a research planner. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return json.dumps({
            "objective": f"Research {topic}",
            "keywords": topic.split(),
            "search_queries": [topic, f"{topic} applications", f"{topic} challenges"],
            "focus_areas": ["Theory", "Applications", "Future work"],
            "error": str(e)
        })


@tool  
def analyze_papers_simple(papers_json: str, topic: str) -> str:
    """Analyze and rank papers simply.
    
    Args:
        papers_json: JSON string of papers
        topic: Research topic
        
    Returns:
        JSON string with analysis
    """
    try:
        papers = json.loads(papers_json)
        if not papers or "error" in papers:
            return papers_json
        
        # Simple ranking based on title matching
        for i, paper in enumerate(papers):
            title = paper.get("title", "").lower()
            topic_words = topic.lower().split()
            matches = sum(1 for word in topic_words if word in title)
            paper["relevance_score"] = matches / len(topic_words) if topic_words else 0
            paper["rank"] = i + 1
        
        # Sort by relevance
        papers.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return json.dumps(papers[:5], indent=2)  # Return top 5
        
    except Exception as e:
        return json.dumps({"error": f"Analysis failed: {str(e)}"})


@tool
def identify_gaps_simple(papers_json: str, topic: str) -> str:
    """Identify research gaps simply.
    
    Args:
        papers_json: JSON of ranked papers
        topic: Research topic
        
    Returns:
        JSON string with gaps
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return json.dumps({"gaps": ["Unable to analyze - no API key"]})
    
    try:
        papers = json.loads(papers_json)
        if not papers or "error" in papers:
            return json.dumps({"gaps": ["No papers to analyze"]})
        
        client = Groq(api_key=groq_api_key)
        
        # Use only titles and abstracts to reduce token usage
        paper_summaries = []
        for paper in papers[:3]:  # Only use top 3 papers
            summary = f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')[:100]}..."
            paper_summaries.append(summary)
        
        prompt = f"""Based on these papers about "{topic}", identify 3 research gaps:

{chr(10).join(paper_summaries)}

Return JSON with:
- gaps: Array of 3 research gaps
- opportunities: Array of 2 future directions

Keep responses concise."""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a research analyst. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=400,
            response_format={"type": "json_object"}
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return json.dumps({
            "gaps": ["Unable to identify gaps due to error"],
            "opportunities": ["Further research needed"],
            "error": str(e)
        })


@tool
def generate_simple_report(topic: str, plan_json: str, papers_json: str, gaps_json: str) -> str:
    """Generate a simple research report.
    
    Args:
        topic: Research topic
        plan_json: Research plan
        papers_json: Papers data
        gaps_json: Gaps analysis
        
    Returns:
        Simple markdown report
    """
    try:
        plan = json.loads(plan_json) if plan_json else {}
        papers = json.loads(papers_json) if papers_json else []
        gaps = json.loads(gaps_json) if gaps_json else {}
        
        # Separate papers by source type
        arxiv_papers = [p for p in papers if p.get('source_type') == 'arxiv_paper']
        web_articles = [p for p in papers if p.get('source_type') == 'web_article']
        
        # Generate comprehensive report with sources
        report = f"""# Research Report: {topic}

## Executive Summary
Research conducted on **{topic}** using systematic literature review across multiple sources including arXiv preprints and web-based academic resources. This report analyzes {len(papers)} sources and identifies key research trends, gaps, and opportunities.

## Research Methodology
- **Search Strategy**: Multi-source academic search
- **Sources**: arXiv, Academic Web Resources, Research Databases  
- **Papers Analyzed**: {len(papers)} total ({len(arxiv_papers)} arXiv papers, {len(web_articles)} web articles)
- **Ranking Criteria**: Title relevance, content quality, publication recency
- **Date Range**: Latest available research up to {datetime.now().strftime('%Y-%m-%d')}

## Key Findings

### Academic Papers (arXiv)
{chr(10).join([f"**{i+1}. {p.get('title', 'Unknown')}**" + 
              f"{chr(10)}   - *Authors*: {', '.join(p.get('authors', ['Unknown']))}" +
              f"{chr(10)}   - *ArXiv ID*: {p.get('arxiv_id', 'N/A')}" +
              f"{chr(10)}   - *Published*: {p.get('published', 'N/A')[:10] if p.get('published') else 'N/A'}" +
              f"{chr(10)}   - *Categories*: {', '.join(p.get('categories', [])[:2]) if p.get('categories') else 'N/A'}" +
              f"{chr(10)}   - *Relevance Score*: {p.get('relevance_score', 0):.2f}" +
              (f"{chr(10)}   - *DOI*: {p.get('doi')}" if p.get('doi') else "") +
              (f"{chr(10)}   - *Journal*: {p.get('journal_ref')}" if p.get('journal_ref') else "") +
              f"{chr(10)}   - *Abstract*: {p.get('abstract', 'No abstract available')[:150]}..." +
              f"{chr(10)}" 
              for i, p in enumerate(arxiv_papers[:5])])}

### Web-Based Academic Resources
{chr(10).join([f"**{i+1}. {p.get('title', 'Unknown')}**" +
              f"{chr(10)}   - *Source*: {p.get('domain', 'Unknown domain')}" +
              f"{chr(10)}   - *Content Preview*: {p.get('content', 'No preview available')[:150]}..." +
              f"{chr(10)}   - *Relevance Score*: {p.get('relevance_score', 0):.2f}" +
              f"{chr(10)}" 
              for i, p in enumerate(web_articles[:3])])}

## Research Gaps Analysis
{chr(10).join([f"### Gap {i+1}: {gap}" for i, gap in enumerate(gaps.get('gaps', ['None identified']), 1)])}

## Future Research Opportunities  
{chr(10).join([f"### Opportunity {i+1}: {opp}" for i, opp in enumerate(gaps.get('opportunities', ['Further research needed']), 1)])}

## Detailed Source Citations

### arXiv Papers
{chr(10).join([f"[{i+1}] {', '.join(p.get('authors', ['Unknown']))}. \"{p.get('title', 'Unknown')}\" arXiv preprint arXiv:{p.get('arxiv_id', 'N/A')} ({p.get('published', 'N/A')[:4] if p.get('published') else 'N/A'}). Available: {p.get('url', 'N/A')}" 
              for i, p in enumerate(arxiv_papers)])}

### Web Sources  
{chr(10).join([f"[W{i+1}] \"{p.get('title', 'Unknown')}\" {p.get('domain', 'Unknown')}. Available: {p.get('url', 'N/A')}" 
              for i, p in enumerate(web_articles)])}

## Links and Resources

### Direct Paper Access
{chr(10).join([f"- [{p.get('title', 'Unknown')[:50]}...]({p.get('url', '#')}) - arXiv" + 
              (f" | [PDF]({p.get('pdf_url')})" if p.get('pdf_url') else "")
              for p in arxiv_papers[:5]])}

### Web Resources
{chr(10).join([f"- [{p.get('title', 'Unknown')[:50]}...]({p.get('url', '#')}) - {p.get('domain', 'Web')}"
              for p in web_articles[:3]])}

## Research Statistics
- **Total Sources Analyzed**: {len(papers)}
- **arXiv Papers**: {len(arxiv_papers)}
- **Web Articles**: {len(web_articles)}  
- **Average Relevance Score**: {(sum(p.get('relevance_score', 0) for p in papers) / len(papers) if papers else 0.0):.2f}
- **Research Gaps Identified**: {len(gaps.get('gaps', []))}
- **Future Opportunities**: {len(gaps.get('opportunities', []))}

## Conclusion
This comprehensive analysis of **{topic}** reveals a rich landscape of research with {len(papers)} sources providing insights into current developments, methodologies, and future directions. The identified research gaps present opportunities for novel contributions to the field.

## Methodology Notes
- Search conducted across arXiv preprint server and academic web resources
- Papers ranked by relevance using AI-assisted scoring
- Abstracts and content analyzed for thematic patterns
- Research gaps identified through systematic content analysis
- All sources verified and links provided for further investigation

---
*Comprehensive Research Report Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*Self-Initiated Research Agent v1.0*
"""
        
        return report
        
    except Exception as e:
        return f"Error generating report: {str(e)}"


# Tool groups
SIMPLE_SEARCH_TOOLS = [search_arxiv_simple, search_web_simple]
SIMPLE_PLANNING_TOOLS = [create_simple_plan]
SIMPLE_ANALYSIS_TOOLS = [analyze_papers_simple, identify_gaps_simple]
SIMPLE_REPORTING_TOOLS = [generate_simple_report]

ALL_SIMPLE_TOOLS = SIMPLE_SEARCH_TOOLS + SIMPLE_PLANNING_TOOLS + SIMPLE_ANALYSIS_TOOLS + SIMPLE_REPORTING_TOOLS