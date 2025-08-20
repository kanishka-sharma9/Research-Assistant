"""Research Tools for LangGraph Agents."""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.tools import ArxivQueryRun, TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper
from tavily import TavilyClient
import arxiv

# Load environment variables
load_dotenv()


@tool
def search_arxiv(query: str, max_results: int = 10) -> str:
    """Search ArXiv for academic papers.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        JSON string containing paper information
    """
    try:
        # Create arxiv search
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
                "authors": [author.name for author in paper.authors],
                "abstract": paper.summary,
                "url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "published": paper.published.isoformat() if paper.published else None,
                "categories": paper.categories,
                "arxiv_id": paper.get_short_id()
            }
            results.append(result)
        
        return json.dumps(results, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"ArXiv search failed: {str(e)}"})


@tool
def search_web(query: str, max_results: int = 10) -> str:
    """Search the web using Tavily API.
    
    Args:
        query: Search query string  
        max_results: Maximum number of results to return
        
    Returns:
        JSON string containing web search results
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return json.dumps({"error": "TAVILY_API_KEY not found in environment"})
    
    try:
        client = TavilyClient(api_key=tavily_api_key)
        
        # Add academic search terms
        academic_query = f"{query} research paper academic study"
        
        search_results = client.search(
            query=academic_query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=True
        )
        
        results = []
        for item in search_results.get("results", []):
            result = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0.0),
                "published": item.get("published_date")
            }
            results.append(result)
        
        # Include AI-generated answer if available
        if search_results.get("answer"):
            results.insert(0, {
                "title": "AI-Generated Summary",
                "content": search_results["answer"],
                "type": "summary",
                "score": 1.0
            })
        
        return json.dumps(results, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Web search failed: {str(e)}"})


@tool
def create_research_plan(topic: str, context: str = "") -> str:
    """Create a comprehensive research plan for the given topic.
    
    Args:
        topic: Research topic to plan for
        context: Additional context information
        
    Returns:
        JSON string containing detailed research plan
    """
    from groq import Groq
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return json.dumps({"error": "GROQ_API_KEY not found in environment"})
    
    client = Groq(api_key=groq_api_key)
    
    planning_prompt = f"""
    Create a COMPREHENSIVE research plan for: "{topic}"
    
    Context: {context}
    
    Generate a detailed JSON plan with these sections:
    
    1. RESEARCH_OBJECTIVES:
       - primary_question: Main research question
       - secondary_questions: List of 3-5 supporting questions
       - expected_outcomes: What we hope to discover
    
    2. SEARCH_STRATEGY:
       - keywords: List of key terms to search
       - search_queries: List of specific queries with purpose
       - databases: Sources to search (arxiv, web, scholar)
       - filters: Year ranges, categories, etc.
    
    3. METHODOLOGY:
       - evaluation_criteria: How to assess papers
       - ranking_factors: What makes a paper relevant
       - synthesis_approach: How to combine findings
    
    4. TIMELINE:
       - phases: List of research phases with durations
       - milestones: Key checkpoints
    
    5. EXPECTED_CHALLENGES:
       - potential_issues: What might go wrong
       - mitigation_strategies: How to handle problems
    
    Return ONLY valid JSON. Be specific and actionable.
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert research planner. Return only valid JSON."},
                {"role": "user", "content": planning_prompt}
            ],
            temperature=0.3,
            max_tokens=3000,
            response_format={"type": "json_object"}
        )
        
        plan = json.loads(response.choices[0].message.content)
        plan["metadata"] = {
            "created_at": datetime.now().isoformat(),
            "topic": topic
        }
        
        return json.dumps(plan, indent=2)
        
    except Exception as e:
        # Fallback basic plan
        fallback_plan = {
            "research_objectives": {
                "primary_question": f"What is the current state of {topic}?",
                "secondary_questions": [
                    "What are the key methodologies?",
                    "Who are the main contributors?",
                    "What are the current limitations?",
                    "What are future directions?"
                ]
            },
            "search_strategy": {
                "keywords": topic.split(),
                "search_queries": [
                    {"query": topic, "purpose": "General overview"},
                    {"query": f"{topic} survey", "purpose": "Review papers"},
                    {"query": f"{topic} recent advances", "purpose": "Latest developments"}
                ]
            },
            "error": f"Planning failed: {str(e)}"
        }
        return json.dumps(fallback_plan, indent=2)


@tool
def rank_papers(papers_json: str, topic: str) -> str:
    """Rank research papers by relevance to the topic.
    
    Args:
        papers_json: JSON string containing papers to rank
        topic: Original research topic
        
    Returns:
        JSON string with ranked papers and scores
    """
    from groq import Groq
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return json.dumps({"error": "GROQ_API_KEY not found in environment"})
    
    try:
        papers = json.loads(papers_json)
        if not papers or "error" in papers:
            return papers_json
        
        client = Groq(api_key=groq_api_key)
        
        # Take first 10 papers for ranking
        papers_to_rank = papers[:10] if len(papers) > 10 else papers
        
        ranking_prompt = f"""
        Rank these research papers by relevance to: "{topic}"
        
        Papers:
        {json.dumps([{
            'title': p.get('title', ''),
            'abstract': p.get('abstract', p.get('content', ''))[:500],
            'authors': p.get('authors', []),
            'published': p.get('published', '')
        } for p in papers_to_rank], indent=2)}
        
        For each paper, provide:
        - relevance_score: 0.0-1.0 based on relevance to topic
        - key_contributions: Main findings or contributions
        - methodology: Research approach used
        - limitations: What the paper doesn't cover
        
        Return as JSON array with all original paper data plus ranking info.
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert at evaluating research papers. Return valid JSON."},
                {"role": "user", "content": ranking_prompt}
            ],
            temperature=0.2,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        ranking_result = json.loads(response.choices[0].message.content)
        
        # Merge ranking info with original papers
        ranked_papers = []
        rankings = ranking_result.get("papers", ranking_result.get("rankings", []))
        
        for i, paper in enumerate(papers_to_rank):
            if i < len(rankings):
                paper.update(rankings[i])
            ranked_papers.append(paper)
        
        # Sort by relevance score
        ranked_papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return json.dumps(ranked_papers, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Ranking failed: {str(e)}", "original_papers": papers_json})


@tool
def identify_research_gaps(ranked_papers_json: str, topic: str) -> str:
    """Identify gaps in current research based on ranked papers.
    
    Args:
        ranked_papers_json: JSON string of ranked papers
        topic: Original research topic
        
    Returns:
        JSON string containing identified research gaps
    """
    from groq import Groq
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return json.dumps({"error": "GROQ_API_KEY not found in environment"})
    
    try:
        papers = json.loads(ranked_papers_json)
        if not papers or "error" in papers:
            return json.dumps({"gaps": ["Unable to analyze gaps due to paper retrieval issues"]})
        
        client = Groq(api_key=groq_api_key)
        
        # Use top papers for gap analysis
        top_papers = papers[:8] if len(papers) > 8 else papers
        
        gap_prompt = f"""
        Based on these research papers about "{topic}", identify research gaps:
        
        Top Papers Analysis:
        {json.dumps([{
            'title': p.get('title', ''),
            'key_contributions': p.get('key_contributions', ''),
            'methodology': p.get('methodology', ''),
            'limitations': p.get('limitations', ''),
            'relevance_score': p.get('relevance_score', 0)
        } for p in top_papers], indent=2)}
        
        Identify:
        1. METHODOLOGICAL_GAPS: Missing or underdeveloped research methods
        2. THEORETICAL_GAPS: Unexplored theoretical frameworks
        3. EMPIRICAL_GAPS: Lack of empirical validation or real-world studies  
        4. TECHNOLOGICAL_GAPS: Missing technological implementations
        5. APPLICATION_GAPS: Underexplored application domains
        6. INTERDISCIPLINARY_GAPS: Missing connections to other fields
        7. TEMPORAL_GAPS: Areas that need longitudinal studies
        8. SCALABILITY_GAPS: Issues with scaling to real-world scenarios
        
        Return JSON with detailed gap analysis and future research directions.
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert at identifying research opportunities. Return valid JSON."},
                {"role": "user", "content": gap_prompt}
            ],
            temperature=0.4,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        gaps_analysis = json.loads(response.choices[0].message.content)
        gaps_analysis["metadata"] = {
            "analyzed_papers": len(top_papers),
            "analysis_date": datetime.now().isoformat()
        }
        
        return json.dumps(gaps_analysis, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Gap analysis failed: {str(e)}"})


@tool 
def generate_research_report(
    topic: str,
    plan_json: str,
    ranked_papers_json: str, 
    gaps_json: str
) -> str:
    """Generate final comprehensive research report.
    
    Args:
        topic: Research topic
        plan_json: Research plan JSON
        ranked_papers_json: Ranked papers JSON
        gaps_json: Research gaps JSON
        
    Returns:
        Comprehensive research report as markdown string
    """
    from groq import Groq
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return "Error: GROQ_API_KEY not found in environment"
    
    try:
        client = Groq(api_key=groq_api_key)
        
        report_prompt = f"""
        Generate a comprehensive academic research report for: "{topic}"
        
        Research Plan:
        {plan_json[:1000]}...
        
        Top Ranked Papers:
        {ranked_papers_json[:2000]}...
        
        Research Gaps Analysis:
        {gaps_json[:1000]}...
        
        Create a well-structured markdown report with:
        
        # Research Report: {topic}
        
        ## Executive Summary
        - Brief overview of findings
        - Key insights discovered
        - Main research gaps identified
        
        ## Methodology 
        - Research approach used
        - Sources consulted
        - Evaluation criteria
        
        ## Literature Analysis
        ### Key Papers and Findings
        - Top 5 most relevant papers with analysis
        - Main methodologies identified
        - Current state of the field
        
        ### Research Landscape
        - Major contributors and institutions
        - Emerging trends and patterns
        - Theoretical frameworks in use
        
        ## Research Gaps and Opportunities
        - Detailed gap analysis
        - Future research directions
        - Potential impact areas
        
        ## Recommendations
        - Priority areas for future research
        - Methodological improvements needed
        - Practical applications to explore
        
        ## Conclusion
        - Summary of key findings
        - Implications for the field
        - Next steps for researchers
        
        Make it professional, comprehensive, and suitable for academic/research contexts.
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert research report writer."},
                {"role": "user", "content": report_prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        report = response.choices[0].message.content
        
        # Add metadata footer
        report += f"\n\n---\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return report
        
    except Exception as e:
        return f"Error generating report: {str(e)}"


# Tool lists for different agent types
SEARCH_TOOLS = [search_arxiv, search_web]
PLANNING_TOOLS = [create_research_plan]
ANALYSIS_TOOLS = [rank_papers, identify_research_gaps]
REPORTING_TOOLS = [generate_research_report]

ALL_TOOLS = SEARCH_TOOLS + PLANNING_TOOLS + ANALYSIS_TOOLS + REPORTING_TOOLS