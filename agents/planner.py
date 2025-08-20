"""Planner Agent - Creates comprehensive research plans using Groq directly."""

import os
import json
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class PlannerAgent:
    """Agent responsible for creating detailed research plans."""
    
    def __init__(self):
        """Initialize the Planner Agent."""
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required. Please set it in your .env file.")
        self.groq_client = Groq(api_key=self.groq_api_key)
    
    # Enhanced ultra-detailed planning prompt for beginners and comprehensive research
    PLANNING_PROMPT = """
    You are an elite research planning specialist with decades of experience across multiple academic disciplines,
    industry research, and scientific investigation. You excel at creating EXHAUSTIVELY DETAILED research plans
    that guide even novice researchers through complex investigations step-by-step.
    
    RESEARCH TOPIC: {topic}
    
    ADDITIONAL CONTEXT & CLARIFICATIONS: {context}
    
    Your task is to create an ULTRA-COMPREHENSIVE, BEGINNER-FRIENDLY research plan that leaves no stone unturned.
    This plan should be so detailed that someone with minimal research experience could follow it successfully.
    
    ═══════════════════════════════════════════════════════════════════════════════════════════════
    
    PART I: FOUNDATIONAL UNDERSTANDING & OBJECTIVES
    
    1. TOPIC DECOMPOSITION & CONCEPT MAPPING
       a) Core Concept Analysis:
          - Break down the main topic into 5-10 fundamental components
          - Define each component in simple terms (ELI5 style)
          - Identify the relationships between components
          - Create a hierarchy of concepts from most to least important
       
       b) Terminology Clarification:
          - List 15-20 key technical terms that will appear in research
          - Provide clear definitions for each term
          - Include common synonyms and alternative terminology
          - Note any acronyms and their full forms
       
       c) Background Knowledge Requirements:
          - Prerequisites needed to understand this topic
          - Recommended foundational readings or courses
          - Skills or expertise that would be helpful
          - Common misconceptions to avoid
    
    2. RESEARCH QUESTIONS & OBJECTIVES HIERARCHY
       a) Grand Challenge Question:
          - The overarching question this research addresses
          - Why this question matters to the field and society
          - Historical context of this question
       
       b) Primary Research Question:
          - Specific, measurable, achievable main question
          - How answering this advances the field
          - Metrics for determining if question is answered
       
       c) Secondary Research Questions (5-8 questions):
          - Supporting questions that contribute to primary question
          - Order of importance and logical sequence
          - Dependencies between questions
       
       d) Exploratory Questions (3-5 questions):
          - Tangential but potentially valuable questions
          - Questions that might lead to unexpected insights
       
       e) Success Criteria & Measurable Outcomes:
          - Specific deliverables expected
          - Quality metrics for each deliverable
          - Minimum viable research outcome
          - Stretch goals if everything goes perfectly
    
    3. SCOPE DEFINITION WITH EXPLICIT BOUNDARIES
       a) Inclusion Criteria (BE VERY SPECIFIC):
          - Exactly what aspects will be investigated
          - Specific methodologies to be examined
          - Geographic regions if applicable
          - Time periods (with justification for chosen periods)
          - Types of studies to include (empirical, theoretical, review, etc.)
          - Minimum quality standards for sources
       
       b) Exclusion Criteria (BE EQUALLY SPECIFIC):
          - What will NOT be covered and why
          - Related topics that are out of scope
          - Methodologies not being considered
          - Types of sources to avoid
       
       c) Edge Cases & Gray Areas:
          - Borderline topics and how to handle them
          - Criteria for making inclusion/exclusion decisions
          - Protocol for unexpected discoveries
    
    ═══════════════════════════════════════════════════════════════════════════════════════════════
    
    PART II: COMPREHENSIVE SEARCH & DISCOVERY STRATEGY
    
    4. DETAILED SEARCH QUERIES (CREATE 20-25 QUERIES)
       Generate highly specific search queries organized by purpose:
       
       a) Foundational Understanding Queries (3-4):
          - Queries to find seminal/classic papers
          - Historical development queries
          - Theoretical foundation queries
       
       b) State-of-the-Art Queries (4-5):
          - Latest developments (last 2 years)
          - Cutting-edge methodology queries
          - Recent breakthrough queries
       
       c) Methodology-Specific Queries (3-4):
          - Queries for specific techniques/methods
          - Comparative methodology studies
          - Best practices and standards
       
       d) Application Domain Queries (3-4):
          - Real-world applications
          - Industry adoption queries
          - Case study searches
       
       e) Review & Survey Paper Queries (2-3):
          - Systematic review searches
          - Meta-analysis searches
          - Survey paper identification
       
       f) Cross-Disciplinary Queries (2-3):
          - Adjacent field applications
          - Interdisciplinary approaches
          - Technology transfer queries
       
       g) Challenge & Limitation Queries (2-3):
          - Known problems and challenges
          - Failure case studies
          - Limitation discussions
       
       h) Future Direction Queries (2-3):
          - Emerging trends
          - Future research agenda papers
          - Visionary/speculative works
       
       FORMAT EACH QUERY AS:
       {{
         "query_id": "unique_identifier",
         "query": "exact search string with operators",
         "purpose": "detailed explanation of what this finds",
         "rationale": "why this query is important",
         "sources": ["arxiv", "scholar", "web"],
         "max_results": 15,
         "filters": {{
           "year_from": 2019,
           "year_to": 2024,
           "document_type": ["journal", "conference", "preprint"],
           "min_citations": 5
         }},
         "expected_findings": "what we hope to discover",
         "backup_query": "alternative if main query fails"
       }}
    
    ═══════════════════════════════════════════════════════════════════════════════════════════════
    
    Return this comprehensive plan as a meticulously structured JSON object with all sections,
    subsections, and specific details clearly organized. Every element should be actionable,
    measurable, and sufficiently detailed for a beginner researcher to execute successfully.
    
    The JSON should follow this structure:
    {{
      "research_topic": "exact topic",
      "plan_version": "2.0-enhanced",
      "total_timeline_days": 21,
      "complexity_level": "beginner-friendly",
      "all_sections": {{ ... complete nested structure ... }}
    }}
    """
    
    async def create_plan(self, topic: str, context: str = "") -> Dict[str, Any]:
        """Create a comprehensive research plan for the given topic."""
        
        prompt = self.PLANNING_PROMPT.format(
            topic=topic,
            context=context if context else "No additional context provided."
        )
        
        try:
            response = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert research planning system. 
                        You create detailed, actionable research plans that guide systematic investigation.
                        Your plans are comprehensive, well-structured, and academically rigorous.
                        Always return valid JSON output."""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            plan = json.loads(response.choices[0].message.content)
            plan["metadata"] = {
                "created_at": datetime.now().isoformat(),
                "topic": topic,
                "planner_version": "2.0",
                "model_used": "mixtral-8x7b-32768"
            }
            
            return self._validate_and_enhance_plan(plan, topic)
            
        except json.JSONDecodeError as e:
            return self._create_fallback_plan(topic, str(e))
        except Exception as e:
            return self._create_fallback_plan(topic, str(e))
    
    def _validate_and_enhance_plan(self, plan: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """Validate and enhance the generated plan."""
        
        # Ensure search_queries exists and is properly formatted
        if "search_queries" not in plan or not isinstance(plan["search_queries"], list):
            plan["search_queries"] = self._generate_default_queries(topic)
        
        # Ensure all queries have required fields
        for query in plan["search_queries"]:
            if not isinstance(query, dict):
                continue
            query.setdefault("sources", ["arxiv", "web"])
            query.setdefault("max_results", 10)
            query.setdefault("purpose", "General search")
        
        # Add summary if not present
        if "summary" not in plan:
            plan["summary"] = f"Comprehensive research plan for: {topic}"
        
        return plan
    
    def _generate_default_queries(self, topic: str) -> List[Dict[str, Any]]:
        """Generate default search queries based on the topic."""
        
        keywords = topic.lower().replace(",", "").replace(".", "").split()
        main_terms = " ".join(keywords[:5])
        
        return [
            {
                "query": topic,
                "purpose": "Direct topic search",
                "sources": ["arxiv", "scholar", "web"],
                "max_results": 15
            },
            {
                "query": f"{main_terms} state of the art",
                "purpose": "Find latest developments",
                "sources": ["arxiv", "scholar"],
                "max_results": 10
            },
            {
                "query": f"{main_terms} survey",
                "purpose": "Find survey and review papers",
                "sources": ["arxiv", "scholar"],
                "max_results": 10
            },
            {
                "query": f"{main_terms} challenges",
                "purpose": "Identify research challenges",
                "sources": ["arxiv", "web"],
                "max_results": 10
            },
            {
                "query": f"{main_terms} applications",
                "purpose": "Find practical applications",
                "sources": ["scholar", "web"],
                "max_results": 10
            }
        ]
    
    def _create_fallback_plan(self, topic: str, error: str) -> Dict[str, Any]:
        """Create a basic fallback plan if the main generation fails."""
        
        return {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "topic": topic,
                "planner_version": "2.0-fallback",
                "error": error
            },
            "summary": f"Basic research plan for: {topic}",
            "objectives": {
                "primary": f"Investigate the current state of {topic}",
                "secondary": [
                    "Identify key papers and contributors",
                    "Understand main methodologies",
                    "Find research gaps"
                ]
            },
            "search_queries": self._generate_default_queries(topic),
            "methodology": {
                "approach": "Systematic literature review",
                "evaluation_criteria": ["relevance", "quality", "recency", "citations"],
                "synthesis_method": "Thematic analysis"
            },
            "expected_outputs": [
                "Comprehensive literature review",
                "Ranked list of relevant papers",
                "Identification of research gaps",
                "Future research recommendations"
            ]
        }