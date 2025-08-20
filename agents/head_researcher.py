"""Head Researcher Agent - Main orchestrator using LangGraph and Groq."""

import os
import json
import asyncio
from typing import Dict, List, Any, TypedDict, Annotated, Sequence
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint import MemorySaver
import operator

from agents.planner import PlannerAgent
from agents.search_agent import SearchAgent
from agents.clarification_agent import ClarificationAgent

# Load environment variables
load_dotenv()


class ResearchState(TypedDict):
    """State for the research workflow."""
    topic: str
    clarifying_questions: Dict[str, Any]
    user_answers: Dict[int, str]
    enhanced_context: Dict[str, Any]
    research_plan: Dict[str, Any]
    search_results: List[Dict[str, Any]]
    ranked_papers: List[Dict[str, Any]]
    research_gaps: List[str]
    final_report: str
    messages: Annotated[Sequence[Dict], operator.add]
    current_step: str
    errors: List[str]
    skip_clarification: bool


class HeadResearcher:
    """Main orchestrator for research workflow using LangGraph."""
    
    def __init__(self):
        """Initialize the Head Researcher."""
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required. Please set it in your .env file.")
        
        self.groq_client = Groq(api_key=self.groq_api_key)
        self.planner = PlannerAgent()
        self.search_agent = SearchAgent()
        self.clarification_agent = ClarificationAgent()
        
        # Initialize the workflow graph
        self.workflow = self._build_workflow()
        self.memory = MemorySaver()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("analyze_topic", self.analyze_topic)
        workflow.add_node("generate_questions", self.generate_clarifying_questions)
        workflow.add_node("process_answers", self.process_user_answers)
        workflow.add_node("create_plan", self.create_research_plan)
        workflow.add_node("execute_search", self.execute_search)
        workflow.add_node("rank_papers", self.rank_papers)
        workflow.add_node("identify_gaps", self.identify_research_gaps)
        workflow.add_node("generate_report", self.generate_final_report)
        
        # Define the flow with conditional branching
        workflow.set_entry_point("analyze_topic")
        
        # Add conditional edge after analysis
        workflow.add_conditional_edges(
            "analyze_topic",
            self.should_ask_questions,
            {
                "ask_questions": "generate_questions",
                "skip_questions": "create_plan"
            }
        )
        
        workflow.add_edge("generate_questions", "process_answers")
        workflow.add_edge("process_answers", "create_plan")
        workflow.add_edge("create_plan", "execute_search")
        workflow.add_edge("execute_search", "rank_papers")
        workflow.add_edge("rank_papers", "identify_gaps")
        workflow.add_edge("identify_gaps", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    async def analyze_topic(self, state: ResearchState) -> ResearchState:
        """Analyze and understand the research topic.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with topic analysis
        """
        topic = state["topic"]
        
        prompt = f"""
        Analyze the following research topic and extract key components:
        
        Topic: {topic}
        
        Please identify:
        1. Main research domain and subdomain
        2. Key concepts and terms to search for
        3. Related fields that might have relevant work
        4. Temporal scope (recent developments vs historical context)
        5. Specific aspects that need investigation
        
        Return your analysis in a structured format.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an expert research analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            analysis = response.choices[0].message.content
            
            state["messages"].append({
                "role": "analyzer",
                "content": analysis,
                "timestamp": datetime.now().isoformat()
            })
            state["current_step"] = "topic_analyzed"
            
        except Exception as e:
            state["errors"].append(f"Topic analysis error: {str(e)}")
            
        return state
    
    def should_ask_questions(self, state: ResearchState) -> str:
        """Determine whether to ask clarifying questions.
        
        Args:
            state: Current research state
            
        Returns:
            Next node to execute
        """
        # Skip if explicitly disabled
        if state.get("skip_clarification", False):
            return "skip_questions"
        
        # Evaluate topic ambiguity
        topic = state["topic"]
        ambiguity_level, _ = self.clarification_agent.evaluate_topic_ambiguity(topic)
        
        # Always ask questions for high ambiguity, sometimes for medium
        if ambiguity_level == "high" or (ambiguity_level == "medium" and len(topic.split()) < 10):
            return "ask_questions"
        
        return "skip_questions"
    
    async def generate_clarifying_questions(self, state: ResearchState) -> ResearchState:
        """Generate clarifying questions for the research topic.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with clarifying questions
        """
        topic = state["topic"]
        analysis = state["messages"][-1]["content"] if state["messages"] else ""
        
        try:
            # Generate questions using the clarification agent
            questions_data = await self.clarification_agent.generate_clarifying_questions(
                topic=topic,
                initial_analysis=analysis
            )
            
            state["clarifying_questions"] = questions_data
            state["messages"].append({
                "role": "clarifier",
                "content": f"Generated {len(questions_data.get('questions', []))} clarifying questions",
                "timestamp": datetime.now().isoformat()
            })
            state["current_step"] = "questions_generated"
            
        except Exception as e:
            state["errors"].append(f"Question generation error: {str(e)}")
            state["clarifying_questions"] = {}
            
        return state
    
    async def process_user_answers(self, state: ResearchState) -> ResearchState:
        """Process user answers to clarifying questions.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with enhanced context
        """
        # Note: In actual implementation, this would interact with the user
        # For now, we'll create a placeholder for the enhanced context
        
        topic = state["topic"]
        questions = state.get("clarifying_questions", {}).get("questions", [])
        answers = state.get("user_answers", {})
        
        if answers:
            try:
                # Process answers using the clarification agent
                enhanced_context = await self.clarification_agent.process_answers(
                    topic=topic,
                    questions=questions,
                    answers=answers
                )
                
                state["enhanced_context"] = enhanced_context
                state["messages"].append({
                    "role": "clarifier",
                    "content": f"Processed {len(answers)} user answers to enhance research context",
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                state["errors"].append(f"Answer processing error: {str(e)}")
                state["enhanced_context"] = {"refined_topic": topic}
        else:
            # No answers provided, use original topic
            state["enhanced_context"] = {"refined_topic": topic}
        
        state["current_step"] = "answers_processed"
        return state
    
    async def create_research_plan(self, state: ResearchState) -> ResearchState:
        """Create a detailed research plan using the Planner agent.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with research plan
        """
        topic = state["topic"]
        
        # Use enhanced context if available
        enhanced_context = state.get("enhanced_context", {})
        if enhanced_context:
            # Combine original analysis with enhanced context
            context_parts = []
            if state["messages"]:
                context_parts.append(f"Initial Analysis: {state['messages'][0]['content']}")
            if enhanced_context.get("refined_topic"):
                context_parts.append(f"Refined Topic: {enhanced_context['refined_topic']}")
            if enhanced_context.get("scope_boundaries"):
                context_parts.append(f"Scope: {json.dumps(enhanced_context['scope_boundaries'])}")
            if enhanced_context.get("technical_requirements"):
                context_parts.append(f"Technical Requirements: {json.dumps(enhanced_context['technical_requirements'])}")
            if enhanced_context.get("constraints"):
                context_parts.append(f"Constraints: {json.dumps(enhanced_context['constraints'])}")
            
            context = "\n\n".join(context_parts)
        else:
            context = state["messages"][-1]["content"] if state["messages"] else ""
        
        # Use the planner agent to create a comprehensive plan
        plan = await self.planner.create_plan(topic, context)
        
        state["research_plan"] = plan
        state["messages"].append({
            "role": "planner",
            "content": json.dumps(plan, indent=2),
            "timestamp": datetime.now().isoformat()
        })
        state["current_step"] = "plan_created"
        
        return state
    
    async def execute_search(self, state: ResearchState) -> ResearchState:
        """Execute search based on the research plan.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with search results
        """
        plan = state["research_plan"]
        
        # Extract search queries from the plan
        search_queries = plan.get("search_queries", [])
        
        # Use the search agent to find papers
        all_results = []
        for query in search_queries:
            results = await self.search_agent.search(
                query=query["query"],
                sources=query.get("sources", ["arxiv", "web"]),
                max_results=query.get("max_results", 10)
            )
            all_results.extend(results)
        
        state["search_results"] = all_results
        state["messages"].append({
            "role": "searcher",
            "content": f"Found {len(all_results)} papers/resources",
            "timestamp": datetime.now().isoformat()
        })
        state["current_step"] = "search_completed"
        
        return state
    
    async def rank_papers(self, state: ResearchState) -> ResearchState:
        """Rank papers based on relevance and quality.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with ranked papers
        """
        papers = state["search_results"]
        topic = state["topic"]
        
        ranking_prompt = f"""
        Rank the following research papers based on their relevance to the topic: "{topic}"
        
        Papers:
        {json.dumps(papers, indent=2)}
        
        Ranking criteria:
        1. Direct relevance to the research topic
        2. Citation count and impact
        3. Recency of publication
        4. Quality of methodology
        5. Novelty of approach
        
        Return a ranked list with scores and justifications for top papers.
        Format as JSON with fields: rank, title, score, justification, key_findings
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating research quality."},
                    {"role": "user", "content": ranking_prompt}
                ],
                temperature=0.2,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            ranked_papers = json.loads(response.choices[0].message.content)
            state["ranked_papers"] = ranked_papers.get("papers", [])
            
        except Exception as e:
            state["errors"].append(f"Paper ranking error: {str(e)}")
            state["ranked_papers"] = papers[:10]  # Fallback to first 10
            
        state["current_step"] = "papers_ranked"
        return state
    
    async def identify_research_gaps(self, state: ResearchState) -> ResearchState:
        """Identify gaps in current research.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with identified research gaps
        """
        ranked_papers = state["ranked_papers"]
        topic = state["topic"]
        
        gap_analysis_prompt = f"""
        Based on the following ranked papers on "{topic}", identify research gaps:
        
        Top Papers:
        {json.dumps(ranked_papers[:10], indent=2)}
        
        Please identify:
        1. Unexplored areas within this topic
        2. Methodological limitations in current research
        3. Missing connections between related fields
        4. Unanswered questions
        5. Potential future research directions
        6. Practical applications not yet explored
        7. Theoretical frameworks that need development
        
        Return a detailed list of research gaps with explanations.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an expert at identifying research opportunities."},
                    {"role": "user", "content": gap_analysis_prompt}
                ],
                temperature=0.4,
                max_tokens=1500
            )
            
            gaps_text = response.choices[0].message.content
            
            # Parse gaps into a list
            gaps = []
            for line in gaps_text.split('\n'):
                if line.strip() and (line[0].isdigit() or line.startswith('-')):
                    gaps.append(line.strip())
            
            state["research_gaps"] = gaps
            
        except Exception as e:
            state["errors"].append(f"Gap analysis error: {str(e)}")
            state["research_gaps"] = ["Unable to identify gaps due to error"]
            
        state["current_step"] = "gaps_identified"
        return state
    
    async def generate_final_report(self, state: ResearchState) -> ResearchState:
        """Generate the final research report.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with final report
        """
        report_prompt = f"""
        Generate a comprehensive research report based on the following:
        
        Topic: {state['topic']}
        
        Research Plan Summary:
        {json.dumps(state['research_plan'].get('summary', ''), indent=2)}
        
        Top Ranked Papers (showing top 5):
        {json.dumps(state['ranked_papers'][:5], indent=2)}
        
        Identified Research Gaps:
        {json.dumps(state['research_gaps'], indent=2)}
        
        Please create a well-structured report with:
        1. Executive Summary
        2. Research Methodology
        3. Key Findings from Literature
        4. Analysis of Top Papers
        5. Research Gaps and Opportunities
        6. Recommendations for Future Research
        7. Conclusion
        
        Make it comprehensive but concise, suitable for academic or professional presentation.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an expert research report writer."},
                    {"role": "user", "content": report_prompt}
                ],
                temperature=0.3,
                max_tokens=3000
            )
            
            state["final_report"] = response.choices[0].message.content
            
        except Exception as e:
            state["errors"].append(f"Report generation error: {str(e)}")
            state["final_report"] = self._create_fallback_report(state)
            
        state["current_step"] = "report_generated"
        return state
    
    def _create_fallback_report(self, state: ResearchState) -> str:
        """Create a basic report if generation fails."""
        return f"""
        # Research Report: {state['topic']}
        
        ## Summary
        Research conducted on: {state['topic']}
        Papers found: {len(state['search_results'])}
        Research gaps identified: {len(state['research_gaps'])}
        
        ## Top Papers
        {chr(10).join([f"- {p.get('title', 'Unknown')}" for p in state['ranked_papers'][:5]])}
        
        ## Research Gaps
        {chr(10).join(state['research_gaps'][:5])}
        
        ## Errors Encountered
        {chr(10).join(state['errors'])}
        """
    
    async def conduct_research(self, topic: str, skip_clarification: bool = False, user_answers: Dict[int, str] = None) -> Dict[str, Any]:
        """Main entry point to conduct research on a topic.
        
        Args:
            topic: Research topic to investigate
            skip_clarification: Whether to skip clarifying questions
            user_answers: Pre-provided answers to clarifying questions
            
        Returns:
            Dictionary containing the final report and metadata
        """
        initial_state = ResearchState(
            topic=topic,
            clarifying_questions={},
            user_answers=user_answers or {},
            enhanced_context={},
            research_plan={},
            search_results=[],
            ranked_papers=[],
            research_gaps=[],
            final_report="",
            messages=[],
            current_step="initialized",
            errors=[],
            skip_clarification=skip_clarification
        )
        
        # Run the workflow
        config = {"configurable": {"thread_id": f"research_{datetime.now().timestamp()}"}}
        final_state = await self.workflow.ainvoke(initial_state, config)
        
        return {
            "topic": topic,
            "clarifying_questions": final_state.get("clarifying_questions", {}),
            "enhanced_context": final_state.get("enhanced_context", {}),
            "report": final_state["final_report"],
            "top_papers": final_state["ranked_papers"][:10],
            "research_gaps": final_state["research_gaps"],
            "total_papers_found": len(final_state["search_results"]),
            "errors": final_state["errors"],
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Example usage
    async def main():
        researcher = HeadResearcher()
        results = await researcher.conduct_research(
            "Applications of transformer models in time series forecasting"
        )
        print(results["report"])
    
    asyncio.run(main())