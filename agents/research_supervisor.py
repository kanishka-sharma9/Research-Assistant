"""Research Supervisor using LangGraph with Planning and ReAct Agents."""

import os
import json
from typing import Dict, List, Any, Annotated, Sequence
from datetime import datetime
import functools
import operator

from dotenv import load_dotenv
from groq import Groq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from agents.simple_tools import (
    SIMPLE_SEARCH_TOOLS, 
    SIMPLE_PLANNING_TOOLS, 
    SIMPLE_ANALYSIS_TOOLS, 
    SIMPLE_REPORTING_TOOLS,
    ALL_SIMPLE_TOOLS
)

# Load environment variables
load_dotenv()


# Define the state that will be shared across agents
class ResearchState(dict):
    """State for the research workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    topic: str
    research_plan: str
    search_results: str
    ranked_papers: str
    research_gaps: str
    final_report: str


class ResearchSupervisor:
    """Supervisor node that orchestrates planning and react agents."""
    
    def __init__(self):
        """Initialize the research supervisor with agents."""
        # Get Groq API key
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        # Create Groq LLM instance
        self.llm = self._create_groq_llm()
        
        # Create agents
        self.planner_agent = self._create_planner_agent()
        self.search_agent = self._create_search_agent()
        self.analysis_agent = self._create_analysis_agent()
        self.reporter_agent = self._create_reporter_agent()
        
        # Create supervisor
        self.supervisor_agent = self._create_supervisor()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
    def _create_groq_llm(self):
        """Create a Groq LLM wrapper for LangGraph."""
        from langchain_groq import ChatGroq
        
        return ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.3
        )
    
    def _create_planner_agent(self):
        """Create the planning agent using ReAct pattern."""
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        
        system_prompt = """You are a Research Planning Agent. Your ONLY available tool is create_simple_plan.

        When you receive a research topic, you MUST:
        1. Call create_simple_plan with the topic as parameter
        
        Do NOT attempt to use any other tools. Only use create_simple_plan."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        return create_react_agent(self.llm, SIMPLE_PLANNING_TOOLS)
    
    def _create_search_agent(self):
        """Create the search agent using ReAct pattern."""
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        
        system_prompt = """You are a Research Search Agent. Your available tools are: search_arxiv_simple and search_web_simple.

        When you receive search instructions:
        1. Use search_arxiv_simple to find academic papers
        2. Use search_web_simple to find additional resources
        
        ONLY use search_arxiv_simple and search_web_simple tools. Do NOT use any other tools."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        return create_react_agent(self.llm, SIMPLE_SEARCH_TOOLS)
    
    def _create_analysis_agent(self):
        """Create the analysis agent using ReAct pattern."""
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        
        system_prompt = """You are a Research Analysis Agent. Your available tools are: analyze_papers_simple and identify_gaps_simple.

        When you receive search results:
        1. Use analyze_papers_simple to analyze papers
        2. Use identify_gaps_simple to find research gaps
        
        ONLY use analyze_papers_simple and identify_gaps_simple tools. Do NOT use any other tools."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        return create_react_agent(self.llm, SIMPLE_ANALYSIS_TOOLS)
    
    def _create_reporter_agent(self):
        """Create the reporting agent using ReAct pattern."""
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        
        system_prompt = """You are a Research Report Generator Agent. Your ONLY available tool is generate_simple_report.

        When you receive research data, you MUST:
        1. Call generate_simple_report with all available data
        
        ONLY use generate_simple_report tool. Do NOT use any other tools."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        return create_react_agent(self.llm, SIMPLE_REPORTING_TOOLS)
    
    def _create_supervisor(self):
        """Create the supervisor agent that routes between workers."""
        members = ["planner", "searcher", "analyzer", "reporter"]
        
        system_prompt = f"""
        You are a Research Supervisor managing a team of research agents.
        
        Team members: {members}
        
        Your job is to coordinate the research workflow by deciding which agent should work next.
        
        Workflow stages:
        1. PLANNING: Use 'planner' to create research plan
        2. SEARCHING: Use 'searcher' to find papers and resources  
        3. ANALYSIS: Use 'analyzer' to rank papers and identify gaps
        4. REPORTING: Use 'reporter' to generate final report
        5. FINISH: Use 'FINISH' when the research is complete
        
        Given the conversation so far, decide who should act next.
        Select one of: {members + ['FINISH']}
        
        Rules:
        - Start with 'planner' for new research topics
        - Move to 'searcher' after planning is complete
        - Use 'analyzer' after search results are available
        - Use 'reporter' after analysis is complete
        - Use 'FINISH' only when final report is generated
        
        Respond with just the name of the next agent or 'FINISH'.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Who should work next based on the current state?")
        ])
        
        def supervisor_node(state):
            messages = state["messages"]
            response = self.llm.invoke(
                prompt.format_messages(messages=messages)
            )
            next_agent = response.content.strip().lower()
            
            # Map agent names to node names
            agent_mapping = {
                "planner": "planner",
                "searcher": "searcher", 
                "analyzer": "analyzer",
                "reporter": "reporter",
                "finish": "FINISH"
            }
            
            return {"next": agent_mapping.get(next_agent, "FINISH")}
        
        return supervisor_node
    
    def _build_workflow(self):
        """Build the LangGraph workflow with supervisor and agents."""
        workflow = StateGraph(ResearchState)
        
        # Add agent nodes
        workflow.add_node("supervisor", self.supervisor_agent)
        workflow.add_node("planner", self._agent_node(self.planner_agent, "planner"))
        workflow.add_node("searcher", self._agent_node(self.search_agent, "searcher"))
        workflow.add_node("analyzer", self._agent_node(self.analysis_agent, "analyzer"))
        workflow.add_node("reporter", self._agent_node(self.reporter_agent, "reporter"))
        
        # Add edges
        workflow.add_edge("planner", "supervisor")
        workflow.add_edge("searcher", "supervisor") 
        workflow.add_edge("analyzer", "supervisor")
        workflow.add_edge("reporter", "supervisor")
        
        # Conditional edges from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {
                "planner": "planner",
                "searcher": "searcher",
                "analyzer": "analyzer", 
                "reporter": "reporter",
                "FINISH": END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("supervisor")
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def _agent_node(self, agent, name):
        """Create an agent node that executes the agent and updates state."""
        def node(state):
            result = agent.invoke(state)
            
            # Extract relevant information from messages
            last_message = result["messages"][-1]
            
            # Update state based on agent type
            updates = {"messages": result["messages"]}
            
            if name == "planner" and hasattr(last_message, 'content'):
                # Extract plan from planner response
                content = last_message.content
                if "research_objectives" in content or "search_strategy" in content:
                    updates["research_plan"] = content
            
            elif name == "searcher" and hasattr(last_message, 'content'):
                # Extract search results
                content = last_message.content
                if "title" in content and "abstract" in content:
                    updates["search_results"] = content
            
            elif name == "analyzer" and hasattr(last_message, 'content'):
                # Extract analysis results
                content = last_message.content
                if "relevance_score" in content:
                    updates["ranked_papers"] = content
                elif "gaps" in content or "GAPS" in content:
                    updates["research_gaps"] = content
            
            elif name == "reporter" and hasattr(last_message, 'content'):
                # Extract final report
                content = last_message.content
                if "# Research Report" in content or "Executive Summary" in content:
                    updates["final_report"] = content
            
            return updates
        
        return node
    
    async def conduct_research(self, topic: str) -> Dict[str, Any]:
        """Main entry point to conduct research on a topic.
        
        Args:
            topic: Research topic to investigate
            
        Returns:
            Dictionary containing research results
        """
        initial_state = ResearchState(
            messages=[HumanMessage(content=f"Conduct comprehensive research on: {topic}")],
            next="planner",
            topic=topic,
            research_plan="",
            search_results="",
            ranked_papers="",
            research_gaps="", 
            final_report=""
        )
        
        # Run the workflow
        config = {"configurable": {"thread_id": f"research_{datetime.now().timestamp()}"}}
        final_state = await self.workflow.ainvoke(initial_state, config)
        
        return {
            "topic": topic,
            "research_plan": final_state.get("research_plan", ""),
            "search_results": final_state.get("search_results", ""),
            "ranked_papers": final_state.get("ranked_papers", ""),
            "research_gaps": final_state.get("research_gaps", ""),
            "final_report": final_state.get("final_report", ""),
            "messages": [msg.content for msg in final_state.get("messages", [])],
            "timestamp": datetime.now().isoformat()
        }
    
    def conduct_research_sync(self, topic: str) -> Dict[str, Any]:
        """Synchronous version of conduct_research.
        
        Args:
            topic: Research topic to investigate
            
        Returns:
            Dictionary containing research results
        """
        initial_state = ResearchState(
            messages=[HumanMessage(content=f"Conduct comprehensive research on: {topic}")],
            next="planner",
            topic=topic,
            research_plan="",
            search_results="",
            ranked_papers="",
            research_gaps="",
            final_report=""
        )
        
        # Run the workflow
        config = {"configurable": {"thread_id": f"research_{datetime.now().timestamp()}"}}
        final_state = self.workflow.invoke(initial_state, config)
        
        return {
            "topic": topic,
            "research_plan": final_state.get("research_plan", ""),
            "search_results": final_state.get("search_results", ""),
            "ranked_papers": final_state.get("ranked_papers", ""),
            "research_gaps": final_state.get("research_gaps", ""),
            "final_report": final_state.get("final_report", ""),
            "messages": [msg.content for msg in final_state.get("messages", [])],
            "timestamp": datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        supervisor = ResearchSupervisor()
        results = await supervisor.conduct_research(
            "Applications of transformer models in time series forecasting"
        )
        print("Final Report:")
        print(results["final_report"])
    
    asyncio.run(main())