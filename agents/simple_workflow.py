"""Simplified Research Workflow without complex supervisor logic."""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from agents.simple_tools import (
    search_arxiv_simple,
    search_web_simple,
    create_simple_plan,
    analyze_papers_simple,
    identify_gaps_simple,
    generate_simple_report
)

# Load environment variables
load_dotenv()


class SimpleResearchWorkflow:
    """Simple sequential workflow for research."""
    
    def __init__(self):
        """Initialize the workflow."""
        pass
    
    async def conduct_research(self, topic: str) -> dict:
        """Conduct research using simple sequential steps.
        
        Args:
            topic: Research topic
            
        Returns:
            Dictionary with results
        """
        results = {
            "topic": topic,
            "research_plan": "",
            "search_results": "",
            "ranked_papers": "",
            "research_gaps": "",
            "final_report": "",
            "messages": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Create research plan
            print("Step 1: Creating research plan...")
            plan_result = create_simple_plan.invoke({"topic": topic, "context": ""})
            results["research_plan"] = plan_result
            results["messages"].append(f"Plan created: {plan_result[:100]}...")
            
            # Step 2: Search for papers
            print("Step 2: Searching for papers...")
            
            # Search ArXiv
            arxiv_results = search_arxiv_simple.invoke({"query": topic, "max_results": 5})
            results["messages"].append(f"ArXiv search completed")
            
            # Search Web
            web_results = search_web_simple.invoke({"query": topic, "max_results": 3})
            results["messages"].append(f"Web search completed")
            
            # Combine search results
            try:
                arxiv_data = json.loads(arxiv_results)
                web_data = json.loads(web_results)
                
                combined_results = []
                if isinstance(arxiv_data, list):
                    combined_results.extend(arxiv_data)
                if isinstance(web_data, list):
                    combined_results.extend(web_data)
                
                results["search_results"] = json.dumps(combined_results, indent=2)
                
            except json.JSONDecodeError:
                results["search_results"] = arxiv_results
            
            # Step 3: Analyze papers
            print("Step 3: Analyzing papers...")
            analysis_result = analyze_papers_simple.invoke({
                "papers_json": results["search_results"],
                "topic": topic
            })
            results["ranked_papers"] = analysis_result
            results["messages"].append("Paper analysis completed")
            
            # Step 4: Identify gaps
            print("Step 4: Identifying research gaps...")
            gaps_result = identify_gaps_simple.invoke({
                "papers_json": results["ranked_papers"],
                "topic": topic
            })
            results["research_gaps"] = gaps_result
            results["messages"].append("Gap analysis completed")
            
            # Step 5: Generate report
            print("Step 5: Generating final report...")
            report_result = generate_simple_report.invoke({
                "topic": topic,
                "plan_json": results["research_plan"],
                "papers_json": results["ranked_papers"],
                "gaps_json": results["research_gaps"]
            })
            results["final_report"] = report_result
            results["messages"].append("Report generated")
            
        except Exception as e:
            error_msg = f"Error in workflow: {str(e)}"
            results["messages"].append(error_msg)
            print(f"ERROR: {error_msg}")
        
        return results
    
    def conduct_research_sync(self, topic: str) -> dict:
        """Synchronous version of research workflow."""
        results = {
            "topic": topic,
            "research_plan": "",
            "search_results": "",
            "ranked_papers": "",
            "research_gaps": "",
            "final_report": "",
            "messages": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Create research plan
            print("Step 1: Creating research plan...")
            plan_result = create_simple_plan.invoke({"topic": topic, "context": ""})
            results["research_plan"] = plan_result
            results["messages"].append(f"Plan created: {plan_result[:100]}...")
            
            # Step 2: Search for papers
            print("Step 2: Searching for papers...")
            
            # Search ArXiv
            arxiv_results = search_arxiv_simple.invoke({"query": topic, "max_results": 5})
            results["messages"].append(f"ArXiv search completed")
            
            # Search Web
            web_results = search_web_simple.invoke({"query": topic, "max_results": 3})
            results["messages"].append(f"Web search completed")
            
            # Combine search results
            try:
                arxiv_data = json.loads(arxiv_results)
                web_data = json.loads(web_results)
                
                combined_results = []
                if isinstance(arxiv_data, list):
                    combined_results.extend(arxiv_data)
                if isinstance(web_data, list):
                    combined_results.extend(web_data)
                
                results["search_results"] = json.dumps(combined_results, indent=2)
                
            except json.JSONDecodeError:
                results["search_results"] = arxiv_results
            
            # Step 3: Analyze papers
            print("Step 3: Analyzing papers...")
            analysis_result = analyze_papers_simple.invoke({
                "papers_json": results["search_results"],
                "topic": topic
            })
            results["ranked_papers"] = analysis_result
            results["messages"].append("Paper analysis completed")
            
            # Step 4: Identify gaps
            print("Step 4: Identifying research gaps...")
            gaps_result = identify_gaps_simple.invoke({
                "papers_json": results["ranked_papers"],
                "topic": topic
            })
            results["research_gaps"] = gaps_result
            results["messages"].append("Gap analysis completed")
            
            # Step 5: Generate report
            print("Step 5: Generating final report...")
            report_result = generate_simple_report.invoke({
                "topic": topic,
                "plan_json": results["research_plan"],
                "papers_json": results["ranked_papers"],
                "gaps_json": results["research_gaps"]
            })
            results["final_report"] = report_result
            results["messages"].append("Report generated")
            
        except Exception as e:
            error_msg = f"Error in workflow: {str(e)}"
            results["messages"].append(error_msg)
            print(f"ERROR: {error_msg}")
        
        return results


# For backwards compatibility
class ResearchSupervisor:
    """Wrapper to maintain interface compatibility."""
    
    def __init__(self):
        self.workflow = SimpleResearchWorkflow()
    
    async def conduct_research(self, topic: str) -> dict:
        return await self.workflow.conduct_research(topic)
    
    def conduct_research_sync(self, topic: str) -> dict:
        return self.workflow.conduct_research_sync(topic)


if __name__ == "__main__":
    import asyncio
    
    async def main():
        workflow = SimpleResearchWorkflow()
        results = await workflow.conduct_research("your research topic")
        print("Final Report:")
        print(results["final_report"])
    
    asyncio.run(main())