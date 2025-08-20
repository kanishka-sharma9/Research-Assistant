"""Enhanced workflow wrapper for the research system with clarification support."""

import asyncio
from typing import Dict, Any, Optional
from agents.head_researcher import HeadResearcher
from agents.clarification_agent import ClarificationAgent


class EnhancedResearchSupervisor:
    """Enhanced interface for conducting research with clarification support."""
    
    def __init__(self):
        """Initialize the enhanced research supervisor."""
        self.researcher = HeadResearcher()
        self.clarification_agent = ClarificationAgent()
    
    async def conduct_research_with_clarification(
        self, 
        topic: str,
        auto_mode: bool = False
    ) -> Dict[str, Any]:
        """Conduct research with optional clarifying questions.
        
        Args:
            topic: Research topic
            auto_mode: If True, skip clarification questions
            
        Returns:
            Research results dictionary
        """
        if auto_mode:
            # Skip clarification in auto mode
            return await self.researcher.conduct_research(
                topic=topic,
                skip_clarification=True
            )
        
        # First, check if we need clarification
        ambiguity_level, _ = self.clarification_agent.evaluate_topic_ambiguity(topic)
        
        if ambiguity_level in ["medium", "high"]:
            print(f"\n{'='*80}")
            print(f"Topic Ambiguity: {ambiguity_level.upper()}")
            print("Generating clarifying questions to improve research quality...")
            print(f"{'='*80}\n")
            
            # Generate questions
            questions_data = await self.clarification_agent.generate_clarifying_questions(topic)
            
            # Display questions
            formatted_questions = self.clarification_agent.format_questions_for_display(questions_data)
            print(formatted_questions)
            
            # Collect answers
            answers = await self._collect_user_answers(questions_data.get("questions", []))
            
            if answers:
                print("\nProcessing your answers to enhance the research plan...")
                
                # Conduct research with answers
                return await self.researcher.conduct_research(
                    topic=topic,
                    skip_clarification=False,
                    user_answers=answers
                )
        
        # Low ambiguity or no questions needed
        return await self.researcher.conduct_research(
            topic=topic,
            skip_clarification=True
        )
    
    async def _collect_user_answers(self, questions: list) -> Dict[int, str]:
        """Collect user answers to clarifying questions.
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            Dictionary mapping question IDs to answers
        """
        answers = {}
        
        print("\nPlease answer the following questions (press Enter to skip):\n")
        
        for q in questions:
            q_id = q.get("id", 0)
            question_text = q.get("question", "")
            priority = q.get("priority", "medium")
            
            # Mark critical questions
            if priority == "critical":
                print(f"[CRITICAL] ", end="")
            
            print(f"Q{q_id}: {question_text}")
            
            # Get user input
            answer = input("Your answer: ").strip()
            
            if answer:
                answers[q_id] = answer
            elif priority == "critical":
                print("  (Skipping critical question - research quality may be affected)")
            
            print()  # Add spacing
        
        if not answers:
            print("No answers provided. Proceeding with original topic...")
        else:
            print(f"\nThank you! Received {len(answers)} answers.")
        
        return answers
    
    def conduct_research_sync(
        self, 
        topic: str,
        auto_mode: bool = False
    ) -> Dict[str, Any]:
        """Synchronous wrapper for conduct_research_with_clarification.
        
        Args:
            topic: Research topic
            auto_mode: If True, skip clarification questions
            
        Returns:
            Research results dictionary
        """
        return asyncio.run(self.conduct_research_with_clarification(topic, auto_mode))
    
    def print_research_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of research results.
        
        Args:
            results: Research results dictionary
        """
        print("\n" + "="*80)
        print("RESEARCH COMPLETE")
        print("="*80)
        
        print(f"\nTopic: {results.get('topic', 'N/A')}")
        
        if results.get("enhanced_context"):
            refined_topic = results["enhanced_context"].get("refined_topic")
            if refined_topic and refined_topic != results.get('topic'):
                print(f"Refined Topic: {refined_topic}")
        
        print(f"Papers Found: {results.get('total_papers_found', 0)}")
        print(f"Research Gaps Identified: {len(results.get('research_gaps', []))}")
        
        if results.get("errors"):
            print(f"\nWarnings/Errors: {len(results['errors'])}")
            for error in results['errors'][:3]:
                print(f"  - {error}")
        
        print("\n" + "-"*80)
        print("TOP PAPERS:")
        print("-"*80)
        
        for i, paper in enumerate(results.get("top_papers", [])[:5], 1):
            title = paper.get("title", "Unknown Title")
            score = paper.get("score", "N/A")
            print(f"{i}. {title[:70]}... (Score: {score})")
        
        print("\n" + "-"*80)
        print("KEY RESEARCH GAPS:")
        print("-"*80)
        
        for gap in results.get("research_gaps", [])[:5]:
            if isinstance(gap, str):
                print(f"• {gap[:100]}...")
        
        print("\n" + "="*80)


def main():
    """Example usage of the enhanced research supervisor."""
    import sys
    
    # Get topic from command line or use default
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = input("Enter your research topic: ").strip()
        if not topic:
            topic = "Applications of transformer models in time series forecasting"
            print(f"Using default topic: {topic}")
    
    # Check for auto mode flag
    auto_mode = "--auto" in sys.argv
    
    # Create supervisor and conduct research
    supervisor = EnhancedResearchSupervisor()
    
    print(f"\nStarting research on: {topic}")
    if auto_mode:
        print("(Running in auto mode - skipping clarification questions)")
    
    try:
        results = supervisor.conduct_research_sync(topic, auto_mode)
        
        # Print summary
        supervisor.print_research_summary(results)
        
        # Optionally save full report
        save_report = input("\nSave full report to file? (y/n): ").strip().lower()
        if save_report == 'y':
            filename = f"research_report_{results['timestamp'][:10]}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Research Report: {topic}\n\n")
                f.write(f"Generated: {results['timestamp']}\n\n")
                f.write(results.get("report", "No report generated"))
            print(f"Report saved to: {filename}")
    
    except KeyboardInterrupt:
        print("\n\nResearch interrupted by user.")
    except Exception as e:
        print(f"\nError during research: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()