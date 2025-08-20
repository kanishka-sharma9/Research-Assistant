"""Main entry point for the Self-Initiated Research Agent with Clarifying Questions."""

import asyncio
import argparse
import os
import json
from datetime import datetime
from typing import Dict, Any
from agents.clarification_agent import ClarificationAgent

class ResearchAgentWithClarification:
    """Research agent that asks clarifying questions for ambiguous topics."""
    
    def __init__(self):
        """Initialize the research agent."""
        self.clarification_agent = ClarificationAgent()
        
        # Try to import the working research components
        try:
            from agents.simple_workflow import ResearchSupervisor
            self.research_supervisor = ResearchSupervisor()
            self.has_research_backend = True
        except ImportError:
            print("Warning: Research backend not available. Clarification demo mode only.")
            self.has_research_backend = False
    
    async def conduct_research_with_questions(
        self, 
        topic: str, 
        skip_questions: bool = False,
        output_file: str = None
    ) -> Dict[str, Any]:
        """Conduct research with optional clarifying questions."""
        
        print(f"\n{'='*80}")
        print(f"RESEARCH TOPIC: {topic}")
        print(f"{'='*80}")
        
        results = {
            "topic": topic,
            "original_topic": topic,
            "clarifying_questions_asked": 0,
            "ambiguity_level": "unknown",
            "enhanced_context": {},
            "user_answers": {},
            "timestamp": datetime.now().isoformat(),
            "report": "",
            "final_report": "",
            "research_plan": "",
            "search_results": "",
            "research_gaps": "",
            "messages": [],
            "errors": []
        }
        
        if not skip_questions:
            # Step 1: Evaluate topic ambiguity
            ambiguity_level, elements = self.clarification_agent.evaluate_topic_ambiguity(topic)
            results["ambiguity_level"] = ambiguity_level
            
            print(f"\nTopic Analysis:")
            print(f"  Ambiguity Level: {ambiguity_level.upper()}")
            print(f"  Issues Found: {', '.join(elements[:3])}")
            
            # Step 2: Ask clarifying questions if needed
            if ambiguity_level in ["medium", "high"]:
                print(f"\n{'-'*80}")
                print("CLARIFYING QUESTIONS")
                print(f"{'-'*80}")
                print("To improve research quality, please answer these questions:")
                print("(Press Enter to skip any question)\n")
                
                try:
                    # Generate questions
                    questions_data = await self.clarification_agent.generate_clarifying_questions(topic)
                    questions = questions_data.get("questions", [])
                    
                    if questions:
                        user_answers = {}
                        
                        for q in questions:
                            q_id = q.get("id", 0)
                            priority = q.get("priority", "medium").upper()
                            question = q.get("question", "")
                            why = q.get("why_important", "")
                            example = q.get("example_answer", "")
                            
                            print(f"[{priority}] Question {q_id}:")
                            print(f"  {question}")
                            print(f"  Why this matters: {why}")
                            if example:
                                print(f"  Example answer: {example}")
                            
                            try:
                                answer = input("\n  Your answer: ").strip()
                                if answer:
                                    user_answers[q_id] = answer
                                    print(f"  [RECORDED] {answer[:60]}...")
                                else:
                                    print(f"  [SKIPPED]")
                            except (KeyboardInterrupt, EOFError):
                                print("\n\n[INTERRUPTED] Stopping questions...")
                                break
                            
                            print("-" * 50)
                        
                        results["clarifying_questions_asked"] = len(user_answers)
                        results["user_answers"] = user_answers
                        
                        # Process answers if any were provided
                        if user_answers:
                            print(f"\n[PROCESSING] Analyzing {len(user_answers)} answers...")
                            
                            try:
                                enhanced_context = await self.clarification_agent.process_answers(
                                    topic=topic,
                                    questions=questions,
                                    answers=user_answers
                                )
                                results["enhanced_context"] = enhanced_context
                                
                                refined_topic = enhanced_context.get("refined_topic", topic)
                                if refined_topic and refined_topic != topic:
                                    print(f"[REFINED] Topic: {refined_topic}")
                                    results["topic"] = refined_topic
                                
                                print(f"[SUCCESS] Enhanced research context created!")
                                
                            except Exception as e:
                                print(f"[ERROR] Failed to process answers: {e}")
                                results["errors"].append(f"Answer processing error: {str(e)}")
                        else:
                            print("\n[INFO] No answers provided. Using original topic.")
                    else:
                        print("[ERROR] No questions were generated.")
                        results["errors"].append("Question generation failed")
                        
                except Exception as e:
                    print(f"[ERROR] Question generation failed: {e}")
                    results["errors"].append(f"Question generation error: {str(e)}")
                    
                print(f"\n{'='*80}")
                print("CLARIFICATION COMPLETE")
                print(f"{'='*80}")
            else:
                print(f"\n[INFO] Topic is clear enough. No clarification needed.")
        else:
            print(f"\n[SKIP] Clarifying questions disabled (auto mode)")
            results["ambiguity_level"] = "skipped"
        
        # Step 3: Conduct actual research
        print(f"\n[RESEARCH] Starting research process...")
        
        if self.has_research_backend:
            try:
                # Use the actual research backend
                research_results = self.research_supervisor.conduct_research_sync(results["topic"])
                
                # Merge research results
                for key in ["research_plan", "search_results", "ranked_papers", "research_gaps", "final_report", "messages"]:
                    if key in research_results:
                        results[key] = research_results[key]
                
                # Handle both report formats
                if research_results.get("final_report"):
                    results["report"] = research_results["final_report"]
                
                print(f"[SUCCESS] Research completed using full backend!")
                
            except Exception as e:
                print(f"[ERROR] Research backend failed: {e}")
                results["errors"].append(f"Research error: {str(e)}")
                results["report"] = f"Research backend error: {str(e)}"
        else:
            # Demonstration mode
            results["report"] = f"""# Research Report: {results['topic']}

## Executive Summary
This is a demonstration of the clarifying questions feature. In the full system, 
comprehensive research would be conducted here using the enhanced context.

## Clarification Summary
- Original Topic: {results['original_topic']}
- Refined Topic: {results['topic']}
- Ambiguity Level: {results['ambiguity_level']}
- Questions Asked: {results['clarifying_questions_asked']}

## Enhanced Context
{json.dumps(results['enhanced_context'], indent=2) if results['enhanced_context'] else 'No enhanced context available'}

## Next Steps
The clarifying questions feature is working correctly. 
To enable full research capabilities, ensure all research backend dependencies are installed.
"""
            print(f"[DEMO] Clarification demonstration completed!")
        
        return results

async def main():
    """Main function to run the research agent."""
    parser = argparse.ArgumentParser(
        description="Self-Initiated Research Agent with Clarifying Questions"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="Research topic to investigate",
        default=None
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file to save the report",
        default=None
    )
    parser.add_argument(
        "--no-interactive",
        "-n",
        action="store_true",
        help="Skip clarifying questions (auto mode)",
        default=False
    )
    
    args = parser.parse_args()
    
    print("Self-Initiated Research Agent")
    print("With Clarifying Questions Support")
    print("=" * 60)
    
    # Get research topic from user if not provided via command line
    if not args.topic:
        print("\nWelcome! Please enter your research topic below:")
        print("-" * 60)
        while True:
            try:
                topic_input = input("\nEnter research topic: ").strip()
                if topic_input:
                    args.topic = topic_input
                    break
                else:
                    print("Please enter a valid research topic.")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                return 1
    
    print(f"\nResearch Topic: {args.topic}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show guaranteed question topics hint
    if not args.no_interactive:
        print(f"\nTIP: These topics will ask clarifying questions:")
        print(f"     AI, machine learning, blockchain, data science, neural networks")
    
    print("-" * 60)
    
    try:
        # Initialize the research agent
        print("Initializing Research Agent...")
        agent = ResearchAgentWithClarification()
        
        # Conduct research with clarification
        results = await agent.conduct_research_with_questions(
            topic=args.topic,
            skip_questions=args.no_interactive,
            output_file=args.output
        )
        
        print(f"\n{'='*80}")
        print("RESEARCH COMPLETE")
        print(f"{'='*80}")
        
        # Display results summary
        print(f"\nRESULTS SUMMARY:")
        print(f"  Original Topic: {results['original_topic']}")
        if results['topic'] != results['original_topic']:
            print(f"  Refined Topic: {results['topic']}")
        print(f"  Ambiguity Level: {results['ambiguity_level']}")
        print(f"  Questions Asked: {results['clarifying_questions_asked']}")
        
        if results.get("errors"):
            print(f"  Warnings: {len(results['errors'])}")
            for error in results['errors'][:2]:
                print(f"    - {error}")
        
        # Display report excerpt
        report = results.get("report") or results.get("final_report", "")
        if report:
            print(f"\n{'-'*60}")
            print("REPORT EXCERPT (first 500 characters):")
            print(f"{'-'*60}")
            print(report[:500] + "..." if len(report) > 500 else report)
        
        # Save report to file
        output_filename = args.output if args.output else f"{args.topic.replace(' ', '_')}_research_report.md"
        
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"# Research Report: {args.topic}\n\n")
            f.write(f"Generated on: {results['timestamp']}\n\n")
            
            # Add clarification summary
            if results['clarifying_questions_asked'] > 0:
                f.write(f"## Clarification Summary\n")
                f.write(f"- Original Topic: {results['original_topic']}\n")
                f.write(f"- Refined Topic: {results['topic']}\n")
                f.write(f"- Ambiguity Level: {results['ambiguity_level']}\n")
                f.write(f"- Questions Answered: {results['clarifying_questions_asked']}\n\n")
                
                if results.get('user_answers'):
                    f.write(f"### User Answers\n")
                    for q_id, answer in results['user_answers'].items():
                        f.write(f"- Q{q_id}: {answer}\n")
                    f.write(f"\n")
            
            # Write the main report
            if report:
                f.write(report)
            else:
                f.write("No report content generated.\n")
        
        print(f"\n[SAVED] Report saved to: {output_filename}")
        
        # Show success message
        if results['clarifying_questions_asked'] > 0:
            print(f"\n[SUCCESS] Clarifying questions feature working correctly!")
            print(f"           {results['clarifying_questions_asked']} questions were answered")
        else:
            print(f"\n[INFO] No clarification needed for this topic")
        
        print(f"\nResearch completed successfully!")
            
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def sync_main():
    """Synchronous wrapper for the main function."""
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        print("\nResearch interrupted by user.")
        return 1


if __name__ == "__main__":
    exit(sync_main())