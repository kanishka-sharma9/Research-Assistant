"""Clarification Agent - Asks intelligent clarifying questions for ambiguous research topics."""

import os
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class ClarificationAgent:
    """Generates and processes clarifying questions for research topics."""
    
    def __init__(self):
        """Initialize the Clarification Agent."""
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required.")
        self.groq_client = Groq(api_key=self.groq_api_key)
    
    def evaluate_topic_ambiguity(self, topic: str) -> Tuple[str, List[str]]:
        """Evaluate the ambiguity level of a research topic."""
        
        # Broad terms that are inherently ambiguous
        broad_terms = [
            "ai", "artificial intelligence", "machine learning", "ml", "deep learning", "dl",
            "nlp", "natural language processing", "computer vision", "cv", "robotics",
            "blockchain", "cryptocurrency", "iot", "internet of things", "big data",
            "cloud computing", "cybersecurity", "data science", "analytics"
        ]
        
        # Ambiguous indicator words
        high_ambiguity_words = [
            "impact", "effect", "relationship", "influence", "role", "implications", 
            "applications", "potential", "future", "trends", "challenges", "opportunities",
            "benefits", "advantages", "disadvantages", "problems", "issues", "solutions"
        ]
        
        topic_lower = topic.lower().strip()
        word_count = len(topic.split())
        ambiguous_elements = []
        
        # Check for broad terms
        is_broad = any(term in topic_lower for term in broad_terms)
        if is_broad:
            ambiguous_elements.append("extremely broad topic")
        
        # Check for ambiguous words
        has_ambiguous_words = any(word in topic_lower for word in high_ambiguity_words)
        if has_ambiguous_words:
            ambiguous_elements.extend([word for word in high_ambiguity_words if word in topic_lower])
        
        # Check specifics
        has_temporal = any(char.isdigit() for char in topic) or any(
            temporal in topic_lower for temporal in ["recent", "latest", "current", "2023", "2024", "2025"]
        )
        has_domain_specifics = any(
            specific in topic_lower for specific in [
                "twitter", "facebook", "medical", "healthcare", "finance", "banking", 
                "sentiment", "classification", "prediction", "detection", "recognition"
            ]
        )
        
        if not has_temporal:
            ambiguous_elements.append("no temporal specification")
        if word_count <= 2:
            ambiguous_elements.append("topic too brief")
        if not has_domain_specifics and word_count < 8:
            ambiguous_elements.append("lacks domain specifics")
        
        # Determine ambiguity level
        if (is_broad and word_count <= 3) or (has_ambiguous_words and not has_temporal and not has_domain_specifics):
            level = "high"
        elif is_broad or has_ambiguous_words or (word_count <= 4 and not has_domain_specifics):
            level = "medium"
        elif word_count >= 8 and has_temporal and has_domain_specifics:
            level = "low"
        else:
            level = "medium"  # Default to asking questions when unsure
        
        return level, ambiguous_elements
    
    async def generate_clarifying_questions(self, topic: str, initial_analysis: str = "") -> Dict[str, Any]:
        """Generate clarifying questions based on the research topic."""
        
        prompt = f"""
        You are an expert research consultant. Generate 2-8 clarifying questions for this research topic:
        
        TOPIC: {topic}
        ANALYSIS: {initial_analysis or "No analysis provided"}
        
        Create questions in these categories:
        1. Scope (time period, geographic boundaries, what to include/exclude)
        2. Technical depth (introductory, detailed, expert-level)
        3. Application focus (academic, practical, business)
        4. Specific outcomes desired
        
        Return JSON format:
        {{
            "ambiguity_assessment": {{
                "level": "low|medium|high",
                "reasoning": "brief explanation"
            }},
            "questions": [
                {{
                    "id": 1,
                    "category": "scope|technical|application|output",
                    "question": "the actual question",
                    "why_important": "why this matters",
                    "example_answer": "example response",
                    "priority": "critical|high|medium|low"
                }}
            ]
        }}
        
        Make questions specific and actionable. Prioritize based on importance for clarifying the research scope.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an expert at generating clarifying questions for research topics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            questions_data = json.loads(response.choices[0].message.content)
            questions_data["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "topic": topic,
                "agent_version": "2.0"
            }
            return questions_data
            
        except Exception as e:
            return self._generate_fallback_questions(topic, str(e))
    
    async def process_answers(self, topic: str, questions: List[Dict[str, Any]], answers: Dict[int, str]) -> Dict[str, Any]:
        """Process user answers to create enhanced research context."""
        
        formatted_qa = []
        for q in questions:
            q_id = q.get("id", 0)
            if q_id in answers:
                formatted_qa.append({
                    "question": q.get("question", ""),
                    "category": q.get("category", ""),
                    "answer": answers[q_id]
                })
        
        prompt = f"""
        Based on these clarifying questions and answers, create an enhanced research context:
        
        ORIGINAL TOPIC: {topic}
        
        Q&A PAIRS:
        {json.dumps(formatted_qa, indent=2)}
        
        Return JSON with:
        {{
            "refined_topic": "more specific version of original topic",
            "scope_boundaries": "clear inclusion/exclusion criteria",
            "technical_requirements": "specific technical aspects to focus on",
            "application_context": "practical considerations",
            "constraints": "time, resource, or access limitations",
            "research_priorities": ["ordered list of priorities"]
        }}
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You synthesize clarifying answers into enhanced research contexts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            enhanced_context = json.loads(response.choices[0].message.content)
            enhanced_context["metadata"] = {
                "processed_at": datetime.now().isoformat(),
                "questions_answered": len(answers),
                "total_questions": len(questions)
            }
            return enhanced_context
            
        except Exception as e:
            return {
                "refined_topic": topic,
                "error": str(e),
                "answers_received": answers,
                "metadata": {"processed_at": datetime.now().isoformat(), "status": "fallback"}
            }
    
    def _generate_fallback_questions(self, topic: str, error: str) -> Dict[str, Any]:
        """Generate basic questions if main generation fails."""
        
        ambiguity_level, ambiguous_elements = self.evaluate_topic_ambiguity(topic)
        
        questions = [
            {
                "id": 1,
                "category": "scope",
                "question": "What time period should this research cover?",
                "why_important": "Focuses the literature search",
                "example_answer": "Focus on developments from 2020 onwards",
                "priority": "critical"
            },
            {
                "id": 2,
                "category": "technical",
                "question": "What level of technical depth are you looking for?",
                "why_important": "Determines complexity of analysis needed",
                "example_answer": "Graduate-level technical analysis",
                "priority": "high"
            }
        ]
        
        if ambiguity_level in ["medium", "high"]:
            questions.append({
                "id": 3,
                "category": "application",
                "question": "Are there specific applications or subfields to focus on?",
                "why_important": "Narrows research scope to most relevant areas",
                "example_answer": "Focus on healthcare applications",
                "priority": "high"
            })
        
        return {
            "ambiguity_assessment": {
                "level": ambiguity_level,
                "reasoning": f"Fallback assessment: {ambiguity_level} ambiguity detected"
            },
            "questions": questions,
            "error": f"Fallback mode: {error}",
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "topic": topic,
                "agent_version": "2.0-fallback"
            }
        }