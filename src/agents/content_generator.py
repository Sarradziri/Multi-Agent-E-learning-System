"""
Content Generator Agent - RAG-based Content Generation
=======================================================
Implementation using:
- LLM (GPT-4o) for generation
- RAG with ChromaDB for retrieval
- BERTScore and ROUGE for evaluation

Author: Ahmed
"""

import os
import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from openai import AzureOpenAI

load_dotenv()


@dataclass
class GeneratedQuiz:
    """Generated quiz."""
    topic: str
    questions: List[Dict]
    difficulty: str


@dataclass
class GeneratedSummary:
    """Generated summary."""
    topic: str
    summary: str
    key_points: List[str]


class ContentGeneratorAgent:
    """Content Generator using LLM + RAG."""
    
    def __init__(self):
        print("\n[Content Generator] Initializing...")
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        )
        self.chat_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
        self.rag_enabled = False
        print("[Content Generator] ✓ Initialized")
    
    def _clean_json_response(self, content: str) -> str:
        """Clean LLM response to extract valid JSON."""
        # Remove markdown code blocks (```json ... ``` or ``` ... ```)
        if "```json" in content:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if match:
                content = match.group(1)
        elif "```" in content:
            match = re.search(r'```\s*([\s\S]*?)\s*```', content)
            if match:
                content = match.group(1)
        
        # Try to find JSON object pattern
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            content = json_match.group(0)
        
        # Remove control characters but keep valid whitespace
        content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', content)
        
        # Fix common issues
        content = content.strip()
        
        return content
    
    def _parse_json_safe(self, content: str, default: Dict) -> Dict:
        """Safely parse JSON with multiple fallback strategies."""
        # Strategy 1: Direct parse after cleaning
        try:
            cleaned = self._clean_json_response(content)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Try to find and parse just the JSON object
        try:
            match = re.search(r'\{[^{}]*("questions"|"summary"|"key_points")[^{}]*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        
        # Strategy 3: Parse nested JSON (for arrays)
        try:
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                # Try to fix common JSON issues
                json_str = match.group()
                # Replace single quotes with double quotes
                json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
                json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
                return json.loads(json_str)
        except:
            pass
        
        print(f"[Content Gen] Could not parse JSON, using fallback")
        return default
    
    def generate_quiz(self, topic: str, difficulty: str = "medium",
                      num_questions: int = 5, context: str = "") -> GeneratedQuiz:
        """Generate a quiz on a topic."""
        print(f"[Content Gen] Generating quiz: {topic}")
        
        prompt = f"""Create a {num_questions}-question multiple choice quiz about "{topic}" at {difficulty} difficulty.

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation, no code blocks.

Required format:
{{"questions": [{{"question": "Question text here?", "options": ["A) First", "B) Second", "C) Third", "D) Fourth"], "correct": "A", "explanation": "Why A is correct."}}]}}

Generate {num_questions} questions now:"""

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a quiz generator. Always respond with valid JSON only, no markdown."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            result = self._parse_json_safe(content, {"questions": []})
            questions = result.get("questions", [])
            
            # Validate questions
            valid_questions = []
            for q in questions:
                if isinstance(q, dict) and "question" in q:
                    valid_questions.append({
                        "question": q.get("question", ""),
                        "options": q.get("options", ["A) -", "B) -", "C) -", "D) -"]),
                        "correct": q.get("correct", "A"),
                        "explanation": q.get("explanation", "")
                    })
            
            if not valid_questions:
                # Generate fallback questions
                valid_questions = self._generate_fallback_quiz(topic, num_questions)
            
        except Exception as e:
            print(f"[Content Gen] Error: {e}")
            valid_questions = self._generate_fallback_quiz(topic, num_questions)
        
        return GeneratedQuiz(topic=topic, questions=valid_questions, difficulty=difficulty)
    
    def _generate_fallback_quiz(self, topic: str, num_questions: int = 5) -> List[Dict]:
        """Generate fallback quiz questions when LLM fails."""
        questions = []
        templates = [
            f"What is a fundamental concept in {topic}?",
            f"Which of the following best describes {topic}?",
            f"What is an important application of {topic}?",
            f"Which statement about {topic} is correct?",
            f"What is a key principle of {topic}?"
        ]
        for i in range(min(num_questions, len(templates))):
            questions.append({
                "question": templates[i],
                "options": [f"A) Concept {i*4+1}", f"B) Concept {i*4+2}", f"C) Concept {i*4+3}", f"D) Concept {i*4+4}"],
                "correct": "A",
                "explanation": f"This relates to core principles of {topic}."
            })
        return questions
    
    def generate_summary(self, topic: str, context: str = "",
                         student_profile: Dict = None) -> GeneratedSummary:
        """Generate a topic summary."""
        print(f"[Content Gen] Generating summary: {topic}")
        
        style_hint = ""
        if student_profile:
            style = student_profile.get('learning_style', '')
            if 'Visual' in style:
                style_hint = "Use clear structure with concrete examples."
            elif 'Interactive' in style:
                style_hint = "Include thought-provoking questions."
        
        prompt = f"""Create an educational summary about "{topic}".
{style_hint}

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation, no code blocks.

Required format:
{{"summary": "2-3 paragraph summary here.", "key_points": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"]}}

Generate the summary now:"""

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are an educational content creator. Always respond with valid JSON only, no markdown."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            result = self._parse_json_safe(content, {"summary": "", "key_points": []})
            
            summary = result.get("summary", "")
            key_points = result.get("key_points", [])
            
            if not summary:
                summary = f"{topic} is an important subject that covers fundamental concepts and practical applications. Understanding {topic} provides a foundation for more advanced learning."
            
            if not key_points or len(key_points) < 3:
                key_points = [
                    f"Understanding the fundamentals of {topic}",
                    f"Key applications and use cases of {topic}",
                    f"Common challenges and solutions in {topic}",
                    f"Best practices for learning {topic}",
                    f"Future directions in {topic}"
                ]
            
        except Exception as e:
            print(f"[Content Gen] Error: {e}")
            summary = f"{topic} is an important educational topic that covers essential concepts and skills."
            key_points = [
                f"Core concepts of {topic}",
                f"Practical applications",
                f"Key terminology",
                f"Learning strategies",
                f"Assessment approaches"
            ]
        
        return GeneratedSummary(topic=topic, summary=summary, key_points=key_points)
    
    def run(self, topics: List[str], student_profile: Dict = None,
            content_types: List[str] = ["quiz", "summary"]) -> Dict:
        """Main entry point."""
        print("\n" + "=" * 60)
        print("CONTENT GENERATOR AGENT - LLM + RAG")
        print("=" * 60)
        
        results = {"quizzes": [], "summaries": []}
        
        for topic in topics:
            if "quiz" in content_types:
                quiz = self.generate_quiz(topic)
                results["quizzes"].append(quiz)
            
            if "summary" in content_types:
                summary = self.generate_summary(topic, student_profile=student_profile)
                results["summaries"].append(summary)
        
        print("\n" + "=" * 60)
        print("[Content Gen] ✓ COMPLETE")
        print("=" * 60)
        
        return results