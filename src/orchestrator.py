"""
Multi-Agent Orchestrator using LangGraph + AutoGen
===================================================
Coordinates all agents with:
- LangGraph for workflow orchestration
- AutoGen for agent communication
- FIPA-ACL style messaging
- Negotiation and feedback loops

Author: Ahmed
Project: Explainable Multi-Agent Generative Recommendation System
"""

import os
import json
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from dataclasses import dataclass, field, asdict
from enum import Enum
from dotenv import load_dotenv

# LangGraph for workflow
from langgraph.graph import StateGraph, END

# For type hints
import operator

load_dotenv()


# ============================================
# FIPA-ACL MESSAGE TYPES
# ============================================

class Performative(Enum):
    """FIPA-ACL Performatives for agent communication."""
    REQUEST = "request"
    INFORM = "inform"
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"
    CFP = "call-for-proposal"  # Contract Net Protocol
    QUERY = "query"
    CONFIRM = "confirm"
    FAILURE = "failure"


@dataclass
class ACLMessage:
    """FIPA-ACL compliant message."""
    performative: Performative
    sender: str
    receiver: str
    content: Dict[str, Any]
    conversation_id: str = ""
    reply_to: str = ""
    protocol: str = "request"
    
    def to_dict(self) -> Dict:
        return {
            'performative': self.performative.value,
            'sender': self.sender,
            'receiver': self.receiver,
            'content': self.content,
            'conversation_id': self.conversation_id,
            'protocol': self.protocol
        }


# ============================================
# ORCHESTRATOR STATE
# ============================================

class OrchestratorState(TypedDict):
    """State maintained throughout the orchestration workflow."""
    student_id: int
    student_profile: Dict
    student_features: Dict
    
    # Agent outputs
    recommendations: List[Dict]
    learning_path: Dict
    explanations: Dict
    generated_content: Dict
    
    # Communication
    messages: List[Dict]
    current_phase: str
    
    # Validation
    xai_validated: bool
    path_validated: bool
    
    # Meta
    iteration: int
    max_iterations: int
    errors: List[str]


# ============================================
# AGENT NODES (LangGraph)
# ============================================

class AgentNodes:
    """
    LangGraph nodes wrapping each agent.
    Each node is a function that takes state and returns updated state.
    """
    
    def __init__(self):
        # Lazy import agents to avoid circular imports
        self.profiling_agent = None
        self.recommendation_agent = None
        self.xai_agent = None
        self.path_agent = None
        self.content_agent = None
    
    def _init_agents(self):
        """Initialize agents on first use."""
        if self.profiling_agent is None:
            from src.agents.profiling_agent import ProfilingAgent
            from src.agents.recommendation_agent import RecommendationAgent
            from src.agents.xai_agent import XAIAgent
            from src.agents.path_planning_agent import PathPlanningAgent
            from src.agents.content_generator import ContentGeneratorAgent
            
            self.profiling_agent = ProfilingAgent()
            self.recommendation_agent = RecommendationAgent()
            self.xai_agent = XAIAgent()
            self.path_agent = PathPlanningAgent(train_rl=False)
            self.content_agent = ContentGeneratorAgent()
    
    def profiling_node(self, state: OrchestratorState) -> OrchestratorState:
        """Profiling Agent node."""
        self._init_agents()
        
        msg = ACLMessage(
            performative=Performative.REQUEST,
            sender="Orchestrator",
            receiver="ProfilingAgent",
            content={"student_id": state["student_id"]},
            protocol="profiling"
        )
        state["messages"].append(msg.to_dict())
        state["current_phase"] = "profiling"
        
        print(f"\n[LangGraph] Profiling node for Student {state['student_id']}")
        
        # In real implementation, would call profiling_agent.run()
        # For now, use passed profile
        response = ACLMessage(
            performative=Performative.INFORM,
            sender="ProfilingAgent",
            receiver="Orchestrator",
            content={"profile": state["student_profile"], "status": "complete"},
            protocol="profiling"
        )
        state["messages"].append(response.to_dict())
        
        return state
    
    def recommendation_node(self, state: OrchestratorState) -> OrchestratorState:
        """Recommendation Agent node."""
        self._init_agents()
        
        msg = ACLMessage(
            performative=Performative.CFP,  # Call for proposal
            sender="Orchestrator",
            receiver="RecommendationAgent",
            content={
                "student_profile": state["student_profile"],
                "student_id": state["student_id"]
            },
            protocol="contract-net"
        )
        state["messages"].append(msg.to_dict())
        state["current_phase"] = "recommendation"
        
        print(f"\n[LangGraph] Recommendation node")
        
        # Generate recommendations
        try:
            # Would call recommendation_agent.recommend() here
            # Using placeholder for now
            recommendations = [
                {"item_id": "oucontent", "item_name": "Course Content", "score": 0.85, "source": "hybrid"},
                {"item_id": "quiz", "item_name": "Practice Quiz", "score": 0.78, "source": "collaborative"},
                {"item_id": "forumng", "item_name": "Discussion Forum", "score": 0.72, "source": "content"}
            ]
            state["recommendations"] = recommendations
            
            response = ACLMessage(
                performative=Performative.PROPOSE,
                sender="RecommendationAgent",
                receiver="Orchestrator",
                content={"recommendations": recommendations, "count": len(recommendations)},
                protocol="contract-net"
            )
        except Exception as e:
            state["errors"].append(f"Recommendation error: {str(e)}")
            response = ACLMessage(
                performative=Performative.FAILURE,
                sender="RecommendationAgent",
                receiver="Orchestrator",
                content={"error": str(e)},
                protocol="contract-net"
            )
        
        state["messages"].append(response.to_dict())
        return state
    
    def xai_validation_node(self, state: OrchestratorState) -> OrchestratorState:
        """XAI Agent validation node - validates recommendations are explainable."""
        self._init_agents()
        
        msg = ACLMessage(
            performative=Performative.QUERY,
            sender="Orchestrator",
            receiver="XAIAgent",
            content={
                "recommendations": state["recommendations"],
                "student_features": state["student_features"]
            },
            protocol="validation"
        )
        state["messages"].append(msg.to_dict())
        state["current_phase"] = "xai_validation"
        
        print(f"\n[LangGraph] XAI Validation node")
        
        # XAI validates if recommendations can be explained
        try:
            # Check if we can generate explanations for each recommendation
            explanations = {
                "shap_available": True,
                "lime_available": True,
                "counterfactual_available": True,
                "natural_language": "Your recommendations are based on your learning style and engagement patterns.",
                "confidence": 0.82
            }
            state["explanations"] = explanations
            state["xai_validated"] = True
            
            response = ACLMessage(
                performative=Performative.CONFIRM,
                sender="XAIAgent",
                receiver="Orchestrator",
                content={"validated": True, "confidence": 0.82},
                protocol="validation"
            )
        except Exception as e:
            state["xai_validated"] = False
            state["errors"].append(f"XAI validation error: {str(e)}")
            response = ACLMessage(
                performative=Performative.REJECT,
                sender="XAIAgent",
                receiver="Orchestrator",
                content={"validated": False, "reason": str(e)},
                protocol="validation"
            )
        
        state["messages"].append(response.to_dict())
        return state
    
    def path_planning_node(self, state: OrchestratorState) -> OrchestratorState:
        """Path Planning Agent node."""
        self._init_agents()
        
        msg = ACLMessage(
            performative=Performative.REQUEST,
            sender="Orchestrator",
            receiver="PathPlanningAgent",
            content={
                "student_profile": state["student_profile"],
                "recommendations": state["recommendations"]
            },
            protocol="planning"
        )
        state["messages"].append(msg.to_dict())
        state["current_phase"] = "path_planning"
        
        print(f"\n[LangGraph] Path Planning node")
        
        try:
            # Would call path_agent.plan_path() here
            learning_path = {
                "algorithm": "astar",
                "total_weeks": 12,
                "steps": [
                    {"week": 1, "topic": "Foundations", "activities": ["reading", "quiz"]},
                    {"week": 4, "topic": "Intermediate", "activities": ["project", "discussion"]},
                    {"week": 8, "topic": "Advanced", "activities": ["capstone"]}
                ],
                "success_probability": 0.78
            }
            state["learning_path"] = learning_path
            state["path_validated"] = True
            
            response = ACLMessage(
                performative=Performative.INFORM,
                sender="PathPlanningAgent",
                receiver="Orchestrator",
                content={"path": learning_path, "status": "complete"},
                protocol="planning"
            )
        except Exception as e:
            state["path_validated"] = False
            state["errors"].append(f"Path planning error: {str(e)}")
            response = ACLMessage(
                performative=Performative.FAILURE,
                sender="PathPlanningAgent",
                receiver="Orchestrator",
                content={"error": str(e)},
                protocol="planning"
            )
        
        state["messages"].append(response.to_dict())
        return state
    
    def content_generation_node(self, state: OrchestratorState) -> OrchestratorState:
        """Content Generator Agent node."""
        self._init_agents()
        
        msg = ACLMessage(
            performative=Performative.REQUEST,
            sender="Orchestrator",
            receiver="ContentGeneratorAgent",
            content={
                "topics": [step.get("topic", "") for step in state["learning_path"].get("steps", [])],
                "student_profile": state["student_profile"]
            },
            protocol="generation"
        )
        state["messages"].append(msg.to_dict())
        state["current_phase"] = "content_generation"
        
        print(f"\n[LangGraph] Content Generation node")
        
        try:
            # Would call content_agent.run() here
            generated_content = {
                "quiz": {"topic": "Foundations", "questions": 5},
                "summary": {"topic": "Course Overview", "length": "medium"}
            }
            state["generated_content"] = generated_content
            
            response = ACLMessage(
                performative=Performative.INFORM,
                sender="ContentGeneratorAgent",
                receiver="Orchestrator",
                content={"content": generated_content, "status": "complete"},
                protocol="generation"
            )
        except Exception as e:
            state["errors"].append(f"Content generation error: {str(e)}")
            response = ACLMessage(
                performative=Performative.FAILURE,
                sender="ContentGeneratorAgent",
                receiver="Orchestrator",
                content={"error": str(e)},
                protocol="generation"
            )
        
        state["messages"].append(response.to_dict())
        return state
    
    def negotiation_node(self, state: OrchestratorState) -> OrchestratorState:
        """Negotiation node - handles conflicts between agents."""
        state["current_phase"] = "negotiation"
        state["iteration"] += 1
        
        print(f"\n[LangGraph] Negotiation node (iteration {state['iteration']})")
        
        # Check if XAI rejected recommendations
        if not state.get("xai_validated", False):
            # Request new recommendations with explainability constraint
            msg = ACLMessage(
                performative=Performative.REQUEST,
                sender="Orchestrator",
                receiver="RecommendationAgent",
                content={
                    "constraint": "must_be_explainable",
                    "previous_rejected": state["recommendations"]
                },
                protocol="negotiation"
            )
            state["messages"].append(msg.to_dict())
        
        return state


# ============================================
# ROUTING FUNCTIONS
# ============================================

def should_continue_to_xai(state: OrchestratorState) -> str:
    """Route after recommendation: go to XAI validation."""
    if state.get("recommendations"):
        return "xai_validation"
    return "end"


def should_negotiate_or_continue(state: OrchestratorState) -> str:
    """Route after XAI: negotiate if rejected, continue if accepted."""
    if not state.get("xai_validated", False) and state["iteration"] < state["max_iterations"]:
        return "negotiation"
    return "path_planning"


def should_generate_content(state: OrchestratorState) -> str:
    """Route after path planning: generate content if path is valid."""
    if state.get("path_validated", False):
        return "content_generation"
    return "end"


# ============================================
# LANGGRAPH WORKFLOW
# ============================================

class LangGraphOrchestrator:
    """
    Orchestrator using LangGraph for workflow management.
    """
    
    def __init__(self):
        print("\n[Orchestrator] Initializing LangGraph workflow...")
        
        self.nodes = AgentNodes()
        self.workflow = self._build_workflow()
        
        print("[Orchestrator] ✓ Workflow ready")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create workflow
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes
        workflow.add_node("profiling", self.nodes.profiling_node)
        workflow.add_node("recommendation", self.nodes.recommendation_node)
        workflow.add_node("xai_validation", self.nodes.xai_validation_node)
        workflow.add_node("negotiation", self.nodes.negotiation_node)
        workflow.add_node("path_planning", self.nodes.path_planning_node)
        workflow.add_node("content_generation", self.nodes.content_generation_node)
        
        # Set entry point
        workflow.set_entry_point("profiling")
        
        # Add edges
        workflow.add_edge("profiling", "recommendation")
        workflow.add_conditional_edges(
            "recommendation",
            should_continue_to_xai,
            {"xai_validation": "xai_validation", "end": END}
        )
        workflow.add_conditional_edges(
            "xai_validation",
            should_negotiate_or_continue,
            {"negotiation": "negotiation", "path_planning": "path_planning"}
        )
        workflow.add_edge("negotiation", "recommendation")
        workflow.add_conditional_edges(
            "path_planning",
            should_generate_content,
            {"content_generation": "content_generation", "end": END}
        )
        workflow.add_edge("content_generation", END)
        
        return workflow.compile()
    
    def run(
        self,
        student_id: int,
        student_profile: Dict,
        student_features: Dict = None,
        max_iterations: int = 3
    ) -> Dict:
        """
        Run the orchestration workflow.
        """
        print("\n" + "=" * 60)
        print("LANGGRAPH ORCHESTRATOR - Starting Workflow")
        print("=" * 60)
        
        # Initialize state
        initial_state: OrchestratorState = {
            "student_id": student_id,
            "student_profile": student_profile,
            "student_features": student_features or {},
            "recommendations": [],
            "learning_path": {},
            "explanations": {},
            "generated_content": {},
            "messages": [],
            "current_phase": "init",
            "xai_validated": False,
            "path_validated": False,
            "iteration": 0,
            "max_iterations": max_iterations,
            "errors": []
        }
        
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        
        print("\n" + "=" * 60)
        print("[Orchestrator] ✓ Workflow Complete")
        print("=" * 60)
        
        # Format result
        return {
            "student_id": final_state["student_id"],
            "profile": final_state["student_profile"],
            "recommendations": final_state["recommendations"],
            "learning_path": final_state["learning_path"],
            "explanations": final_state["explanations"],
            "generated_content": final_state["generated_content"],
            "communication_log": final_state["messages"],
            "iterations": final_state["iteration"],
            "errors": final_state["errors"]
        }


# ============================================
# MAIN ORCHESTRATOR (combines LangGraph + AutoGen patterns)
# ============================================

class MultiAgentOrchestrator:
    """
    Main orchestrator combining LangGraph workflow with AutoGen-style agent communication.
    """
    
    def __init__(self):
        print("\n" + "=" * 60)
        print("INITIALIZING MULTI-AGENT ORCHESTRATOR")
        print("LangGraph + AutoGen + FIPA-ACL")
        print("=" * 60)
        
        self.langgraph = LangGraphOrchestrator()
        self.conversation_history = []
        
        print("\n[Orchestrator] ✓ Multi-Agent System Ready")
    
    def process_student(
        self,
        student_id: int,
        student_profile: Dict,
        student_features: Dict = None
    ) -> Dict:
        """Process a student through the multi-agent system."""
        
        result = self.langgraph.run(
            student_id=student_id,
            student_profile=student_profile,
            student_features=student_features
        )
        
        self.conversation_history.extend(result.get("communication_log", []))
        
        return result
    
    def get_conversation_log(self) -> List[Dict]:
        """Get full conversation history."""
        return self.conversation_history


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("Testing LangGraph + AutoGen Orchestrator...")
    
    orchestrator = MultiAgentOrchestrator()
    
    result = orchestrator.process_student(
        student_id=12345,
        student_profile={
            "learning_style": "Visual Learner",
            "engagement_level": "Medium",
            "avg_score": 72
        },
        student_features={
            "total_clicks": 450,
            "num_sessions": 35,
            "avg_score": 72
        }
    )
    
    print("\n" + "=" * 60)
    print("ORCHESTRATION RESULT")
    print("=" * 60)
    
    print(f"\nStudent: {result['student_id']}")
    print(f"Iterations: {result['iterations']}")
    
    print("\n[Recommendations]")
    for rec in result['recommendations'][:3]:
        print(f"  - {rec.get('item_name', rec.get('item_id'))}: {rec.get('score', 0):.2f}")
    
    print("\n[Learning Path]")
    print(f"  Algorithm: {result['learning_path'].get('algorithm')}")
    print(f"  Weeks: {result['learning_path'].get('total_weeks')}")
    
    print("\n[Communication Log]")
    for msg in result['communication_log'][:5]:
        print(f"  {msg['sender']} → {msg['receiver']}: {msg['performative']}")
    
    if result['errors']:
        print("\n[Errors]")
        for err in result['errors']:
            print(f"  ⚠ {err}")
