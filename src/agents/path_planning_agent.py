"""
Path Planning Agent - Learning Path Optimization
=================================================
REAL Implementation using:
- NetworkX for Graph representation
- A* Search with heuristics
- Q-Learning for adaptive path optimization
- Topological sort for prerequisite ordering

Author: Ahmed
Project: Explainable Multi-Agent Generative Recommendation System
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import heapq
from dotenv import load_dotenv

# Graph library
import networkx as nx

# LLM for personalization
from openai import AzureOpenAI

load_dotenv()


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class CourseNode:
    """Represents a course/topic in the knowledge graph."""
    id: str
    name: str
    difficulty: float      # 0-1 scale
    estimated_hours: float
    prerequisites: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    domain: str = "general"


@dataclass
class PathStep:
    """A single step in the learning path."""
    node_id: str
    node_name: str
    week: int
    estimated_hours: float
    activities: List[str]
    rationale: str
    is_milestone: bool = False


@dataclass
class LearningPath:
    """Complete learning path for a student."""
    student_id: int
    steps: List[PathStep]
    total_weeks: int
    total_hours: float
    path_score: float           # Quality score from heuristics
    q_value: float              # Q-learning value
    algorithm_used: str         # 'astar', 'qlearning', 'topological'
    personalization_notes: str


# ============================================
# KNOWLEDGE GRAPH
# ============================================

class KnowledgeGraph:
    """
    Course/Topic Knowledge Graph using NetworkX.
    Supports prerequisite relationships and skill mappings.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, CourseNode] = {}
        self._build_default_graph()
    
    def _build_default_graph(self):
        """Build default course graph based on OULAD-like structure."""
        
        # Define courses with prerequisites
        courses = [
            # Foundation courses
            CourseNode("INTRO_COMP", "Introduction to Computing", 0.2, 30, [], ["basic_computing"], "STEM"),
            CourseNode("INTRO_MATH", "Foundation Mathematics", 0.2, 25, [], ["basic_math"], "STEM"),
            CourseNode("INTRO_PSYCH", "Introduction to Psychology", 0.2, 30, [], ["psychology_basics"], "Social"),
            
            # Intermediate courses
            CourseNode("PROG_BASICS", "Programming Fundamentals", 0.4, 40, ["INTRO_COMP"], ["programming"], "STEM"),
            CourseNode("DATA_STRUCT", "Data Structures", 0.5, 45, ["PROG_BASICS"], ["data_structures"], "STEM"),
            CourseNode("STATISTICS", "Statistics & Probability", 0.4, 35, ["INTRO_MATH"], ["statistics"], "STEM"),
            CourseNode("SOC_PSYCH", "Social Psychology", 0.4, 35, ["INTRO_PSYCH"], ["social_psychology"], "Social"),
            CourseNode("RESEARCH_METH", "Research Methods", 0.4, 40, ["INTRO_PSYCH", "STATISTICS"], ["research"], "Social"),
            
            # Advanced courses
            CourseNode("ALGORITHMS", "Algorithms & Complexity", 0.7, 50, ["DATA_STRUCT", "STATISTICS"], ["algorithms"], "STEM"),
            CourseNode("MACHINE_LEARN", "Machine Learning", 0.8, 60, ["ALGORITHMS", "STATISTICS"], ["ml"], "STEM"),
            CourseNode("DATA_SCIENCE", "Data Science", 0.7, 55, ["STATISTICS", "PROG_BASICS"], ["data_science"], "STEM"),
            CourseNode("COGNITIVE", "Cognitive Psychology", 0.6, 40, ["SOC_PSYCH"], ["cognitive"], "Social"),
            
            # Capstone
            CourseNode("AI_PROJECT", "AI Capstone Project", 0.9, 80, ["MACHINE_LEARN", "DATA_SCIENCE"], ["ai_project"], "STEM"),
            CourseNode("RESEARCH_PROJ", "Research Project", 0.8, 70, ["RESEARCH_METH", "COGNITIVE"], ["research_project"], "Social"),
        ]
        
        # Add nodes to graph
        for course in courses:
            self.add_node(course)
    
    def add_node(self, course: CourseNode):
        """Add a course node to the graph."""
        self.nodes[course.id] = course
        self.graph.add_node(
            course.id,
            name=course.name,
            difficulty=course.difficulty,
            hours=course.estimated_hours,
            domain=course.domain,
            skills=course.skills
        )
        
        # Add prerequisite edges
        for prereq in course.prerequisites:
            if prereq in self.nodes:
                self.graph.add_edge(prereq, course.id, weight=course.difficulty)
    
    def get_prerequisites(self, course_id: str) -> List[str]:
        """Get all prerequisites for a course."""
        if course_id not in self.graph:
            return []
        return list(self.graph.predecessors(course_id))
    
    def get_all_prerequisites(self, course_id: str) -> Set[str]:
        """Get all transitive prerequisites (ancestors)."""
        if course_id not in self.graph:
            return set()
        return nx.ancestors(self.graph, course_id)
    
    def get_successors(self, course_id: str) -> List[str]:
        """Get courses that depend on this one."""
        if course_id not in self.graph:
            return []
        return list(self.graph.successors(course_id))
    
    def topological_order(self) -> List[str]:
        """Get valid learning order respecting prerequisites."""
        return list(nx.topological_sort(self.graph))
    
    def shortest_path(self, start: str, goal: str) -> List[str]:
        """Find shortest path between two courses."""
        try:
            return nx.shortest_path(self.graph, start, goal)
        except nx.NetworkXNoPath:
            return []
    
    def get_entry_points(self) -> List[str]:
        """Get courses with no prerequisites (starting points)."""
        return [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
    
    def get_exit_points(self) -> List[str]:
        """Get final courses (no successors)."""
        return [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]


# ============================================
# HEURISTICS
# ============================================

class PathHeuristics:
    """
    Heuristics for evaluating and guiding path planning.
    """
    
    @staticmethod
    def difficulty_heuristic(
        current_node: CourseNode,
        goal_node: CourseNode,
        student_level: float
    ) -> float:
        """
        Estimate cost based on difficulty gap.
        Lower is better for A*.
        """
        difficulty_gap = abs(current_node.difficulty - student_level)
        goal_difficulty = goal_node.difficulty
        
        # Penalize large jumps in difficulty
        return difficulty_gap * 2 + (goal_difficulty - current_node.difficulty)
    
    @staticmethod
    def time_heuristic(
        current_node: CourseNode,
        goal_node: CourseNode,
        available_hours_per_week: float
    ) -> float:
        """
        Estimate time-based cost.
        """
        remaining_hours = goal_node.estimated_hours
        weeks_needed = remaining_hours / max(available_hours_per_week, 1)
        return weeks_needed
    
    @staticmethod
    def skill_match_heuristic(
        student_skills: Set[str],
        course_skills: List[str]
    ) -> float:
        """
        Reward courses that build on student's existing skills.
        Returns negative value (reward) for A* to minimize.
        """
        overlap = len(student_skills.intersection(set(course_skills)))
        return -overlap * 0.5  # Reward for skill match
    
    @staticmethod
    def combined_heuristic(
        current: CourseNode,
        goal: CourseNode,
        student_level: float,
        student_skills: Set[str],
        hours_per_week: float,
        weights: Dict[str, float] = None
    ) -> float:
        """
        Combined heuristic for A* search.
        """
        weights = weights or {'difficulty': 0.4, 'time': 0.3, 'skill': 0.3}
        
        h_diff = PathHeuristics.difficulty_heuristic(current, goal, student_level)
        h_time = PathHeuristics.time_heuristic(current, goal, hours_per_week)
        h_skill = PathHeuristics.skill_match_heuristic(student_skills, current.skills)
        
        return (
            weights['difficulty'] * h_diff +
            weights['time'] * h_time +
            weights['skill'] * h_skill
        )


# ============================================
# Q-LEARNING FOR PATH OPTIMIZATION
# ============================================

class QLearningPathOptimizer:
    """
    Q-Learning for adaptive learning path optimization.
    
    State: (current_course, completed_courses_hash, student_level)
    Action: next_course to take
    Reward: based on completion, difficulty match, time efficiency
    """
    
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.2
    ):
        self.kg = knowledge_graph
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        
        # Q-table: state -> action -> value
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Track visit counts for UCB exploration
        self.visit_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def _state_key(self, current: str, completed: Set[str], level: float) -> str:
        """Create hashable state key."""
        completed_hash = hash(frozenset(completed))
        level_bucket = int(level * 10)  # Discretize level
        return f"{current}|{completed_hash}|{level_bucket}"
    
    def get_valid_actions(self, current: str, completed: Set[str]) -> List[str]:
        """Get valid next courses (prerequisites satisfied)."""
        valid = []
        for node_id in self.kg.graph.nodes():
            if node_id in completed:
                continue
            prereqs = set(self.kg.get_prerequisites(node_id))
            if prereqs.issubset(completed):
                valid.append(node_id)
        return valid
    
    def select_action(self, state_key: str, valid_actions: List[str]) -> str:
        """Select action using epsilon-greedy with UCB exploration bonus."""
        if not valid_actions:
            return None
        
        if np.random.random() < self.epsilon:
            # Explore
            return np.random.choice(valid_actions)
        
        # Exploit with UCB bonus
        total_visits = sum(self.visit_counts[state_key].values()) + 1
        
        best_action = None
        best_value = float('-inf')
        
        for action in valid_actions:
            q_val = self.q_table[state_key][action]
            visits = self.visit_counts[state_key][action] + 1
            ucb_bonus = np.sqrt(2 * np.log(total_visits) / visits)
            value = q_val + ucb_bonus
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def calculate_reward(
        self,
        course: CourseNode,
        student_level: float,
        time_spent: float,
        completed: bool
    ) -> float:
        """Calculate reward for taking a course."""
        reward = 0.0
        
        # Completion reward
        if completed:
            reward += 10.0
        
        # Difficulty match reward (optimal when slightly challenging)
        optimal_difficulty = student_level + 0.1
        difficulty_match = 1.0 - abs(course.difficulty - optimal_difficulty)
        reward += difficulty_match * 5.0
        
        # Time efficiency reward
        expected_time = course.estimated_hours
        efficiency = expected_time / max(time_spent, 1)
        reward += min(efficiency, 2.0) * 3.0
        
        return reward
    
    def update(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        next_valid_actions: List[str]
    ):
        """Q-learning update."""
        # Get max Q for next state
        if next_valid_actions:
            max_next_q = max(
                self.q_table[next_state][a] for a in next_valid_actions
            )
        else:
            max_next_q = 0.0
        
        # Q-learning update
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )
        
        # Update visit count
        self.visit_counts[state][action] += 1
    
    def train(self, n_episodes: int = 1000, student_profiles: List[Dict] = None):
        """Train Q-learning on simulated episodes."""
        print(f"[RL] Training Q-learning for {n_episodes} episodes...")
        
        if student_profiles is None:
            # Generate synthetic student profiles
            student_profiles = [
                {'level': np.random.uniform(0.2, 0.8), 'hours_per_week': np.random.randint(5, 20)}
                for _ in range(100)
            ]
        
        for episode in range(n_episodes):
            # Sample a student profile
            profile = np.random.choice(student_profiles)
            student_level = profile['level']
            
            # Start from entry point
            entry_points = self.kg.get_entry_points()
            current = np.random.choice(entry_points)
            completed = set()
            
            # Episode loop
            for step in range(20):  # Max 20 steps
                state_key = self._state_key(current, completed, student_level)
                valid_actions = self.get_valid_actions(current, completed)
                
                if not valid_actions:
                    break
                
                # Select and take action
                action = self.select_action(state_key, valid_actions)
                if action is None:
                    break
                
                # Simulate taking the course
                course = self.kg.nodes[action]
                time_spent = course.estimated_hours * np.random.uniform(0.8, 1.2)
                difficulty_gap = np.clip(course.difficulty - student_level, 0, 0.5)
                completed_course = np.random.random() > difficulty_gap
                
                # Calculate reward
                reward = self.calculate_reward(course, student_level, time_spent, completed_course)
                
                # Update state
                if completed_course:
                    completed.add(action)
                    student_level = min(1.0, student_level + 0.05)  # Level up
                
                next_state_key = self._state_key(action, completed, student_level)
                next_valid = self.get_valid_actions(action, completed)
                
                # Q-learning update
                self.update(state_key, action, reward, next_state_key, next_valid)
                
                current = action
            
            # Decay exploration
            self.epsilon = max(0.05, self.epsilon * 0.995)
        
        print(f"[RL] Training complete. Q-table size: {len(self.q_table)}")
    
    def get_optimal_path(
        self,
        start_courses: List[str],
        goal_course: str,
        completed: Set[str],
        student_level: float
    ) -> Tuple[List[str], float]:
        """Get optimal path using learned Q-values."""
        path = []
        current_completed = completed.copy()
        total_q = 0.0
        
        # Start from the first available course
        current = start_courses[0] if start_courses else self.kg.get_entry_points()[0]
        
        for _ in range(20):  # Max path length
            state_key = self._state_key(current, current_completed, student_level)
            valid_actions = self.get_valid_actions(current, current_completed)
            
            if not valid_actions or goal_course in current_completed:
                break
            
            # Select best action (no exploration)
            best_action = max(valid_actions, key=lambda a: self.q_table[state_key][a])
            q_val = self.q_table[state_key][best_action]
            
            path.append(best_action)
            current_completed.add(best_action)
            total_q += q_val
            current = best_action
        
        return path, total_q


# ============================================
# A* SEARCH
# ============================================

class AStarPathFinder:
    """
    A* Search for finding optimal learning paths.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
    
    def find_path(
        self,
        start: str,
        goal: str,
        student_level: float,
        student_skills: Set[str],
        hours_per_week: float
    ) -> Tuple[List[str], float]:
        """
        Find optimal path from start to goal using A*.
        
        Returns:
            (path, total_cost)
        """
        if start not in self.kg.nodes or goal not in self.kg.nodes:
            return [], float('inf')
        
        # Priority queue: (f_score, g_score, node, path)
        start_node = self.kg.nodes[start]
        goal_node = self.kg.nodes[goal]
        
        h_start = PathHeuristics.combined_heuristic(
            start_node, goal_node, student_level, student_skills, hours_per_week
        )
        
        open_set = [(h_start, 0, start, [start])]
        closed_set = set()
        g_scores = {start: 0}
        
        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)
            
            if current == goal:
                return path, g_score
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Explore neighbors (successors in course graph)
            for neighbor in self.kg.get_successors(current):
                if neighbor in closed_set:
                    continue
                
                neighbor_node = self.kg.nodes[neighbor]
                
                # Edge cost based on difficulty and time
                edge_cost = neighbor_node.difficulty * 10 + neighbor_node.estimated_hours / 10
                tentative_g = g_score + edge_cost
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    
                    h = PathHeuristics.combined_heuristic(
                        neighbor_node, goal_node, student_level, student_skills, hours_per_week
                    )
                    f = tentative_g + h
                    
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (f, tentative_g, neighbor, new_path))
        
        # No path found, return topological path through prerequisites
        prereqs = self.kg.get_all_prerequisites(goal)
        prereqs.add(goal)
        topo = [n for n in self.kg.topological_order() if n in prereqs]
        return topo, float('inf')


# ============================================
# PATH PLANNING AGENT
# ============================================

class PathPlanningAgent:
    """
    Learning Path Planning Agent using:
    - Knowledge Graph (NetworkX)
    - A* Search with heuristics
    - Q-Learning for optimization
    - Topological sort for prerequisites
    """
    
    def __init__(self, train_rl: bool = True):
        """Initialize Path Planning Agent."""
        print("\n[Path Planning] Initializing...")
        
        # Knowledge Graph
        self.kg = KnowledgeGraph()
        print(f"[Path Planning] Knowledge Graph: {len(self.kg.nodes)} courses")
        
        # A* Path Finder
        self.astar = AStarPathFinder(self.kg)
        print("[Path Planning] ✓ A* Search ready")
        
        # Q-Learning Optimizer
        self.rl_optimizer = QLearningPathOptimizer(self.kg)
        if train_rl:
            self.rl_optimizer.train(n_episodes=500)
        print("[Path Planning] ✓ Q-Learning ready")
        
        # LLM for personalization
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        self.chat_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        
        print("[Path Planning] ✓ Agent initialized")
    
    def plan_path(
        self,
        student_id: int,
        goal_course: str,
        student_profile: Dict,
        completed_courses: List[str] = None,
        target_weeks: int = 12,
        hours_per_week: float = 10
    ) -> LearningPath:
        """
        Plan optimal learning path for a student.
        
        Uses multiple algorithms and selects the best:
        1. A* Search for shortest path
        2. Q-Learning for adaptive optimization
        3. Topological sort for prerequisite ordering
        """
        print(f"\n[Path Planning] Planning for Student {student_id} → {goal_course}")
        
        completed = set(completed_courses or [])
        student_level = self._estimate_level(student_profile)
        student_skills = self._extract_skills(completed)
        
        # Method 1: A* Search
        print("[Path Planning] Running A* Search...")
        astar_path, astar_cost = self._plan_with_astar(
            goal_course, student_level, student_skills, hours_per_week, completed
        )
        
        # Method 2: Q-Learning
        print("[Path Planning] Running Q-Learning...")
        rl_path, rl_q_value = self._plan_with_rl(
            goal_course, student_level, completed
        )
        
        # Method 3: Topological (baseline)
        print("[Path Planning] Computing Topological order...")
        topo_path = self._plan_with_topological(goal_course, completed)
        
        # Select best path based on scoring
        best_path, best_score, algorithm = self._select_best_path(
            [(astar_path, astar_cost, 'astar'),
             (rl_path, rl_q_value, 'qlearning'),
             (topo_path, 0, 'topological')],
            student_level,
            hours_per_week
        )
        
        print(f"[Path Planning] Selected: {algorithm} (score: {best_score:.2f})")
        
        # Convert to PathSteps
        steps = self._create_path_steps(best_path, target_weeks, hours_per_week)
        
        # Generate personalization notes
        personalization = self._generate_personalization(
            student_profile, best_path, algorithm
        )
        
        total_hours = sum(self.kg.nodes[n].estimated_hours for n in best_path if n in self.kg.nodes)
        
        return LearningPath(
            student_id=student_id,
            steps=steps,
            total_weeks=len(steps),
            total_hours=total_hours,
            path_score=best_score,
            q_value=rl_q_value if algorithm == 'qlearning' else 0,
            algorithm_used=algorithm,
            personalization_notes=personalization
        )
    
    def _estimate_level(self, profile: Dict) -> float:
        """Estimate student level from profile."""
        avg_score = profile.get('avg_score', 50)
        engagement = profile.get('engagement_level', 'Medium')
        
        level = avg_score / 100
        if engagement == 'High':
            level = min(1.0, level + 0.1)
        elif engagement == 'Low':
            level = max(0.1, level - 0.1)
        
        return level
    
    def _extract_skills(self, completed: Set[str]) -> Set[str]:
        """Extract skills from completed courses."""
        skills = set()
        for course_id in completed:
            if course_id in self.kg.nodes:
                skills.update(self.kg.nodes[course_id].skills)
        return skills
    
    def _plan_with_astar(
        self,
        goal: str,
        student_level: float,
        skills: Set[str],
        hours_per_week: float,
        completed: Set[str]
    ) -> Tuple[List[str], float]:
        """Plan path using A* search."""
        # Find best entry point
        entry_points = self.kg.get_entry_points()
        available_starts = [e for e in entry_points if e not in completed]
        
        if not available_starts:
            # Start from completed courses' successors
            available_starts = []
            for c in completed:
                available_starts.extend(self.kg.get_successors(c))
            available_starts = [s for s in available_starts if s not in completed]
        
        if not available_starts:
            available_starts = [goal]
        
        # Try each start, pick best path
        best_path = []
        best_cost = float('inf')
        
        for start in available_starts[:3]:  # Limit search
            path, cost = self.astar.find_path(
                start, goal, student_level, skills, hours_per_week
            )
            if cost < best_cost:
                best_cost = cost
                best_path = path
        
        return best_path, best_cost
    
    def _plan_with_rl(
        self,
        goal: str,
        student_level: float,
        completed: Set[str]
    ) -> Tuple[List[str], float]:
        """Plan path using Q-learning."""
        entry_points = [e for e in self.kg.get_entry_points() if e not in completed]
        if not entry_points:
            entry_points = list(self.kg.nodes.keys())[:1]
        
        return self.rl_optimizer.get_optimal_path(
            entry_points, goal, completed, student_level
        )
    
    def _plan_with_topological(
        self,
        goal: str,
        completed: Set[str]
    ) -> List[str]:
        """Plan path using topological sort."""
        # Get all prerequisites for goal
        prereqs = self.kg.get_all_prerequisites(goal)
        prereqs.add(goal)
        
        # Filter out completed
        needed = prereqs - completed
        
        # Return in topological order
        topo = self.kg.topological_order()
        return [n for n in topo if n in needed]
    
    def _select_best_path(
        self,
        candidates: List[Tuple[List[str], float, str]],
        student_level: float,
        hours_per_week: float
    ) -> Tuple[List[str], float, str]:
        """Select best path from candidates."""
        best_path = []
        best_score = float('-inf')
        best_algo = 'topological'
        
        for path, raw_score, algorithm in candidates:
            if not path:
                continue
            
            # Score based on multiple factors
            score = 0.0
            
            # Path length (shorter is better, but not too short)
            ideal_length = max(3, int(student_level * 10))
            length_score = 10 - abs(len(path) - ideal_length)
            score += length_score
            
            # Difficulty progression (gradual increase is good)
            if len(path) > 1:
                difficulties = [self.kg.nodes[n].difficulty for n in path if n in self.kg.nodes]
                if difficulties:
                    progression = all(difficulties[i] <= difficulties[i+1] + 0.2 
                                     for i in range(len(difficulties)-1))
                    score += 10 if progression else 0
            
            # Algorithm-specific bonus
            if algorithm == 'astar':
                score += 5 - raw_score * 0.1  # A* cost is penalty
            elif algorithm == 'qlearning':
                score += raw_score * 0.5  # Q-value is reward
            
            if score > best_score:
                best_score = score
                best_path = path
                best_algo = algorithm
        
        # Fallback
        if not best_path:
            best_path = self.kg.topological_order()[:5]
            best_algo = 'topological'
            best_score = 0
        
        return best_path, best_score, best_algo
    
    def _create_path_steps(
        self,
        path: List[str],
        target_weeks: int,
        hours_per_week: float
    ) -> List[PathStep]:
        """Convert path to detailed steps."""
        steps = []
        current_week = 1
        accumulated_hours = 0
        
        for i, course_id in enumerate(path):
            if course_id not in self.kg.nodes:
                continue
            
            course = self.kg.nodes[course_id]
            
            # Calculate weeks for this course
            course_weeks = max(1, int(course.estimated_hours / hours_per_week))
            
            # Determine activities based on course type
            activities = self._suggest_activities(course)
            
            # Rationale
            rationale = f"Builds on {', '.join(course.prerequisites[:2])} " if course.prerequisites else "Foundation course "
            rationale += f"(difficulty: {course.difficulty:.1f})"
            
            steps.append(PathStep(
                node_id=course_id,
                node_name=course.name,
                week=current_week,
                estimated_hours=course.estimated_hours,
                activities=activities,
                rationale=rationale,
                is_milestone=(i == len(path) - 1 or (i + 1) % 3 == 0)
            ))
            
            current_week += course_weeks
            accumulated_hours += course.estimated_hours
        
        return steps
    
    def _suggest_activities(self, course: CourseNode) -> List[str]:
        """Suggest activities based on course characteristics."""
        activities = ["Read course materials"]
        
        if course.difficulty > 0.6:
            activities.append("Practice exercises")
            activities.append("Discussion forums")
        
        if "programming" in course.skills or "data" in course.id.lower():
            activities.append("Coding assignments")
        
        if "research" in course.skills:
            activities.append("Literature review")
        
        activities.append("Self-assessment quiz")
        
        return activities[:4]  # Limit to 4 activities
    
    def _generate_personalization(
        self,
        profile: Dict,
        path: List[str],
        algorithm: str
    ) -> str:
        """Generate personalized notes using LLM."""
        try:
            path_names = [self.kg.nodes[n].name for n in path if n in self.kg.nodes]
            
            prompt = f"""Generate a brief, encouraging note (2-3 sentences) for a student about their personalized learning path.

Student Profile:
- Learning Style: {profile.get('learning_style', 'Unknown')}
- Engagement Level: {profile.get('engagement_level', 'Medium')}
- Current Level: {profile.get('avg_score', 50)}%

Recommended Path: {' → '.join(path_names[:5])}

Algorithm used: {algorithm} (A* for optimal pathing, Q-Learning for adaptive learning, Topological for prerequisites)

Be encouraging and explain why this path suits them."""

            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a supportive learning advisor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"This path was optimized using {algorithm} to match your learning profile."
    
    def run(
        self,
        student_id: int,
        goal_course: str,
        student_profile: Dict,
        completed_courses: List[str] = None,
        target_weeks: int = 12,
        hours_per_week: float = 10
    ) -> LearningPath:
        """Main entry point."""
        print("\n" + "=" * 60)
        print("PATH PLANNING AGENT - Graph Search + RL + Heuristics")
        print("=" * 60)
        
        path = self.plan_path(
            student_id=student_id,
            goal_course=goal_course,
            student_profile=student_profile,
            completed_courses=completed_courses,
            target_weeks=target_weeks,
            hours_per_week=hours_per_week
        )
        
        print("\n" + "=" * 60)
        print("[Path Planning] ✓ AGENT COMPLETE")
        print("=" * 60)
        
        return path


# ============================================
# STANDALONE TEST
# ============================================

if __name__ == "__main__":
    print("Testing Path Planning Agent...")
    
    # Initialize agent (trains RL)
    agent = PathPlanningAgent(train_rl=True)
    
    # Test planning
    result = agent.run(
        student_id=12345,
        goal_course="MACHINE_LEARN",
        student_profile={
            'learning_style': 'Visual Learner',
            'engagement_level': 'Medium',
            'avg_score': 65
        },
        completed_courses=["INTRO_COMP", "INTRO_MATH"],
        target_weeks=16,
        hours_per_week=12
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("LEARNING PATH RESULT")
    print("=" * 60)
    
    print(f"\nStudent: {result.student_id}")
    print(f"Algorithm: {result.algorithm_used}")
    print(f"Path Score: {result.path_score:.2f}")
    print(f"Q-Value: {result.q_value:.2f}")
    print(f"Total Weeks: {result.total_weeks}")
    print(f"Total Hours: {result.total_hours:.1f}")
    
    print("\n[Path Steps]")
    for step in result.steps:
        milestone = "🎯" if step.is_milestone else "  "
        print(f"{milestone} Week {step.week}: {step.node_name}")
        print(f"     Hours: {step.estimated_hours}, Activities: {len(step.activities)}")
        print(f"     Rationale: {step.rationale}")
    
    print(f"\n[Personalization]")
    print(f"  {result.personalization_notes}")