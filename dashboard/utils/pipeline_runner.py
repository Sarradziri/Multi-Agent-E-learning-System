"""
Pipeline Runner Utility
========================
Runs the MAS agents and returns results for the dashboard.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class PipelineRunner:
    """Runs the multi-agent pipeline with callbacks for progress updates."""
    
    def __init__(self, data: dict):
        """
        Initialize the pipeline runner.
        
        Args:
            data: Dict containing loaded OULAD dataframes
        """
        self.data = data
        self.results = {}
        
    def run_profiling(self, student_id: int, custom_profile: dict = None):
        """Run the profiling agent."""
        
        if custom_profile:
            # Use custom profile
            from dataclasses import dataclass
            
            @dataclass
            class MockProfile:
                student_id: int
                cluster_id: int
                learning_style: str
                engagement_level: str
                features: dict
            
            return MockProfile(
                student_id=student_id,
                cluster_id=0,
                learning_style=custom_profile.get('learning_style', 'High Engagement Achiever'),
                engagement_level=custom_profile.get('engagement_level', 'Medium'),
                features={
                    'total_clicks': custom_profile.get('total_clicks', 500),
                    'avg_score': custom_profile.get('avg_score', 65),
                    'num_sessions': custom_profile.get('num_sessions', 100),
                    'final_result': custom_profile.get('final_result', 'Pass')
                }
            )
        
        try:
            from src.agents.profiling_agent import ProfilingAgent
            
            profiling_agent = ProfilingAgent()
            features, clusters, embeddings, descriptions = profiling_agent.run(
                student_info=self.data['student_info'],
                student_vle=self.data['student_vle'],
                student_assessments=self.data['student_assessment'],
                n_clusters=5,
                sample_size=200
            )
            
            student_ids = self.data['student_info']['id_student'].tolist()[:len(clusters)]
            profile = profiling_agent.profile_student(student_id, features, clusters, student_ids)
            
            self.results['features'] = features
            return profile
            
        except Exception as e:
            print(f"Profiling error: {e}")
            # Return mock profile
            from dataclasses import dataclass
            
            @dataclass
            class MockProfile:
                student_id: int
                cluster_id: int
                learning_style: str
                engagement_level: str
                features: dict
            
            return MockProfile(
                student_id=student_id,
                cluster_id=0,
                learning_style="High Engagement Achiever",
                engagement_level="Medium",
                features={'total_clicks': 500, 'avg_score': 65, 'num_sessions': 100, 'final_result': 'Pass'}
            )
    
    def run_recommendations(self, student_id: int, profile):
        """Run the recommendation agent."""
        
        try:
            from src.agents.recommendation_agent import RecommendationAgent
            
            # Prepare data
            student_vle = self.data['student_vle']
            vle = self.data['vle']
            
            # Get student module
            student_module = student_vle[student_vle['id_student'] == student_id]['code_module'].iloc[0] \
                if len(student_vle[student_vle['id_student'] == student_id]) > 0 else 'AAA'
            
            # Sample interactions
            if len(student_vle) > 500000:
                target_interactions = student_vle[student_vle['id_student'] == student_id]
                other_interactions = student_vle[student_vle['id_student'] != student_id]
                sample_size = min(500000 - len(target_interactions), len(other_interactions))
                other_sample = other_interactions.sample(n=sample_size, random_state=42)
                student_vle_sample = pd.concat([target_interactions, other_sample], ignore_index=True)
            else:
                student_vle_sample = student_vle
            
            interactions_df = student_vle_sample.rename(columns={
                'id_student': 'user_id',
                'id_site': 'item_id',
                'sum_click': 'clicks'
            })
            
            items_df = vle.rename(columns={'id_site': 'item_id'})
            items_df['item_name'] = items_df['activity_type'] + '_' + items_df['item_id'].astype(str)
            
            # Get user history and preferred types
            user_history = interactions_df[interactions_df['user_id'] == student_id]['item_id'].unique().tolist()[:20]
            user_history = [str(h) for h in user_history]
            
            student_items = student_vle[student_vle['id_student'] == student_id]['id_site'].tolist()
            student_vle_types = vle[vle['id_site'].isin(student_items)]['activity_type'].value_counts()
            preferred_types = student_vle_types.index.tolist()[:5]
            
            # Create train/test split for metrics
            student_data = student_vle[student_vle['id_student'] == student_id]
            n_test = max(1, int(len(student_data) * 0.2))
            test_data = student_data.iloc[-n_test:]
            ground_truth = [str(g) for g in test_data['id_site'].unique().tolist()]
            
            # Run agent
            rec_agent = RecommendationAgent()
            rec_result = rec_agent.run(
                student_id=student_id,
                student_profile={
                    'learning_style': profile.learning_style,
                    'engagement_level': profile.engagement_level,
                    'avg_score': profile.features.get('avg_score', 50)
                },
                interactions_df=interactions_df,
                items_df=items_df,
                user_history=user_history,
                n=10,
                ground_truth=ground_truth,
                filter_module=student_module,
                preferred_types=preferred_types
            )
            
            # Mark hits
            for rec in rec_result.recommendations:
                rec.is_hit = rec.item_id in ground_truth
            
            # Get metrics
            from src.evaluation.metrics import RecommendationEvaluator
            predicted_ids = [r.item_id for r in rec_result.recommendations]
            metrics = RecommendationEvaluator.evaluate(predicted_ids, ground_truth, k_values=[5, 10])
            
            return rec_result, metrics
            
        except Exception as e:
            print(f"Recommendation error: {e}")
            import traceback
            traceback.print_exc()
            return None, {}
    
    def run_xai(self, student_id: int, profile, recommendations):
        """Run the XAI agent."""
        
        try:
            from src.agents.xai_agent import XAIAgent
            
            xai_agent = XAIAgent()
            
            student_features = {
                'total_clicks': profile.features.get('total_clicks', 100),
                'avg_score': profile.features.get('avg_score', 50),
                'num_sessions': profile.features.get('num_sessions', 20),
                'num_assessments': 5
            }
            
            features_df = self.results.get('features', pd.DataFrame())
            
            rec_list = []
            if recommendations and hasattr(recommendations, 'recommendations'):
                rec_list = [{'item_id': r.item_id, 'item_name': r.item_name} 
                           for r in recommendations.recommendations[:3]]
            
            xai_result = xai_agent.run(
                student_id=student_id,
                student_features=student_features,
                student_profile={
                    'learning_style': profile.learning_style,
                    'engagement_level': profile.engagement_level
                },
                features_df=features_df,
                recommendations=rec_list
            )
            
            return xai_result
            
        except Exception as e:
            print(f"XAI error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_path_planning(self, student_id: int, profile):
        """Run the path planning agent."""
        
        try:
            from src.agents.path_planning_agent import PathPlanningAgent
            
            path_agent = PathPlanningAgent(train_rl=True)
            
            path_result = path_agent.run(
                student_id=student_id,
                goal_course='MACHINE_LEARN',
                student_profile={
                    'learning_style': profile.learning_style,
                    'skill_level': 'Beginner' if profile.features.get('avg_score', 50) < 50 else 'Intermediate',
                    'available_hours_per_week': 10
                },
                completed_courses=[],
                target_weeks=12,
                hours_per_week=10
            )
            
            return path_result
            
        except Exception as e:
            print(f"Path planning error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_content_generation(self, learning_path):
        """Run the content generator agent."""
        
        try:
            from src.agents.content_generator import ContentGeneratorAgent
            
            content_agent = ContentGeneratorAgent()
            
            # Get topics from learning path
            topics = []
            if learning_path and hasattr(learning_path, 'steps'):
                topics = [step.node_name for step in learning_path.steps[:2]]
            elif learning_path and isinstance(learning_path, dict):
                steps = learning_path.get('steps', [])
                topics = [s.get('node_name', s.get('topic', 'Introduction')) for s in steps[:2]]
            
            if not topics:
                topics = ['Introduction to Computing', 'Programming Fundamentals']
            
            content_result = content_agent.run(
                topics=topics,
                student_profile={'learning_style': 'Visual Learner'},
                content_types=['quiz', 'summary']
            )
            
            return content_result
            
        except Exception as e:
            print(f"Content generation error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return mock content
            return {
                'quizzes': [
                    {
                        'topic': 'Introduction to Computing',
                        'difficulty': 'medium',
                        'questions': [
                            {
                                'question': 'What is a computer?',
                                'options': ['A) A machine that processes data', 'B) A type of vehicle', 'C) A musical instrument', 'D) A cooking device'],
                                'correct': 'A',
                                'explanation': 'A computer is an electronic device that processes data.'
                            }
                        ]
                    }
                ],
                'summaries': [
                    {
                        'topic': 'Introduction to Computing',
                        'summary': 'Computing is the foundation of modern technology.',
                        'key_points': ['Computers process data', 'Software controls hardware', 'Programming enables automation']
                    }
                ]
            }
