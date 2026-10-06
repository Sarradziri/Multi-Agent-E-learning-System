"""
E-Learning Multi-Agent System - Main Entry Point
=================================================
Usage: python main.py

Author: Ahmed
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8 console output: agents log ✓ / ⚠ which crash on Windows cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.profiling_agent import ProfilingAgent, LearnerProfile
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.xai_agent import XAIAgent
from src.agents.path_planning_agent import PathPlanningAgent
from src.agents.content_generator import ContentGeneratorAgent
from src.orchestrator import MultiAgentOrchestrator
from src.evaluation.metrics import ComprehensiveEvaluator, RecommendationEvaluator

load_dotenv()


def load_oulad_data(data_path: str = "data/oulad") -> dict:
    """Load OULAD dataset files."""
    print("\n[Data] Loading OULAD dataset...")
    
    data = {}
    files = {
        'student_info': 'studentInfo.csv',
        'student_vle': 'studentVle.csv',
        'student_assessment': 'studentAssessment.csv',
        'vle': 'vle.csv',
        'assessments': 'assessments.csv',
        'courses': 'courses.csv'
    }
    
    for key, filename in files.items():
        filepath = os.path.join(data_path, filename)
        if os.path.exists(filepath):
            data[key] = pd.read_csv(filepath)
            print(f"  ✓ {filename}: {len(data[key])} rows")
        else:
            print(f"  ⚠ {filename} not found")
            data[key] = pd.DataFrame()
    
    return data


def create_synthetic_data() -> dict:
    """Create synthetic data for testing without OULAD."""
    print("\n[Data] Creating synthetic test data...")
    
    np.random.seed(42)
    n_students = 500
    n_items = 100
    
    student_info = pd.DataFrame({
        'id_student': range(1, n_students + 1),
        'code_module': np.random.choice(['AAA', 'BBB', 'CCC'], n_students),
        'code_presentation': np.random.choice(['2013J', '2014B'], n_students),
        'final_result': np.random.choice(['Pass', 'Fail', 'Withdrawn', 'Distinction'], n_students, p=[0.4, 0.2, 0.2, 0.2])
    })
    
    vle_records = []
    for student_id in range(1, n_students + 1):
        n_interactions = np.random.randint(10, 200)
        for _ in range(n_interactions):
            vle_records.append({
                'id_student': student_id,
                'id_site': np.random.randint(1, n_items + 1),
                'sum_click': np.random.randint(1, 50)
            })
    student_vle = pd.DataFrame(vle_records)
    
    assessment_records = []
    for student_id in range(1, n_students + 1):
        n_assessments = np.random.randint(3, 10)
        for _ in range(n_assessments):
            assessment_records.append({
                'id_student': student_id,
                'id_assessment': np.random.randint(1, 50),
                'score': np.random.randint(20, 100)
            })
    student_assessment = pd.DataFrame(assessment_records)
    
    vle = pd.DataFrame({
        'id_site': range(1, n_items + 1),
        'activity_type': np.random.choice(['oucontent', 'quiz', 'forumng', 'resource', 'url', 'page'], n_items),
        'code_module': np.random.choice(['AAA', 'BBB', 'CCC'], n_items)
    })
    
    print(f"  ✓ Students: {n_students}")
    print(f"  ✓ VLE interactions: {len(student_vle)}")
    print(f"  ✓ Assessments: {len(student_assessment)}")
    print(f"  ✓ Items: {n_items}")
    
    return {
        'student_info': student_info,
        'student_vle': student_vle,
        'student_assessment': student_assessment,
        'vle': vle
    }


def create_train_test_split(student_vle: pd.DataFrame, student_id: int, test_ratio: float = 0.2):
    """
    Split student interactions into train/test for proper evaluation.
    Uses temporal split - later interactions are test set.
    """
    student_data = student_vle[student_vle['id_student'] == student_id].copy()
    
    if 'date' in student_data.columns:
        student_data = student_data.sort_values('date')
    
    n_test = max(1, int(len(student_data) * test_ratio))
    
    train_data = student_data.iloc[:-n_test]
    test_data = student_data.iloc[-n_test:]
    
    ground_truth_items = test_data['id_site'].unique().tolist()
    train_history = train_data['id_site'].unique().tolist()
    
    return train_history, ground_truth_items


def run_individual_agents(data: dict, student_id: int = 1):
    """Run each agent individually with proper evaluation."""
    
    print("\n" + "=" * 70)
    print("RUNNING INDIVIDUAL AGENTS")
    print("=" * 70)
    
    # =========================================
    # 1. PROFILING AGENT
    # =========================================
    print("\n" + "-" * 50)
    print("1. PROFILING AGENT (Embeddings + Clustering)")
    print("-" * 50)
    
    profiling_agent = ProfilingAgent()
    features, clusters, embeddings, descriptions = profiling_agent.run(
        student_info=data['student_info'],
        student_vle=data['student_vle'],
        student_assessments=data['student_assessment'],
        n_clusters=5,
        sample_size=200
    )
    
    student_ids = data['student_info']['id_student'].tolist()[:len(clusters)]
    profile = profiling_agent.profile_student(student_id, features, clusters, student_ids)
    
    print(f"\nStudent {student_id} Profile:")
    print(f"  Cluster: {profile.cluster_id} - {profile.learning_style}")
    print(f"  Engagement: {profile.engagement_level}")
    print(f"  Features: {profile.features}")
    
    # =========================================
    # 2. RECOMMENDATION AGENT WITH PROPER EVALUATION
    # =========================================
    print("\n" + "-" * 50)
    print("2. RECOMMENDATION AGENT (Hybrid CF + CB + LLM)")
    print("-" * 50)
    
    print("  [Eval] Creating train/test split...")
    train_history, ground_truth = create_train_test_split(data['student_vle'], student_id, test_ratio=0.2)
    print(f"  [Eval] Train history: {len(train_history)} items")
    print(f"  [Eval] Ground truth (future): {len(ground_truth)} items")
    
    # Detect student's module for filtering
    student_module = data['student_vle'][data['student_vle']['id_student'] == student_id]['code_module'].iloc[0]
    print(f"  [Eval] Student's module: {student_module}")
    
    # Detect student's preferred activity types (what they interact with most)
    student_items = data['student_vle'][data['student_vle']['id_student'] == student_id]['id_site'].tolist()
    student_vle_types = data['vle'][data['vle']['id_site'].isin(student_items)]['activity_type'].value_counts()
    preferred_types = student_vle_types.index.tolist()[:5]  # Top 5 activity types
    print(f"  [Eval] Student's preferred types: {preferred_types}")
    
    student_vle_sample = data['student_vle']
    if len(student_vle_sample) > 500000:
        print(f"  [Sampling] {len(student_vle_sample)} rows → ~500,000 rows for speed")
        target_interactions = student_vle_sample[student_vle_sample['id_student'] == student_id]
        other_interactions = student_vle_sample[student_vle_sample['id_student'] != student_id]
        sample_size = min(500000 - len(target_interactions), len(other_interactions))
        other_sample = other_interactions.sample(n=sample_size, random_state=42)
        student_vle_sample = pd.concat([target_interactions, other_sample], ignore_index=True)
        print(f"  [Sampling] Target student has {len(target_interactions)} interactions")
    
    interactions_df = student_vle_sample.rename(columns={
        'id_student': 'user_id',
        'id_site': 'item_id',
        'sum_click': 'clicks'
    })
    
    items_df = data['vle'].rename(columns={'id_site': 'item_id'})
    items_df['item_name'] = items_df['activity_type'] + '_' + items_df['item_id'].astype(str)
    
    user_history = [str(h) for h in train_history[:20]]
    ground_truth_str = [str(g) for g in ground_truth]
    
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
        ground_truth=ground_truth_str,
        filter_module=student_module,
        preferred_types=preferred_types  # Boost preferred activity types!
    )
    
    print(f"\nTop 10 Recommendations:")
    for rec in rec_result.recommendations[:10]:
        in_ground_truth = "✓ HIT" if rec.item_id in ground_truth_str else ""
        print(f"  {rec.rank}. {rec.item_name} ({rec.source}) - Score: {rec.score:.3f} {in_ground_truth}")
    
    predicted_ids = [r.item_id for r in rec_result.recommendations]
    rec_metrics = RecommendationEvaluator.evaluate(predicted_ids, ground_truth_str, k_values=[5, 10])
    
    print(f"\n  [Eval] Recommendation Metrics (against future interactions):")
    print(f"    MRR: {rec_metrics['mrr']:.3f}")
    print(f"    NDCG@5: {rec_metrics['ndcg@5']:.3f}")
    print(f"    NDCG@10: {rec_metrics['ndcg@10']:.3f}")
    print(f"    Recall@5: {rec_metrics['recall@5']:.3f}")
    print(f"    Recall@10: {rec_metrics['recall@10']:.3f}")
    
    # =========================================
    # 3. XAI AGENT
    # =========================================
    print("\n" + "-" * 50)
    print("3. XAI AGENT (SHAP + LIME + DiCE)")
    print("-" * 50)
    
    xai_agent = XAIAgent()
    
    student_features = {
        'total_clicks': profile.features.get('total_clicks', 100),
        'avg_score': profile.features.get('avg_score', 50),
        'num_sessions': profile.features.get('num_sessions', 20),
        'num_assessments': 5
    }
    
    xai_result = xai_agent.run(
        student_id=student_id,
        student_features=student_features,
        student_profile={
            'learning_style': profile.learning_style,
            'engagement_level': profile.engagement_level
        },
        features_df=features,
        recommendations=[{'item_id': r.item_id, 'item_name': r.item_name} for r in rec_result.recommendations[:3]]
    )
    
    print(f"\nExplanation Confidence: {xai_result.confidence:.2f}")
    print(f"Prediction: {xai_result.prediction}")
    
    print(f"\nSHAP Top Features:")
    if xai_result.shap_explanations:
        for exp in xai_result.shap_explanations[:3]:
            direction = "positive" if exp.shap_value > 0 else "negative"
            print(f"  - {exp.feature_name}: {exp.shap_value:.4f} ({direction})")
    
    print(f"\nLIME Top Features:")
    if xai_result.lime_explanations:
        for exp in xai_result.lime_explanations[:3]:
            print(f"  - {exp.feature_name}: {exp.weight:.4f}")
    
    print(f"\nNatural Language Explanation:")
    print(f"  {xai_result.natural_language[:300]}...")
    
    # =========================================
    # 4. PATH PLANNING AGENT
    # =========================================
    print("\n" + "-" * 50)
    print("4. PATH PLANNING AGENT (A* + Q-Learning)")
    print("-" * 50)
    
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
    
    print(f"\nLearning Path ({path_result.algorithm_used}):")
    print(f"  Total Weeks: {path_result.total_weeks}")
    print(f"  Total Hours: {path_result.total_hours}")
    print(f"  Path Score: {path_result.path_score:.2f}")
    print(f"  Q-Value: {path_result.q_value:.2f}")
    print(f"\n  Steps:")
    for step in path_result.steps[:5]:
        print(f"    Week {step.week}: {step.node_name} ({step.estimated_hours}h)")
    
    # =========================================
    # 5. CONTENT GENERATOR
    # =========================================
    print("\n" + "-" * 50)
    print("5. CONTENT GENERATOR (LLM + RAG)")
    print("-" * 50)
    
    content_agent = ContentGeneratorAgent()
    
    topics = [step.node_name for step in path_result.steps[:2]]
    content_result = content_agent.run(
        topics=topics,
        student_profile={'learning_style': profile.learning_style},
        content_types=['quiz', 'summary']
    )
    
    print(f"\nGenerated Content:")
    for quiz in content_result['quizzes']:
        print(f"  Quiz on '{quiz.topic}': {len(quiz.questions)} questions")
    for summary in content_result['summaries']:
        print(f"  Summary on '{summary.topic}': {len(summary.key_points)} key points")
    
    # =========================================
    # 6. FINAL EVALUATION SUMMARY
    # =========================================
    print("\n" + "-" * 50)
    print("6. EVALUATION SUMMARY")
    print("-" * 50)
    
    print(f"\n  RECOMMENDATION METRICS (Holdout Evaluation):")
    print(f"    Ground Truth Size: {len(ground_truth)} items")
    print(f"    MRR: {rec_metrics['mrr']:.3f}")
    print(f"    NDCG@5: {rec_metrics['ndcg@5']:.3f}")
    print(f"    NDCG@10: {rec_metrics['ndcg@10']:.3f}")
    print(f"    Recall@5: {rec_metrics['recall@5']:.3f}")
    print(f"    Recall@10: {rec_metrics['recall@10']:.3f}")
    print(f"    Precision@5: {rec_metrics['precision@5']:.3f}")
    print(f"    Precision@10: {rec_metrics['precision@10']:.3f}")
    
    print(f"\n  XAI METRICS:")
    print(f"    Confidence: {xai_result.confidence:.3f}")
    print(f"    Methods: SHAP + LIME + {'DiCE' if xai_result.counterfactual else 'No CF'}")
    
    print(f"\n  PATH PLANNING METRICS:")
    print(f"    Algorithm: {path_result.algorithm_used}")
    print(f"    Path Score: {path_result.path_score:.2f}")
    print(f"    Q-Value: {path_result.q_value:.2f}")
    
    return {
        'profile': profile,
        'recommendations': rec_result,
        'rec_metrics': rec_metrics,
        'explanations': xai_result,
        'learning_path': path_result,
        'content': content_result
    }


def main():
    """Main entry point."""
    
    print("\n" + "=" * 70)
    print("E-LEARNING MULTI-AGENT RECOMMENDATION SYSTEM")
    print("=" * 70)
    print("\nTechnologies:")
    print("  • Profiling: Embeddings + K-Means Clustering + LLM")
    print("  • Recommendations: Hybrid CF (SVD) + CB (TF-IDF) + LLM Ranking")
    print("  • Explainability: SHAP + LIME + DiCE Counterfactuals")
    print("  • Path Planning: NetworkX Graph + A* Search + Q-Learning")
    print("  • Content Generation: LLM + RAG")
    print("  • Orchestration: LangGraph + FIPA-ACL Messaging")
    print("=" * 70)
    
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("\n⚠ WARNING: AZURE_OPENAI_API_KEY not set!")
    
    if os.path.exists("data/oulad/studentInfo.csv"):
        data = load_oulad_data("data/oulad")
    else:
        print("\n⚠ OULAD data not found, using synthetic data...")
        data = create_synthetic_data()
    
    student_interaction_counts = data['student_vle'].groupby('id_student').size()
    students_with_enough_data = student_interaction_counts[student_interaction_counts >= 50].index.tolist()
    
    if students_with_enough_data:
        student_id = students_with_enough_data[0]
    else:
        student_id = data['student_info']['id_student'].iloc[0]
    
    print(f"\n[Target Student: {student_id}]")
    
    results = run_individual_agents(data, student_id)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()