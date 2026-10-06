"""
Quick XAI Agent Test
====================
Tests SHAP, LIME, DiCE without running the full system
"""

import numpy as np
import pandas as pd
from src.agents.xai_agent import XAIAgent

print("=" * 60)
print("XAI AGENT STANDALONE TEST")
print("=" * 60)

# Create synthetic student data (no need for OULAD)
np.random.seed(42)
n_students = 500

features_df = pd.DataFrame({
    'id_student': range(1, n_students + 1),
    'total_clicks': np.random.randint(50, 2000, n_students),
    'avg_score': np.random.uniform(30, 100, n_students),
    'num_sessions': np.random.randint(10, 300, n_students),
    'num_assessments': np.random.randint(1, 15, n_students),
    'final_result': np.random.choice(['Pass', 'Fail', 'Withdrawn', 'Distinction'], n_students)
})

print(f"\n[Data] Created {n_students} synthetic students")

# Initialize XAI Agent
xai_agent = XAIAgent()

# Test student
student_id = 42
student_features = {
    'total_clicks': 850,
    'avg_score': 72.5,
    'num_sessions': 45,
    'num_assessments': 8
}

student_profile = {
    'learning_style': 'Visual Learner',
    'engagement_level': 'High'
}

print(f"\n[Test] Student {student_id}")
print(f"  Features: {student_features}")

# Run XAI
result = xai_agent.run(
    student_id=student_id,
    student_features=student_features,
    student_profile=student_profile,
    features_df=features_df,
    recommendations=[
        {'item_id': 'quiz1', 'item_name': 'Practice Quiz'},
        {'item_id': 'video1', 'item_name': 'Lecture Video'}
    ]
)

# Print results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\nConfidence: {result.confidence:.2f}")
print(f"Prediction: {result.prediction}")

print("\n[SHAP Explanation]")
if result.shap_explanations:
    for exp in result.shap_explanations[:5]:
        direction = "positive" if exp.shap_value > 0 else "negative"
        print(f"  • {exp.feature_name}: {exp.shap_value:.4f} ({direction})")
else:
    print("  (No SHAP results)")

print("\n[LIME Explanation]")
if result.lime_explanations:
    for exp in result.lime_explanations[:5]:
        print(f"  • {exp.feature_name}: {exp.weight:.4f}")
else:
    print("  (No LIME results)")

print("\n[Counterfactual]")
if result.counterfactual:
    print(f"  Original: {result.counterfactual.original_outcome}")
    print(f"  Desired: {result.counterfactual.desired_outcome}")
    print(f"  Feasibility: {result.counterfactual.feasibility_score:.2f}")
    print(f"  Changes needed:")
    for change in result.counterfactual.changes[:3]:
        print(f"    • {change['feature']}: {change['original']} → {change['counterfactual']}")
else:
    print("  (No counterfactual results)")

print("\n[Natural Language Explanation]")
print(f"  {result.natural_language[:300]}...")

print("\n" + "=" * 60)
print("TEST COMPLETE ✓")
print("=" * 60)