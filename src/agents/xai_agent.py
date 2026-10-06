"""
XAI Agent - Explainable AI for E-Learning Recommendations
=========================================================
REAL Implementation using:
- SHAP (TreeExplainer / KernelExplainer)
- LIME (LimeTabularExplainer)  
- DiCE (Counterfactual Explanations)

Author: Ahmed
Project: Explainable Multi-Agent Generative Recommendation System
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# XAI Libraries - REAL IMPLEMENTATIONS
import shap
import lime
import lime.lime_tabular
import dice_ml
from dice_ml import Dice

# LLM for natural language explanations
from openai import AzureOpenAI

import warnings
warnings.filterwarnings('ignore')

load_dotenv()


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class SHAPExplanation:
    """SHAP-based feature explanation."""
    feature_name: str
    shap_value: float
    base_value: float
    contribution: str  # 'positive' or 'negative'
    magnitude: str     # 'high', 'medium', 'low'


@dataclass
class LIMEExplanation:
    """LIME-based local explanation."""
    feature_name: str
    weight: float
    condition: str     # e.g., "total_clicks > 500"
    direction: str     # 'supports' or 'opposes'


@dataclass
class CounterfactualExplanation:
    """DiCE counterfactual explanation."""
    original_outcome: str
    desired_outcome: str
    changes: List[Dict[str, Any]]  # [{feature, original, counterfactual, change}]
    feasibility_score: float
    narrative: str


@dataclass
class CompleteExplanation:
    """Aggregated explanation from all XAI methods."""
    student_id: int
    prediction: str
    prediction_proba: float
    
    # SHAP explanations
    shap_explanations: List[SHAPExplanation]
    shap_base_value: float
    
    # LIME explanations  
    lime_explanations: List[LIMEExplanation]
    lime_score: float
    
    # Counterfactual
    counterfactual: Optional[CounterfactualExplanation]
    
    # Natural language summary
    natural_language: str
    
    # Meta
    confidence: float
    explanation_method: str  # 'ensemble' when using all methods


# ============================================
# XAI AGENT CLASS
# ============================================

class XAIAgent:
    """
    Explainable AI Agent using REAL SHAP, LIME, and DiCE.
    
    Provides three types of explanations:
    1. SHAP - Global and local feature importance
    2. LIME - Local interpretable explanations
    3. DiCE - Counterfactual explanations
    """
    
    def __init__(self):
        """Initialize XAI Agent with optional Azure OpenAI for NL generation."""
        self.client = None
        self.chat_model = None
        
        # Try to initialize Azure OpenAI (optional - for natural language explanations)
        if os.getenv("AZURE_OPENAI_API_KEY"):
            try:
                self.client = AzureOpenAI(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
                )
                self.chat_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
                print("[XAI Agent] ✓ LLM available for natural language explanations")
            except Exception as e:
                print(f"[XAI Agent] ⚠ LLM not available: {e}")
        
        # ML Model for explanations
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Feature configuration
        self.feature_names = [
            'total_clicks', 'num_sessions', 'avg_score', 'active_days',
            'resource', 'oucontent', 'forumng', 'quiz', 'subpage', 'url',
            'days_before_deadline', 'assessment_submissions'
        ]
        self.categorical_features = []
        self.continuous_features = self.feature_names.copy()
        
        # XAI explainers (initialized after training)
        self.shap_explainer = None
        self.lime_explainer = None
        self.dice_explainer = None
        
        # Training data reference
        self.X_train = None
        self.y_train = None
        self.df_train = None
        
        print("[XAI Agent] Initialized with SHAP, LIME, DiCE support")
    
    # ============================================
    # MODEL TRAINING
    # ============================================
    
    def train_model(self, features_df: pd.DataFrame, target_col: str = 'final_result') -> Dict:
        """
        Train the prediction model and initialize all XAI explainers.
        
        Args:
            features_df: DataFrame with student features
            target_col: Target column name
            
        Returns:
            Training metrics
        """
        print("\n[XAI] Training explainable model...")
        
        # Prepare features
        available_features = [f for f in self.feature_names if f in features_df.columns]
        self.feature_names = available_features
        
        X = features_df[available_features].fillna(0).copy()
        y = features_df[target_col].fillna('Unknown')
        
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store training data for explainers
        self.X_train = pd.DataFrame(X_train_scaled, columns=available_features)
        self.y_train = y_train
        self.df_train = features_df.copy()
        
        # Train Gradient Boosting (works well with SHAP TreeExplainer)
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)
        
        print(f"[XAI] Model trained - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
        
        # Initialize XAI explainers
        self._init_shap_explainer(X_train_scaled)
        self._init_lime_explainer(X_train_scaled)
        self._init_dice_explainer(X_train, y_train, features_df[target_col])
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'n_features': len(available_features),
            'n_samples': len(X_train)
        }
    
    def _init_shap_explainer(self, X_train: np.ndarray):
        """Initialize SHAP TreeExplainer."""
        print("[XAI] Initializing SHAP TreeExplainer...")
        try:
            # TreeExplainer for tree-based models (fast)
            self.shap_explainer = shap.TreeExplainer(self.model)
            print("[XAI] ✓ SHAP TreeExplainer ready")
        except Exception as e:
            print(f"[XAI] TreeExplainer failed, using KernelExplainer: {e}")
            # Fallback to KernelExplainer (slower but universal)
            background = shap.sample(X_train, min(100, len(X_train)))
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                background
            )
            print("[XAI] ✓ SHAP KernelExplainer ready")
    
    def _init_lime_explainer(self, X_train: np.ndarray):
        """Initialize LIME TabularExplainer."""
        print("[XAI] Initializing LIME TabularExplainer...")
        
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=self.feature_names,
            class_names=self.label_encoder.classes_.tolist(),
            mode='classification',
            discretize_continuous=True
        )
        print("[XAI] ✓ LIME TabularExplainer ready")
    
    def _init_dice_explainer(self, X_train: pd.DataFrame, y_train: np.ndarray, y_original: pd.Series):
        """Initialize DiCE for counterfactual explanations."""
        print("[XAI] Initializing DiCE Counterfactual Explainer...")
        
        try:
            # Prepare data for DiCE
            df_dice = X_train.copy()
            df_dice['outcome'] = self.label_encoder.inverse_transform(y_train)
            
            # Define data interface
            d = dice_ml.Data(
                dataframe=df_dice,
                continuous_features=self.feature_names,
                outcome_name='outcome'
            )
            
            # Define model interface
            m = dice_ml.Model(model=self.model, backend='sklearn')
            
            # Create DiCE explainer
            self.dice_explainer = Dice(d, m, method='random')
            print("[XAI] ✓ DiCE Counterfactual Explainer ready")
            
        except Exception as e:
            print(f"[XAI] ⚠ DiCE initialization failed: {e}")
            self.dice_explainer = None
    
    # ============================================
    # SHAP EXPLANATIONS
    # ============================================
    
    def explain_with_shap(self, student_features: Dict, top_k: int = 5) -> Tuple[List[SHAPExplanation], float]:
        """
        Generate SHAP explanations for a student.
        
        Args:
            student_features: Dict of feature values
            top_k: Number of top features to return
            
        Returns:
            List of SHAP explanations and base value
        """
        if self.shap_explainer is None:
            return [], 0.0
        
        # Prepare input
        X = self._prepare_input(student_features)
        
        # Get SHAP values
        shap_values = self.shap_explainer.shap_values(X)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            # For multi-class, use the positive class (Pass/Distinction)
            shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        # Ensure shap_vals is 1D
        shap_vals = np.array(shap_vals).flatten()
        
        # Get base value
        if hasattr(self.shap_explainer, 'expected_value'):
            base_value = self.shap_explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(np.array(base_value).flatten()[0])
            else:
                base_value = float(base_value)
        else:
            base_value = 0.5
        
        # Create explanations
        explanations = []
        feature_importance = list(zip(self.feature_names, shap_vals.tolist()))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feature, shap_val in feature_importance[:top_k]:
            magnitude = 'high' if abs(shap_val) > 0.1 else ('medium' if abs(shap_val) > 0.05 else 'low')
            
            explanations.append(SHAPExplanation(
                feature_name=feature,
                shap_value=float(shap_val),
                base_value=float(base_value),
                contribution='positive' if shap_val > 0 else 'negative',
                magnitude=magnitude
            ))
        
        return explanations, float(base_value)
    
    # ============================================
    # LIME EXPLANATIONS
    # ============================================
    
    def explain_with_lime(self, student_features: Dict, top_k: int = 5) -> Tuple[List[LIMEExplanation], float]:
        """
        Generate LIME explanations for a student.
        
        Args:
            student_features: Dict of feature values
            top_k: Number of top features to return
            
        Returns:
            List of LIME explanations and local score
        """
        if self.lime_explainer is None:
            return [], 0.0
        
        # Prepare input
        X = self._prepare_input(student_features)
        
        # Get LIME explanation
        exp = self.lime_explainer.explain_instance(
            X[0],
            self.model.predict_proba,
            num_features=top_k,
            top_labels=1
        )
        
        # Extract explanations
        explanations = []
        label = list(exp.local_exp.keys())[0]
        
        for feature_idx, weight in exp.local_exp[label][:top_k]:
            feature_name = self.feature_names[feature_idx]
            feature_value = X[0][feature_idx]
            
            # Create readable condition
            condition = f"{feature_name} = {feature_value:.2f}"
            
            explanations.append(LIMEExplanation(
                feature_name=feature_name,
                weight=float(weight),
                condition=condition,
                direction='supports' if weight > 0 else 'opposes'
            ))
        
        # Get local prediction score
        local_score = exp.local_pred[0] if hasattr(exp, 'local_pred') else 0.5
        
        return explanations, float(local_score)
    
    # ============================================
    # COUNTERFACTUAL EXPLANATIONS (DiCE)
    # ============================================
    
    def explain_with_counterfactual(
        self, 
        student_features: Dict, 
        desired_outcome: str = 'Pass',
        n_counterfactuals: int = 3
    ) -> Optional[CounterfactualExplanation]:
        """
        Generate counterfactual explanations using DiCE.
        
        Args:
            student_features: Dict of feature values
            desired_outcome: Target outcome to achieve
            n_counterfactuals: Number of counterfactuals to generate
            
        Returns:
            Counterfactual explanation or None
        """
        if self.dice_explainer is None:
            return self._fallback_counterfactual(student_features, desired_outcome)
        
        try:
            # Prepare input as DataFrame
            X_df = pd.DataFrame([student_features])[self.feature_names].fillna(0)
            X_scaled = self.scaler.transform(X_df)
            X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
            
            # Get current prediction
            current_pred = self.label_encoder.inverse_transform(
                self.model.predict(X_scaled)
            )[0]
            
            # Generate counterfactuals
            cf = self.dice_explainer.generate_counterfactuals(
                X_scaled_df,
                total_CFs=n_counterfactuals,
                desired_class=desired_outcome
            )
            
            # Extract changes
            changes = []
            if cf.cf_examples_list and len(cf.cf_examples_list) > 0:
                cf_example = cf.cf_examples_list[0]
                if cf_example.final_cfs_df is not None and len(cf_example.final_cfs_df) > 0:
                    cf_row = cf_example.final_cfs_df.iloc[0]
                    
                    for feature in self.feature_names:
                        original_val = X_scaled_df[feature].iloc[0]
                        cf_val = cf_row[feature]
                        
                        if abs(original_val - cf_val) > 0.01:
                            # Inverse transform to original scale
                            changes.append({
                                'feature': feature,
                                'original': float(original_val),
                                'counterfactual': float(cf_val),
                                'change': float(cf_val - original_val)
                            })
            
            # Generate narrative
            narrative = self._generate_cf_narrative(current_pred, desired_outcome, changes)
            
            return CounterfactualExplanation(
                original_outcome=current_pred,
                desired_outcome=desired_outcome,
                changes=changes[:5],  # Top 5 changes
                feasibility_score=0.8 if changes else 0.0,
                narrative=narrative
            )
            
        except Exception as e:
            print(f"[XAI] DiCE error: {e}, using fallback")
            return self._fallback_counterfactual(student_features, desired_outcome)
    
    def _fallback_counterfactual(
        self, 
        student_features: Dict, 
        desired_outcome: str
    ) -> CounterfactualExplanation:
        """Fallback counterfactual when DiCE fails."""
        
        # Define target benchmarks for "Pass" outcome
        benchmarks = {
            'total_clicks': 800,
            'num_sessions': 150,
            'avg_score': 70,
            'active_days': 25,
            'forumng': 40,
            'quiz': 15,
            'assessment_submissions': 8
        }
        
        changes = []
        for feature, target in benchmarks.items():
            current = student_features.get(feature, 0)
            if current < target * 0.7:
                changes.append({
                    'feature': feature,
                    'original': current,
                    'counterfactual': target,
                    'change': target - current
                })
        
        changes.sort(key=lambda x: abs(x['change']), reverse=True)
        narrative = self._generate_cf_narrative('current', desired_outcome, changes[:3])
        
        return CounterfactualExplanation(
            original_outcome='At Risk',
            desired_outcome=desired_outcome,
            changes=changes[:5],
            feasibility_score=0.6,
            narrative=narrative
        )
    
    def _generate_cf_narrative(self, current: str, desired: str, changes: List[Dict]) -> str:
        """Generate human-readable counterfactual narrative."""
        if not changes:
            return f"Your profile is already aligned with {desired} outcomes."
        
        suggestions = []
        for c in changes[:3]:
            feature = c['feature'].replace('_', ' ')
            if c['change'] > 0:
                suggestions.append(f"increase {feature} by {abs(c['change']):.0f}")
            else:
                suggestions.append(f"maintain {feature}")
        
        return f"To move from {current} to {desired}: {', '.join(suggestions)}."
    
    # ============================================
    # NATURAL LANGUAGE EXPLANATION
    # ============================================
    
    def generate_natural_explanation(
        self,
        student_profile: Dict,
        shap_explanations: List[SHAPExplanation],
        lime_explanations: List[LIMEExplanation],
        counterfactual: Optional[CounterfactualExplanation],
        recommendations: List[Dict]
    ) -> str:
        """
        Use GPT-4o to generate a cohesive natural language explanation.
        """
        # Format SHAP insights
        shap_text = "\n".join([
            f"- {e.feature_name}: {e.contribution} impact ({e.magnitude})"
            for e in shap_explanations[:3]
        ])
        
        # Format LIME insights
        lime_text = "\n".join([
            f"- {e.condition}: {e.direction} prediction"
            for e in lime_explanations[:3]
        ])
        
        # Counterfactual text
        cf_text = counterfactual.narrative if counterfactual else "No counterfactual needed."
        
        # Convert recommendations to JSON-safe format
        safe_recs = []
        if recommendations:
            for rec in recommendations[:3]:
                safe_rec = {}
                for k, v in rec.items():
                    if hasattr(v, 'item'):  # numpy type
                        safe_rec[k] = v.item()
                    else:
                        safe_rec[k] = v
                safe_recs.append(safe_rec)
        
        prompt = f"""Generate a clear, encouraging explanation for a student about their personalized learning recommendations.

STUDENT PROFILE:
- Learning Style: {student_profile.get('learning_style', 'Unknown')}
- Engagement Level: {student_profile.get('engagement_level', 'Unknown')}
- Current Performance: {student_profile.get('avg_score', 'N/A')}%

SHAP ANALYSIS (what factors matter most):
{shap_text}

LIME ANALYSIS (local explanation):
{lime_text}

COUNTERFACTUAL INSIGHT:
{cf_text}

RECOMMENDATIONS:
{json.dumps(safe_recs, indent=2) if safe_recs else 'General learning activities'}

Write a 3-4 sentence personalized explanation that:
1. Acknowledges their learning style
2. Explains why these recommendations fit them (using the SHAP/LIME insights)
3. Provides encouragement and actionable next steps

Keep it warm, clear, and non-technical."""

        try:
            if self.client is None:
                raise Exception("No LLM client available")
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a supportive learning advisor explaining personalized recommendations to students."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[XAI] LLM not available, using template explanation")
            # Generate explanation without LLM
            top_feature = shap_explanations[0].feature_name if shap_explanations else "engagement"
            return f"Based on your {student_profile.get('learning_style', 'learning')} style and {top_feature.replace('_', ' ')} patterns, we've selected activities that match your learning profile. {cf_text}"
    
    # ============================================
    # MAIN EXPLANATION METHOD
    # ============================================
    
    def explain(
        self,
        student_id: int,
        student_features: Dict,
        student_profile: Dict,
        recommendations: List[Dict] = None
    ) -> CompleteExplanation:
        """
        Generate complete explanation using SHAP, LIME, and DiCE.
        
        Args:
            student_id: Student identifier
            student_features: Dict of feature values
            student_profile: Dict with learning style, engagement level
            recommendations: List of recommendation dicts
            
        Returns:
            CompleteExplanation with all XAI methods
        """
        print(f"\n[XAI] Generating explanations for Student {student_id}...")
        
        # Get prediction
        X = self._prepare_input(student_features)
        prediction_idx = self.model.predict(X)[0]
        prediction = self.label_encoder.inverse_transform([prediction_idx])[0]
        prediction_proba = float(np.max(self.model.predict_proba(X)))
        
        # SHAP explanations
        print("[XAI] Running SHAP...")
        shap_exps, shap_base = self.explain_with_shap(student_features)
        
        # LIME explanations
        print("[XAI] Running LIME...")
        lime_exps, lime_score = self.explain_with_lime(student_features)
        
        # Counterfactual (only if at risk)
        print("[XAI] Running Counterfactual...")
        counterfactual = None
        if prediction in ['Fail', 'Withdrawn', 'At Risk']:
            counterfactual = self.explain_with_counterfactual(student_features, 'Pass')
        
        # Natural language
        print("[XAI] Generating natural language explanation...")
        natural_lang = self.generate_natural_explanation(
            student_profile, shap_exps, lime_exps, counterfactual, recommendations or []
        )
        
        # Calculate confidence from agreement between methods
        shap_top = shap_exps[0].feature_name if shap_exps else None
        lime_top = lime_exps[0].feature_name if lime_exps else None
        agreement_bonus = 0.1 if shap_top == lime_top else 0
        confidence = min(0.95, prediction_proba + agreement_bonus)
        
        print(f"[XAI] ✓ Explanation complete (confidence: {confidence:.1%})")
        
        return CompleteExplanation(
            student_id=student_id,
            prediction=prediction,
            prediction_proba=prediction_proba,
            shap_explanations=shap_exps,
            shap_base_value=shap_base,
            lime_explanations=lime_exps,
            lime_score=lime_score,
            counterfactual=counterfactual,
            natural_language=natural_lang,
            confidence=confidence,
            explanation_method='ensemble'
        )
    
    # ============================================
    # UTILITY METHODS
    # ============================================
    
    def _prepare_input(self, student_features: Dict) -> np.ndarray:
        """Prepare feature dict as scaled numpy array."""
        X = pd.DataFrame([student_features])[self.feature_names].fillna(0)
        return self.scaler.transform(X)
    
    def run(
        self,
        student_id: int,
        student_features: Dict,
        student_profile: Dict,
        recommendations: List[Dict] = None,
        features_df: pd.DataFrame = None
    ) -> CompleteExplanation:
        """
        Main entry point for XAI Agent.
        """
        print("\n" + "=" * 60)
        print("XAI AGENT - SHAP + LIME + DiCE")
        print("=" * 60)
        
        # Train model if needed
        if self.model is None and features_df is not None:
            self.train_model(features_df)
        
        explanation = self.explain(
            student_id=student_id,
            student_features=student_features,
            student_profile=student_profile,
            recommendations=recommendations
        )
        
        print("\n" + "=" * 60)
        print("[XAI] ✓ AGENT COMPLETE")
        print("=" * 60)
        
        return explanation


# ============================================
# STANDALONE TEST
# ============================================

if __name__ == "__main__":
    print("Testing XAI Agent with REAL SHAP, LIME, DiCE...")
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 500
    
    test_df = pd.DataFrame({
        'total_clicks': np.random.randint(100, 2000, n_samples),
        'num_sessions': np.random.randint(10, 200, n_samples),
        'avg_score': np.random.randint(30, 100, n_samples),
        'active_days': np.random.randint(5, 60, n_samples),
        'resource': np.random.randint(0, 100, n_samples),
        'oucontent': np.random.randint(0, 500, n_samples),
        'forumng': np.random.randint(0, 100, n_samples),
        'quiz': np.random.randint(0, 50, n_samples),
        'subpage': np.random.randint(0, 200, n_samples),
        'url': np.random.randint(0, 50, n_samples),
        'days_before_deadline': np.random.randint(-10, 30, n_samples),
        'assessment_submissions': np.random.randint(0, 15, n_samples),
    })
    
    # Create target based on features
    test_df['final_result'] = np.where(
        (test_df['avg_score'] > 60) & (test_df['total_clicks'] > 500),
        'Pass',
        np.where(test_df['avg_score'] > 40, 'Fail', 'Withdrawn')
    )
    
    # Initialize and train
    xai = XAIAgent()
    metrics = xai.train_model(test_df)
    print(f"\nTraining metrics: {metrics}")
    
    # Test student
    test_student = {
        'total_clicks': 300,
        'num_sessions': 50,
        'avg_score': 45,
        'active_days': 15,
        'resource': 20,
        'oucontent': 100,
        'forumng': 10,
        'quiz': 5,
        'subpage': 30,
        'url': 5,
        'days_before_deadline': 5,
        'assessment_submissions': 3
    }
    
    explanation = xai.run(
        student_id=12345,
        student_features=test_student,
        student_profile={
            'learning_style': 'Visual Learner',
            'engagement_level': 'Low',
            'avg_score': 45
        },
        recommendations=[
            {'activity': 'quiz', 'reason': 'Practice assessment'},
            {'activity': 'forumng', 'reason': 'Peer discussion'}
        ]
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("EXPLANATION RESULTS")
    print("=" * 60)
    
    print(f"\nPrediction: {explanation.prediction} ({explanation.prediction_proba:.1%})")
    
    print("\n[SHAP Explanations]")
    for s in explanation.shap_explanations[:3]:
        print(f"  {s.feature_name}: {s.shap_value:+.3f} ({s.contribution}, {s.magnitude})")
    
    print("\n[LIME Explanations]")
    for l in explanation.lime_explanations[:3]:
        print(f"  {l.condition}: {l.direction} ({l.weight:+.3f})")
    
    if explanation.counterfactual:
        print(f"\n[Counterfactual]")
        print(f"  {explanation.counterfactual.narrative}")
        for c in explanation.counterfactual.changes[:3]:
            print(f"  - {c['feature']}: {c['original']:.1f} → {c['counterfactual']:.1f}")
    
    print(f"\n[Natural Language]")
    print(f"  {explanation.natural_language}")
    
    print(f"\nConfidence: {explanation.confidence:.1%}")