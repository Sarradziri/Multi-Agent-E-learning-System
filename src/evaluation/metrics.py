"""
Evaluation Module - Comprehensive Metrics
==========================================
Metrics for:
- Recommendations: NDCG@K, MRR, Recall@K, Precision@K
- Generation: BERTScore, ROUGE
- Explainability: Faithfulness, Plausibility

Author: Ahmed
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    recommendation_metrics: Dict[str, float]
    generation_metrics: Dict[str, float]
    xai_metrics: Dict[str, float]
    overall_score: float


# ============================================
# RECOMMENDATION METRICS
# ============================================

class RecommendationEvaluator:
    """Evaluate recommendation quality."""
    
    @staticmethod
    def dcg_at_k(relevances: List[float], k: int) -> float:
        """Discounted Cumulative Gain."""
        relevances = np.array(relevances[:k])
        if len(relevances) == 0:
            return 0.0
        discounts = np.log2(np.arange(len(relevances)) + 2)
        return float(np.sum(relevances / discounts))
    
    @staticmethod
    def ndcg_at_k(predicted: List[str], relevant: List[str], k: int) -> float:
        """Normalized DCG."""
        relevances = [1.0 if item in relevant else 0.0 for item in predicted[:k]]
        dcg = RecommendationEvaluator.dcg_at_k(relevances, k)
        ideal = [1.0] * min(len(relevant), k)
        idcg = RecommendationEvaluator.dcg_at_k(ideal, k)
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mrr(predicted: List[str], relevant: List[str]) -> float:
        """Mean Reciprocal Rank."""
        for i, item in enumerate(predicted):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def recall_at_k(predicted: List[str], relevant: List[str], k: int) -> float:
        """Recall at K."""
        if not relevant:
            return 0.0
        return len(set(predicted[:k]) & set(relevant)) / len(relevant)
    
    @staticmethod
    def precision_at_k(predicted: List[str], relevant: List[str], k: int) -> float:
        """Precision at K."""
        if k == 0:
            return 0.0
        hits = sum(1 for item in predicted[:k] if item in relevant)
        return hits / k
    
    @staticmethod
    def evaluate(predicted: List[str], relevant: List[str], 
                 k_values: List[int] = [5, 10]) -> Dict[str, float]:
        """Evaluate all recommendation metrics."""
        metrics = {'mrr': RecommendationEvaluator.mrr(predicted, relevant)}
        for k in k_values:
            metrics[f'ndcg@{k}'] = RecommendationEvaluator.ndcg_at_k(predicted, relevant, k)
            metrics[f'recall@{k}'] = RecommendationEvaluator.recall_at_k(predicted, relevant, k)
            metrics[f'precision@{k}'] = RecommendationEvaluator.precision_at_k(predicted, relevant, k)
        return metrics


# ============================================
# GENERATION METRICS
# ============================================

class GenerationEvaluator:
    """Evaluate generated content quality."""
    
    def __init__(self):
        self.bert_scorer = None
        self.rouge_scorer = None
        self._init_scorers()
    
    def _init_scorers(self):
        """Initialize BERTScore and ROUGE scorers."""
        try:
            from bert_score import BERTScorer
            self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            print("[Eval] ✓ BERTScore loaded")
        except ImportError:
            print("[Eval] ⚠ BERTScore not available")
        
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            print("[Eval] ✓ ROUGE loaded")
        except ImportError:
            print("[Eval] ⚠ ROUGE not available")
    
    def bert_score(self, candidates: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BERTScore."""
        if self.bert_scorer is None:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        P, R, F1 = self.bert_scorer.score(candidates, references)
        return {
            'precision': float(P.mean()),
            'recall': float(R.mean()),
            'f1': float(F1.mean())
        }
    
    def rouge_score(self, candidate: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE scores."""
        if self.rouge_scorer is None:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def evaluate(self, generated: str, reference: str) -> Dict[str, float]:
        """Evaluate generated text."""
        metrics = {}
        
        # ROUGE
        rouge = self.rouge_score(generated, reference)
        metrics.update({f'rouge_{k}': v for k, v in rouge.items()})
        
        # BERTScore
        bert = self.bert_score([generated], [reference])
        metrics.update({f'bert_{k}': v for k, v in bert.items()})
        
        return metrics


# ============================================
# XAI METRICS
# ============================================

class XAIEvaluator:
    """Evaluate explanation quality."""
    
    @staticmethod
    def faithfulness(explanation_features: List[str], model_features: List[str],
                     feature_importances: Dict[str, float]) -> float:
        """
        Measure if explanations reflect actual model behavior.
        Higher = explanation matches what model actually uses.
        """
        if not explanation_features or not model_features:
            return 0.0
        
        # Check overlap between explained features and important model features
        explained_set = set(explanation_features)
        
        # Get top important features from model
        sorted_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)
        top_model_features = set([f[0] for f in sorted_features[:len(explanation_features)]])
        
        overlap = len(explained_set & top_model_features)
        return overlap / len(explanation_features)
    
    @staticmethod
    def plausibility(explanation: str, domain_terms: List[str]) -> float:
        """
        Measure if explanation makes sense to humans.
        Checks for domain-relevant terminology.
        """
        if not explanation or not domain_terms:
            return 0.5
        
        explanation_lower = explanation.lower()
        matches = sum(1 for term in domain_terms if term.lower() in explanation_lower)
        
        return min(1.0, matches / max(len(domain_terms) * 0.3, 1))
    
    @staticmethod
    def consistency(explanations: List[Dict], similar_inputs: bool = True) -> float:
        """
        Measure if similar inputs get similar explanations.
        """
        if len(explanations) < 2:
            return 1.0
        
        # Compare feature rankings across explanations
        rankings = []
        for exp in explanations:
            features = exp.get('features', [])
            rankings.append([f.get('name', '') for f in features[:5]])
        
        # Calculate overlap between consecutive explanations
        overlaps = []
        for i in range(len(rankings) - 1):
            overlap = len(set(rankings[i]) & set(rankings[i+1]))
            overlaps.append(overlap / max(len(rankings[i]), 1))
        
        return np.mean(overlaps) if overlaps else 1.0
    
    @staticmethod
    def evaluate(explanations: List[Dict], model_importances: Dict[str, float],
                 domain_terms: List[str] = None) -> Dict[str, float]:
        """Evaluate XAI quality."""
        domain_terms = domain_terms or ['learning', 'score', 'engagement', 'activity', 'progress']
        
        if not explanations:
            return {'faithfulness': 0.0, 'plausibility': 0.0, 'consistency': 0.0}
        
        # Get features from first explanation
        exp_features = [f.get('name', '') for f in explanations[0].get('features', [])]
        model_features = list(model_importances.keys())
        
        faithfulness = XAIEvaluator.faithfulness(exp_features, model_features, model_importances)
        
        # Plausibility from natural language
        nl_explanation = explanations[0].get('natural_language', '')
        plausibility = XAIEvaluator.plausibility(nl_explanation, domain_terms)
        
        # Consistency across explanations
        consistency = XAIEvaluator.consistency(explanations)
        
        return {
            'faithfulness': faithfulness,
            'plausibility': plausibility,
            'consistency': consistency,
            'trust_score': (faithfulness + plausibility + consistency) / 3
        }


# ============================================
# COMPREHENSIVE EVALUATOR
# ============================================

class ComprehensiveEvaluator:
    """Combines all evaluation metrics."""
    
    def __init__(self):
        self.rec_eval = RecommendationEvaluator()
        self.gen_eval = GenerationEvaluator()
        self.xai_eval = XAIEvaluator()
    
    def evaluate_all(
        self,
        # Recommendation inputs
        predicted_recs: List[str] = None,
        relevant_recs: List[str] = None,
        # Generation inputs
        generated_text: str = None,
        reference_text: str = None,
        # XAI inputs
        explanations: List[Dict] = None,
        model_importances: Dict[str, float] = None
    ) -> EvaluationReport:
        """Run comprehensive evaluation."""
        
        # Recommendation metrics
        rec_metrics = {}
        if predicted_recs and relevant_recs:
            rec_metrics = self.rec_eval.evaluate(predicted_recs, relevant_recs)
        
        # Generation metrics
        gen_metrics = {}
        if generated_text and reference_text:
            gen_metrics = self.gen_eval.evaluate(generated_text, reference_text)
        
        # XAI metrics
        xai_metrics = {}
        if explanations:
            xai_metrics = self.xai_eval.evaluate(
                explanations, 
                model_importances or {},
            )
        
        # Overall score
        scores = []
        if rec_metrics:
            scores.append(rec_metrics.get('ndcg@5', 0))
        if gen_metrics:
            scores.append(gen_metrics.get('bert_f1', 0))
        if xai_metrics:
            scores.append(xai_metrics.get('trust_score', 0))
        
        overall = np.mean(scores) if scores else 0.0
        
        return EvaluationReport(
            recommendation_metrics=rec_metrics,
            generation_metrics=gen_metrics,
            xai_metrics=xai_metrics,
            overall_score=float(overall)
        )


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("Testing Evaluation Module...")
    
    evaluator = ComprehensiveEvaluator()
    
    # Test recommendation metrics
    predicted = ['item1', 'item2', 'item3', 'item4', 'item5']
    relevant = ['item1', 'item3', 'item7']
    
    rec_metrics = RecommendationEvaluator.evaluate(predicted, relevant)
    print(f"\nRecommendation Metrics:")
    for k, v in rec_metrics.items():
        print(f"  {k}: {v:.3f}")
    
    # Test XAI metrics
    explanations = [
        {
            'features': [{'name': 'total_clicks'}, {'name': 'avg_score'}, {'name': 'num_sessions'}],
            'natural_language': 'Based on your learning engagement and quiz scores, we recommend these activities.'
        }
    ]
    model_importances = {'total_clicks': 0.3, 'avg_score': 0.25, 'num_sessions': 0.2, 'other': 0.1}
    
    xai_metrics = XAIEvaluator.evaluate(explanations, model_importances)
    print(f"\nXAI Metrics:")
    for k, v in xai_metrics.items():
        print(f"  {k}: {v:.3f}")
    
    # Full evaluation
    report = evaluator.evaluate_all(
        predicted_recs=predicted,
        relevant_recs=relevant,
        explanations=explanations,
        model_importances=model_importances
    )
    
    print(f"\nOverall Score: {report.overall_score:.3f}")
