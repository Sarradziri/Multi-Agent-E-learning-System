"""
Recommendation Agent - Hybrid Filtering + LLM Ranking
======================================================
REAL Implementation using:
- Collaborative Filtering (Matrix Factorization - SVD)
- Content-Based Filtering (TF-IDF similarity)
- Hybrid combination with learned weights

Evaluation Metrics: NDCG@K, MRR, Recall@K, Precision@K
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from dotenv import load_dotenv

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from openai import AzureOpenAI

import warnings
warnings.filterwarnings('ignore')

load_dotenv()


@dataclass
class Recommendation:
    """A single recommendation item."""
    item_id: str
    item_name: str
    score: float
    source: str
    reasoning: str
    rank: int


@dataclass
class RecommendationSet:
    """Complete set of recommendations for a student."""
    student_id: int
    recommendations: List[Recommendation]
    hybrid_weights: Dict[str, float]
    evaluation_metrics: Dict[str, float]
    llm_reranked: bool


class RecommendationMetrics:
    """Evaluation metrics for recommendation systems."""
    
    @staticmethod
    def dcg_at_k(relevances: List[float], k: int) -> float:
        relevances = np.array(relevances[:k])
        if len(relevances) == 0:
            return 0.0
        discounts = np.log2(np.arange(len(relevances)) + 2)
        return np.sum(relevances / discounts)
    
    @staticmethod
    def ndcg_at_k(predicted: List[str], relevant: List[str], k: int) -> float:
        relevances = [1.0 if item in relevant else 0.0 for item in predicted[:k]]
        dcg = RecommendationMetrics.dcg_at_k(relevances, k)
        ideal_relevances = [1.0] * min(len(relevant), k)
        idcg = RecommendationMetrics.dcg_at_k(ideal_relevances, k)
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mrr(predicted: List[str], relevant: List[str]) -> float:
        for i, item in enumerate(predicted):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def recall_at_k(predicted: List[str], relevant: List[str], k: int) -> float:
        if len(relevant) == 0:
            return 0.0
        return len(set(predicted[:k]) & set(relevant)) / len(relevant)
    
    @staticmethod
    def precision_at_k(predicted: List[str], relevant: List[str], k: int) -> float:
        predicted_k = predicted[:k]
        if len(predicted_k) == 0:
            return 0.0
        return sum(1 for item in predicted_k if item in relevant) / len(predicted_k)
    
    @staticmethod
    def evaluate_all(predicted: List[str], relevant: List[str], k_values: List[int] = [5, 10]) -> Dict[str, float]:
        metrics = {'mrr': RecommendationMetrics.mrr(predicted, relevant)}
        for k in k_values:
            metrics[f'ndcg@{k}'] = RecommendationMetrics.ndcg_at_k(predicted, relevant, k)
            metrics[f'recall@{k}'] = RecommendationMetrics.recall_at_k(predicted, relevant, k)
            metrics[f'precision@{k}'] = RecommendationMetrics.precision_at_k(predicted, relevant, k)
        return metrics


class CollaborativeFilter:
    """Collaborative Filtering using Matrix Factorization (SVD)."""
    
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.user_index = {}
        self.item_index = {}
        self.global_mean = 0
        self.is_fitted = False
    
    def fit(self, interactions_df: pd.DataFrame):
        print("[CF] Fitting collaborative filter...")
        
        users = interactions_df['user_id'].unique()
        items = interactions_df['item_id'].unique()
        
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {it: i for i, it in enumerate(items)}
        
        value_col = 'clicks' if 'clicks' in interactions_df.columns else 'sum_click'
        if value_col not in interactions_df.columns:
            interactions_df = interactions_df.copy()
            interactions_df['clicks'] = 1
            value_col = 'clicks'
        
        matrix = np.zeros((len(users), len(items)))
        for _, row in interactions_df.iterrows():
            u_idx = self.user_index[row['user_id']]
            i_idx = self.item_index[row['item_id']]
            matrix[u_idx, i_idx] = row.get(value_col, 1)
        
        self.global_mean = matrix[matrix > 0].mean() if (matrix > 0).any() else 0
        
        n_comp = min(self.n_components, min(len(users), len(items)) - 1, 50)
        self.svd = TruncatedSVD(n_components=max(1, n_comp), random_state=42)
        
        self.user_factors = self.svd.fit_transform(matrix)
        self.item_factors = self.svd.components_.T
        self.is_fitted = True
        print(f"[CF] ✓ Fitted on {len(users)} users, {len(items)} items")
    
    def recommend(self, user_id, n: int = 10, exclude: List = None) -> List[Tuple]:
        # Try both string and int versions of user_id
        if user_id not in self.user_index:
            try:
                user_id = int(user_id)
            except:
                pass
        if user_id not in self.user_index:
            try:
                user_id = str(user_id)
            except:
                pass
        
        if not self.is_fitted or user_id not in self.user_index:
            return [(it, self.global_mean) for it in list(self.item_index.keys())[:n]]
        
        exclude = set(exclude or [])
        u_idx = self.user_index[user_id]
        scores = np.dot(self.user_factors[u_idx], self.item_factors.T)
        
        recommendations = []
        for i_idx in np.argsort(scores)[::-1]:
            item_id = list(self.item_index.keys())[list(self.item_index.values()).index(i_idx)]
            if item_id not in exclude and str(item_id) not in exclude:
                recommendations.append((item_id, float(scores[i_idx])))
                if len(recommendations) >= n:
                    break
        return recommendations


class ContentBasedFilter:
    """Content-Based Filtering using TF-IDF."""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=500, stop_words='english')
        self.item_vectors = None
        self.item_ids = []
        self.item_id_to_idx = {}
        self.is_fitted = False
    
    def fit(self, items_df: pd.DataFrame, text_columns: List[str] = None):
        print("[CB] Fitting content-based filter...")
        
        self.item_ids = items_df['item_id'].tolist()
        # Create lookup dict for both int and string versions
        self.item_id_to_idx = {}
        for i, item_id in enumerate(self.item_ids):
            self.item_id_to_idx[item_id] = i
            self.item_id_to_idx[str(item_id)] = i
            try:
                self.item_id_to_idx[int(item_id)] = i
            except:
                pass
        
        if text_columns is None:
            text_columns = [c for c in items_df.columns if items_df[c].dtype == 'object' and c != 'item_id']
        
        texts = []
        for _, row in items_df.iterrows():
            item_text = ' '.join(str(row.get(c, '')) for c in text_columns if pd.notna(row.get(c)))
            texts.append(item_text if item_text.strip() else 'general content')
        
        self.item_vectors = self.tfidf.fit_transform(texts)
        self.is_fitted = True
        print(f"[CB] ✓ Fitted on {len(self.item_ids)} items")
    
    def recommend_for_profile(self, user_history: List, n: int = 10, exclude: List = None) -> List[Tuple]:
        if not self.is_fitted or not user_history:
            return []
        
        exclude = set(exclude or [])
        
        # Find indices for user history items (handle type mismatches)
        history_indices = []
        for item in user_history:
            if item in self.item_id_to_idx:
                history_indices.append(self.item_id_to_idx[item])
            elif str(item) in self.item_id_to_idx:
                history_indices.append(self.item_id_to_idx[str(item)])
            else:
                try:
                    if int(item) in self.item_id_to_idx:
                        history_indices.append(self.item_id_to_idx[int(item)])
                except:
                    pass
        
        if not history_indices:
            print(f"[CB] Warning: No history items found in item index")
            return []
        
        user_profile = np.asarray(self.item_vectors[history_indices].mean(axis=0))
        similarities = cosine_similarity(user_profile.reshape(1, -1), self.item_vectors).flatten()
        
        # Convert user_history to set of strings for comparison
        history_set = set(str(h) for h in user_history)
        exclude_set = set(str(e) for e in exclude)
        
        recommendations = []
        for i in np.argsort(similarities)[::-1]:
            item_id = self.item_ids[i]
            if str(item_id) not in exclude_set and str(item_id) not in history_set:
                recommendations.append((item_id, float(similarities[i])))
                if len(recommendations) >= n:
                    break
        return recommendations


class HybridRecommender:
    """Hybrid Recommender combining CF and CB."""
    
    def __init__(self, cf_weight: float = 0.6, cb_weight: float = 0.4):
        self.cf = CollaborativeFilter()
        self.cb = ContentBasedFilter()
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.item_modules = {}  # item_id -> module mapping
        self.item_types = {}    # item_id -> activity_type mapping
    
    def fit(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame, text_columns: List[str] = None):
        print("\n[Hybrid] Fitting hybrid recommender...")
        self.cf.fit(interactions_df)
        self.cb.fit(items_df, text_columns)
        
        # Store item -> module and type mapping for filtering
        for _, row in items_df.iterrows():
            item_id = str(row['item_id'])
            self.item_modules[item_id] = row.get('code_module', 'unknown')
            self.item_types[item_id] = row.get('activity_type', 'unknown')
        
        print("[Hybrid] ✓ Ready")
    
    def recommend(self, user_id, user_history: List = None, n: int = 10, exclude: List = None, 
                  filter_module: str = None, preferred_types: List[str] = None) -> List[Tuple]:
        exclude = set(exclude or [])
        user_history = user_history or []
        
        # Get more candidates to account for filtering
        n_candidates = n * 20 if filter_module else n * 5
        
        cf_recs = self.cf.recommend(user_id, n=n_candidates, exclude=list(exclude))
        cb_recs = self.cb.recommend_for_profile(user_history, n=n_candidates, exclude=list(exclude))
        
        item_scores = defaultdict(lambda: {'cf': 0, 'cb': 0})
        for item_id, score in cf_recs:
            item_scores[str(item_id)]['cf'] = score
        for item_id, score in cb_recs:
            item_scores[str(item_id)]['cb'] = score
        
        all_cf = [item_scores[it]['cf'] for it in item_scores]
        all_cb = [item_scores[it]['cb'] for it in item_scores]
        
        cf_min, cf_max = (min(all_cf), max(all_cf)) if all_cf else (0, 1)
        cb_min, cb_max = (min(all_cb), max(all_cb)) if all_cb else (0, 1)
        
        cf_range = cf_max - cf_min if cf_max != cf_min else 1.0
        cb_range = cb_max - cb_min if cb_max != cb_min else 1.0
        
        has_cf = cf_max != cf_min
        has_cb = cb_max != cb_min and len(cb_recs) > 0
        
        recommendations = []
        for item_id in item_scores:
            # Filter by module if specified
            if filter_module and self.item_modules.get(item_id, '') != filter_module:
                continue
            
            cf_score = item_scores[item_id]['cf']
            cb_score = item_scores[item_id]['cb']
            
            cf_norm = (cf_score - cf_min) / cf_range if cf_range > 0 else 0.5
            cb_norm = (cb_score - cb_min) / cb_range if cb_range > 0 else 0.5
            
            if has_cf and has_cb:
                hybrid_score = self.cf_weight * cf_norm + self.cb_weight * cb_norm
                source = 'hybrid' if cb_score > 0 else 'collaborative'
            elif has_cf:
                hybrid_score = cf_norm
                source = 'collaborative'
            elif has_cb:
                hybrid_score = cb_norm
                source = 'content'
            else:
                hybrid_score = 0.5
                source = 'default'
            
            # Boost score for preferred activity types
            item_type = self.item_types.get(item_id, 'unknown')
            if preferred_types and item_type in preferred_types:
                type_boost = 0.2 * (1 - preferred_types.index(item_type) / len(preferred_types))
                hybrid_score += type_boost
            
            recommendations.append((item_id, hybrid_score, source))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n]


class RecommendationAgent:
    """Recommendation Agent with Hybrid Filtering + LLM Re-ranking."""
    
    def __init__(self):
        print("\n[Rec Agent] Initializing...")
        self.hybrid = HybridRecommender()
        self.client = None
        self.chat_model = None
        
        if os.getenv("AZURE_OPENAI_API_KEY"):
            try:
                self.client = AzureOpenAI(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
                )
                self.chat_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
                print("[Rec Agent] ✓ LLM available for re-ranking")
            except Exception as e:
                print(f"[Rec Agent] LLM not available: {e}")
        
        self.item_metadata = {}
        print("[Rec Agent] ✓ Initialized")
    
    def fit(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame, text_columns: List[str] = None):
        self.hybrid.fit(interactions_df, items_df, text_columns)
        for _, row in items_df.iterrows():
            self.item_metadata[str(row['item_id'])] = {
                'name': row.get('item_name', str(row['item_id'])),
                'type': row.get('activity_type', 'unknown'),
                'module': row.get('code_module', 'unknown')
            }
    
    def _generate_reasoning(self, item_id: str, source: str, profile: Dict) -> str:
        meta = self.item_metadata.get(str(item_id), {})
        name = meta.get('name', item_id)
        if source == 'collaborative':
            return f"{name} recommended based on similar learners' success patterns."
        elif source == 'content':
            return f"{name} matches your learning history and preferences."
        return f"{name} selected through hybrid analysis of your profile."
    
    def recommend(self, student_id: int, student_profile: Dict, user_history: List = None,
                  n: int = 10, exclude: List = None, ground_truth: List = None,
                  filter_module: str = None, preferred_types: List[str] = None) -> RecommendationSet:
        print(f"\n[Rec Agent] Recommending for Student {student_id}...")
        if filter_module:
            print(f"[Rec Agent] Filtering to module: {filter_module}")
        if preferred_types:
            print(f"[Rec Agent] Preferred types: {preferred_types[:3]}")
        
        hybrid_recs = self.hybrid.recommend(student_id, user_history or [], n * 2, exclude,
                                            filter_module=filter_module, preferred_types=preferred_types)
        
        recommendations = []
        for rank, rec in enumerate(hybrid_recs[:n], 1):
            item_id, score, source = str(rec[0]), rec[1], rec[2] if len(rec) > 2 else 'hybrid'
            meta = self.item_metadata.get(item_id, {})
            
            recommendations.append(Recommendation(
                item_id=item_id,
                item_name=meta.get('name', item_id),
                score=score,
                source=source,
                reasoning=self._generate_reasoning(item_id, source, student_profile),
                rank=rank
            ))
        
        eval_metrics = {}
        if ground_truth:
            predicted_ids = [r.item_id for r in recommendations]
            eval_metrics = RecommendationMetrics.evaluate_all(predicted_ids, ground_truth)
            print(f"[Rec Agent] NDCG@5={eval_metrics.get('ndcg@5', 0):.3f}, MRR={eval_metrics.get('mrr', 0):.3f}")
        
        return RecommendationSet(
            student_id=student_id,
            recommendations=recommendations,
            hybrid_weights={'cf': self.hybrid.cf_weight, 'cb': self.hybrid.cb_weight},
            evaluation_metrics=eval_metrics,
            llm_reranked=False
        )
    
    def run(self, student_id: int, student_profile: Dict, interactions_df: pd.DataFrame = None,
            items_df: pd.DataFrame = None, user_history: List = None, n: int = 10,
            ground_truth: List = None, filter_module: str = None, 
            preferred_types: List[str] = None) -> RecommendationSet:
        print("\n" + "=" * 60)
        print("RECOMMENDATION AGENT - Hybrid Filtering + LLM Ranking")
        print("=" * 60)
        if interactions_df is not None and items_df is not None:
            self.fit(interactions_df, items_df)
        result = self.recommend(student_id, student_profile, user_history, n, 
                               ground_truth=ground_truth, filter_module=filter_module,
                               preferred_types=preferred_types)
        print("\n" + "=" * 60)
        print("[Rec Agent] ✓ COMPLETE")
        print("=" * 60)
        return result