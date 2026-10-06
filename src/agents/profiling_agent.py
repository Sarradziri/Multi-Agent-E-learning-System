"""
Profiling Agent - Student Analysis and Clustering
==================================================
Implementation using:
- Embeddings (text-embedding-3-small)
- K-Means Clustering
- LLM for cluster interpretation

Author: Ahmed
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from openai import AzureOpenAI

load_dotenv()


@dataclass
class LearnerProfile:
    """Student profile from clustering."""
    student_id: int
    cluster_id: int
    learning_style: str
    engagement_level: str
    features: Dict
    embedding: Optional[List[float]] = None


class ProfilingAgent:
    """Profiling Agent using Embeddings + Clustering + LLM."""
    
    def __init__(self):
        print("\n[Profiling Agent] Initializing...")
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        self.embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        self.chat_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        
        self.scaler = StandardScaler()
        self.kmeans = None
        self.cluster_descriptions = {}
        print("[Profiling Agent] ✓ Initialized")
    
    def extract_features(self, student_info: pd.DataFrame, student_vle: pd.DataFrame,
                         student_assessments: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral features from OULAD data."""
        print("[Profiling] Extracting features...")
        
        # Aggregate VLE interactions
        if student_vle is not None and len(student_vle) > 0:
            vle_agg = student_vle.groupby('id_student').agg({
                'sum_click': ['sum', 'mean', 'count']
            }).reset_index()
            vle_agg.columns = ['id_student', 'total_clicks', 'avg_clicks', 'num_sessions']
        else:
            vle_agg = pd.DataFrame({'id_student': [], 'total_clicks': [], 'avg_clicks': [], 'num_sessions': []})
        
        # Aggregate assessments
        if student_assessments is not None and len(student_assessments) > 0:
            assess_agg = student_assessments.groupby('id_student').agg({
                'score': ['mean', 'count']
            }).reset_index()
            assess_agg.columns = ['id_student', 'avg_score', 'num_assessments']
        else:
            assess_agg = pd.DataFrame({'id_student': [], 'avg_score': [], 'num_assessments': []})
        
        # Merge with student info
        features = student_info[['id_student', 'final_result']].copy() if 'final_result' in student_info.columns else student_info[['id_student']].copy()
        features = features.merge(vle_agg, on='id_student', how='left')
        features = features.merge(assess_agg, on='id_student', how='left')
        features = features.fillna(0)
        
        print(f"[Profiling] ✓ Extracted features for {len(features)} students")
        return features
    
    def create_embeddings(self, features: pd.DataFrame, sample_size: int = 500) -> Tuple[np.ndarray, List[int]]:
        """Create embeddings from student features."""
        print("[Profiling] Creating embeddings...")
        
        # Sample if too large
        if len(features) > sample_size:
            features = features.sample(n=sample_size, random_state=42)
        
        student_ids = features['id_student'].tolist()
        
        # Create text descriptions for embedding
        texts = []
        for _, row in features.iterrows():
            text = f"Student with {row.get('total_clicks', 0):.0f} clicks, {row.get('avg_score', 0):.1f}% avg score, {row.get('num_sessions', 0):.0f} sessions"
            texts.append(text)
        
        # Get embeddings in batches
        embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(model=self.embedding_model, input=batch)
            embeddings.extend([e.embedding for e in response.data])
        
        print(f"[Profiling] ✓ Created {len(embeddings)} embeddings")
        return np.array(embeddings), student_ids
    
    def cluster_students(self, embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
        """Cluster students using K-Means."""
        print(f"[Profiling] Clustering into {n_clusters} groups...")
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(embeddings)
        
        print(f"[Profiling] ✓ Created {n_clusters} clusters")
        return clusters
    
    def describe_clusters(self, features: pd.DataFrame, clusters: np.ndarray, student_ids: List[int]) -> Dict[int, str]:
        """Use LLM to describe each cluster."""
        print("[Profiling] Generating cluster descriptions...")
        
        # Map clusters to students
        id_to_cluster = dict(zip(student_ids, clusters))
        features_subset = features[features['id_student'].isin(student_ids)].copy()
        features_subset['cluster'] = features_subset['id_student'].map(id_to_cluster)
        
        descriptions = {}
        for cluster_id in range(max(clusters) + 1):
            cluster_data = features_subset[features_subset['cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                descriptions[cluster_id] = "Unknown Learners"
                continue
            
            stats = {
                'avg_clicks': cluster_data['total_clicks'].mean(),
                'avg_score': cluster_data['avg_score'].mean(),
                'avg_sessions': cluster_data['num_sessions'].mean(),
                'count': len(cluster_data)
            }
            
            prompt = f"""Analyze this student cluster and give it a short descriptive name (2-4 words):
- Average clicks: {stats['avg_clicks']:.0f}
- Average score: {stats['avg_score']:.1f}%
- Average sessions: {stats['avg_sessions']:.0f}
- Students: {stats['count']}

Return ONLY the cluster name, nothing else."""

            try:
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                    temperature=0.5
                )
                descriptions[cluster_id] = response.choices[0].message.content.strip()
            except:
                descriptions[cluster_id] = f"Cluster {cluster_id}"
        
        self.cluster_descriptions = descriptions
        print(f"[Profiling] ✓ Described {len(descriptions)} clusters")
        return descriptions
    
    def profile_student(self, student_id: int, features: pd.DataFrame, clusters: np.ndarray,
                        student_ids: List[int] = None) -> LearnerProfile:
        """Get profile for a specific student."""
        student_row = features[features['id_student'] == student_id]
        
        if len(student_row) == 0:
            return LearnerProfile(student_id=student_id, cluster_id=0,
                                  learning_style="Unknown", engagement_level="Unknown", features={})
        
        row = student_row.iloc[0]
        
        # Find cluster
        if student_ids and student_id in student_ids:
            idx = student_ids.index(student_id)
            cluster_id = int(clusters[idx])
        else:
            cluster_id = 0
        
        # Determine engagement level
        total_clicks = row.get('total_clicks', 0)
        if total_clicks > 1000:
            engagement = "High"
        elif total_clicks > 300:
            engagement = "Medium"
        else:
            engagement = "Low"
        
        return LearnerProfile(
            student_id=student_id,
            cluster_id=cluster_id,
            learning_style=self.cluster_descriptions.get(cluster_id, f"Cluster {cluster_id}"),
            engagement_level=engagement,
            features={
                'total_clicks': row.get('total_clicks', 0),
                'avg_score': row.get('avg_score', 0),
                'num_sessions': row.get('num_sessions', 0),
                'final_result': row.get('final_result', 'Unknown')
            }
        )
    
    def run(self, student_info: pd.DataFrame, student_vle: pd.DataFrame = None,
            student_assessments: pd.DataFrame = None, n_clusters: int = 5,
            sample_size: int = 500) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict]:
        """Main entry point."""
        print("\n" + "=" * 60)
        print("PROFILING AGENT - Embeddings + Clustering + LLM")
        print("=" * 60)
        
        features = self.extract_features(student_info, student_vle, student_assessments)
        embeddings, student_ids = self.create_embeddings(features, sample_size)
        clusters = self.cluster_students(embeddings, n_clusters)
        descriptions = self.describe_clusters(features, clusters, student_ids)
        
        # Add cluster info to features
        id_to_cluster = dict(zip(student_ids, clusters))
        features['cluster'] = features['id_student'].map(id_to_cluster).fillna(-1).astype(int)
        
        print("\n" + "=" * 60)
        print("[Profiling] ✓ COMPLETE")
        print("=" * 60)
        
        return features, clusters, embeddings, descriptions
