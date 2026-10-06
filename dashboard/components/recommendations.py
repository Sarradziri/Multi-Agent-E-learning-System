"""
Recommendations Component
==========================
Display recommendations with metrics and explanations.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


def render_recommendations(recommendations, metrics: dict):
    """
    Render the recommendations panel with cards and metrics.
    
    Args:
        recommendations: RecommendationSet object or dict
        metrics: Dict of evaluation metrics
    """
    
    # Metrics summary
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mrr = metrics.get('mrr', 0)
            st.metric(
                "MRR",
                f"{mrr:.3f}",
                help="Mean Reciprocal Rank - higher is better"
            )
        
        with col2:
            ndcg = metrics.get('ndcg@5', 0)
            st.metric(
                "NDCG@5",
                f"{ndcg:.3f}",
                help="Normalized Discounted Cumulative Gain at 5"
            )
        
        with col3:
            precision = metrics.get('precision@5', 0)
            st.metric(
                "Precision@5",
                f"{precision:.1%}",
                help="Precision at top 5 recommendations"
            )
        
        with col4:
            recall = metrics.get('recall@10', 0)
            st.metric(
                "Recall@10",
                f"{recall:.1%}",
                help="Recall at top 10 recommendations"
            )
    
    st.markdown("---")
    
    # Get recommendations list
    if hasattr(recommendations, 'recommendations'):
        rec_list = recommendations.recommendations
    elif isinstance(recommendations, dict) and 'recommendations' in recommendations:
        rec_list = recommendations['recommendations']
    else:
        rec_list = recommendations if isinstance(recommendations, list) else []
    
    if not rec_list:
        st.info("No recommendations generated yet.")
        return
    
    # Recommendation cards
    for i, rec in enumerate(rec_list[:10]):
        # Extract attributes
        if hasattr(rec, 'item_name'):
            item_name = rec.item_name
            score = rec.score
            source = rec.source
            reasoning = getattr(rec, 'reasoning', '')
            is_hit = getattr(rec, 'is_hit', False)
        elif isinstance(rec, dict):
            item_name = rec.get('item_name', rec.get('item_id', 'Unknown'))
            score = rec.get('score', 0)
            source = rec.get('source', 'hybrid')
            reasoning = rec.get('reasoning', '')
            is_hit = rec.get('is_hit', False)
        else:
            continue
        
        # Determine item type from name
        item_type = item_name.split('_')[0] if '_' in item_name else 'unknown'
        type_icons = {
            'oucontent': '📄',
            'quiz': '❓',
            'forumng': '💬',
            'resource': '📁',
            'url': '🔗',
            'page': '📃',
            'subpage': '📑',
            'homepage': '🏠'
        }
        icon = type_icons.get(item_type, '📦')
        
        # Source badge color
        source_colors = {
            'collaborative': '#2196F3',
            'content': '#4CAF50',
            'hybrid': '#9C27B0',
            'default': '#757575'
        }
        source_color = source_colors.get(source, '#757575')
        
        # Create card
        with st.container():
            col1, col2, col3 = st.columns([0.5, 3, 1])
            
            with col1:
                st.markdown(f"### {icon}")
                st.caption(f"#{i+1}")
            
            with col2:
                # Title with hit badge
                title_html = f"**{item_name}**"
                if is_hit:
                    title_html += ' <span style="background-color: #4CAF50; color: white; padding: 0.1rem 0.4rem; border-radius: 10px; font-size: 0.7rem;">✓ HIT</span>'
                st.markdown(title_html, unsafe_allow_html=True)
                
                # Source badge
                st.markdown(f"""
                <span style="background-color: {source_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 10px; font-size: 0.75rem;">
                    {source}
                </span>
                """, unsafe_allow_html=True)
                
                # Reasoning (expandable)
                if reasoning:
                    with st.expander("Why this recommendation?"):
                        st.write(reasoning)
            
            with col3:
                # Score as progress
                st.markdown(f"**{score:.2f}**")
                st.progress(min(score, 1.0))
            
            st.markdown("---")
    
    # Score distribution chart
    with st.expander("📊 Score Distribution"):
        scores = []
        names = []
        sources = []
        
        for rec in rec_list[:10]:
            if hasattr(rec, 'score'):
                scores.append(rec.score)
                names.append(getattr(rec, 'item_name', 'Item')[:20])
                sources.append(getattr(rec, 'source', 'hybrid'))
            elif isinstance(rec, dict):
                scores.append(rec.get('score', 0))
                names.append(rec.get('item_name', 'Item')[:20])
                sources.append(rec.get('source', 'hybrid'))
        
        if scores:
            fig = go.Figure(data=[
                go.Bar(
                    x=names,
                    y=scores,
                    marker_color=['#2196F3' if s == 'collaborative' else '#4CAF50' if s == 'content' else '#9C27B0' for s in sources],
                    text=[f"{s:.2f}" for s in scores],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Recommendation Scores",
                xaxis_title="Item",
                yaxis_title="Score",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)


def render_recommendation_comparison(rec1, rec2, label1="Student A", label2="Student B"):
    """Compare recommendations between two students."""
    
    st.markdown(f"### Comparison: {label1} vs {label2}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{label1}**")
        for rec in rec1[:5]:
            name = getattr(rec, 'item_name', rec.get('item_name', 'Unknown'))
            score = getattr(rec, 'score', rec.get('score', 0))
            st.markdown(f"- {name} ({score:.2f})")
    
    with col2:
        st.markdown(f"**{label2}**")
        for rec in rec2[:5]:
            name = getattr(rec, 'item_name', rec.get('item_name', 'Unknown'))
            score = getattr(rec, 'score', rec.get('score', 0))
            st.markdown(f"- {name} ({score:.2f})")
