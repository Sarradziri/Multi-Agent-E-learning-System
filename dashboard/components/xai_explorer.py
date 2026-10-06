"""
XAI Explorer Component
=======================
Interactive explainability visualizations with SHAP, LIME, and counterfactuals.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def _attr(obj, name, default=None):
    """Read an attribute from either an object or a dict (no eager default eval)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def render_xai_explorer(explanations, profile=None):
    """
    Render interactive XAI exploration panel.
    
    Args:
        explanations: CompleteExplanation object or dict
        profile: Student profile for context
    """
    
    # Confidence gauge
    confidence = getattr(explanations, 'confidence', 0.5)
    if isinstance(explanations, dict):
        confidence = explanations.get('confidence', 0.5)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2196F3"},
                'steps': [
                    {'range': [0, 40], 'color': "#FFCDD2"},
                    {'range': [40, 70], 'color': "#FFF9C4"},
                    {'range': [70, 100], 'color': "#C8E6C9"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        prediction = getattr(explanations, 'prediction', 'Unknown')
        if isinstance(explanations, dict):
            prediction = explanations.get('prediction', 'Unknown')
        
        st.metric("Prediction", prediction)
        
        # Prediction probability bar
        pred_proba = getattr(explanations, 'prediction_proba', confidence)
        if isinstance(explanations, dict):
            pred_proba = explanations.get('prediction_proba', confidence)
        
        st.progress(pred_proba, text=f"Probability: {pred_proba:.1%}")
    
    st.markdown("---")
    
    # Tabs for different XAI methods
    tab1, tab2, tab3, tab4 = st.tabs(["📊 SHAP", "🔍 LIME", "🔄 Counterfactual", "💬 Explanation"])
    
    with tab1:
        render_shap_tab(explanations)
    
    with tab2:
        render_lime_tab(explanations)
    
    with tab3:
        render_counterfactual_tab(explanations)
    
    with tab4:
        render_natural_explanation(explanations)


def render_shap_tab(explanations):
    """Render SHAP visualizations."""
    
    st.markdown("#### SHAP Feature Importance")
    st.caption("Shows how each feature contributes to the prediction")
    
    # Get SHAP explanations
    if hasattr(explanations, 'shap_explanations'):
        shap_exps = explanations.shap_explanations
    elif isinstance(explanations, dict):
        shap_exps = explanations.get('shap_explanations', [])
    else:
        shap_exps = []
    
    if not shap_exps:
        st.info("No SHAP explanations available.")
        return
    
    # Extract data
    features = []
    values = []
    
    for exp in shap_exps:
        if hasattr(exp, 'feature_name'):
            features.append(exp.feature_name)
            values.append(exp.shap_value)
        elif isinstance(exp, dict):
            features.append(exp.get('feature_name', 'Unknown'))
            values.append(exp.get('shap_value', 0))
    
    # Sort by absolute value
    sorted_indices = np.argsort(np.abs(values))[::-1]
    features = [features[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    # Create waterfall-style chart
    colors = ['#4CAF50' if v > 0 else '#F44336' for v in values]
    
    fig = go.Figure(go.Bar(
        y=features,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Feature Contributions to Prediction",
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis_title="Feature",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Add zero line
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive feature exploration
    st.markdown("#### 🎛️ What-If Analysis")
    st.caption("See how changing features affects the prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_feature = st.selectbox(
            "Select Feature",
            features,
            key="shap_feature"
        )
    
    with col2:
        # Get current value (simulated)
        current_idx = features.index(selected_feature)
        current_shap = values[current_idx]
        
        # Simulate feature slider
        multiplier = st.slider(
            f"Adjust {selected_feature}",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key="shap_slider"
        )
        
        new_impact = current_shap * multiplier
        delta = new_impact - current_shap
        
        st.metric(
            "Impact Change",
            f"{new_impact:+.4f}",
            delta=f"{delta:+.4f}",
            delta_color="normal" if delta > 0 else "inverse"
        )


def render_lime_tab(explanations):
    """Render LIME visualizations."""
    
    st.markdown("#### LIME Local Explanation")
    st.caption("Local interpretable model-agnostic explanations")
    
    # Get LIME explanations
    if hasattr(explanations, 'lime_explanations'):
        lime_exps = explanations.lime_explanations
    elif isinstance(explanations, dict):
        lime_exps = explanations.get('lime_explanations', [])
    else:
        lime_exps = []
    
    if not lime_exps:
        st.info("No LIME explanations available.")
        return
    
    # Extract data
    features = []
    weights = []
    
    for exp in lime_exps:
        if hasattr(exp, 'feature_name'):
            features.append(exp.feature_name)
            weights.append(exp.weight)
        elif isinstance(exp, dict):
            features.append(exp.get('feature_name', 'Unknown'))
            weights.append(exp.get('weight', 0))
    
    # Sort by absolute weight
    sorted_indices = np.argsort(np.abs(weights))[::-1]
    features = [features[i] for i in sorted_indices]
    weights = [weights[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    colors = ['#2196F3' if w > 0 else '#FF5722' for w in weights]
    
    fig = go.Figure(go.Bar(
        y=features,
        x=weights,
        orientation='h',
        marker_color=colors,
        text=[f"{w:+.4f}" for w in weights],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="LIME Feature Weights",
        xaxis_title="Weight (contribution to prediction)",
        yaxis_title="Feature",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # LIME score
    lime_score = getattr(explanations, 'lime_score', 0)
    if isinstance(explanations, dict):
        lime_score = explanations.get('lime_score', 0)
    
    st.metric("LIME Model R²", f"{lime_score:.3f}", help="How well the local linear model fits")


def render_counterfactual_tab(explanations):
    """Render counterfactual explanations."""
    
    st.markdown("#### Counterfactual Explanation")
    st.caption("What changes would lead to a different outcome?")
    
    # Get counterfactual
    if hasattr(explanations, 'counterfactual'):
        cf = explanations.counterfactual
    elif isinstance(explanations, dict):
        cf = explanations.get('counterfactual')
    else:
        cf = None
    
    if not cf:
        st.info("No counterfactual explanation available.")
        st.markdown("""
        **Why might this happen?**
        - The current prediction is already optimal
        - DiCE couldn't find a valid counterfactual within constraints
        - The feature space doesn't allow for the desired change
        """)
        return
    
    # Display counterfactual
    col1, col2 = st.columns(2)
    
    with col1:
        original = _attr(cf, 'original_outcome', 'Unknown')
        st.markdown("**Current Outcome:**")
        st.error(f"🔴 {original}")

    with col2:
        desired = _attr(cf, 'desired_outcome', 'Unknown')
        st.markdown("**Target Outcome:**")
        st.success(f"🟢 {desired}")
    
    st.markdown("---")
    st.markdown("**Changes Needed:**")
    
    # Get changes
    changes = _attr(cf, 'changes', [])
    
    if changes:
        for change in changes:
            if isinstance(change, dict):
                feature = change.get('feature', 'Unknown')
                original_val = change.get('original', 'N/A')
                new_val = change.get('counterfactual', 'N/A')
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{feature}**")
                
                with col2:
                    st.markdown(f"Current: `{original_val}`")
                
                with col3:
                    st.markdown(f"→ Change to: `{new_val}`")
    else:
        st.info("No specific changes identified.")
    
    # Feasibility score
    feasibility = _attr(cf, 'feasibility_score', 0)
    st.progress(feasibility, text=f"Feasibility: {feasibility:.1%}")


def render_natural_explanation(explanations):
    """Render natural language explanation."""
    
    st.markdown("#### 💬 Natural Language Explanation")
    
    # Get explanation text
    if hasattr(explanations, 'natural_language'):
        text = explanations.natural_language
    elif isinstance(explanations, dict):
        text = explanations.get('natural_language', 'No explanation available.')
    else:
        text = 'No explanation available.'
    
    # Display in a nice box (dark-theme friendly)
    st.markdown(f"""
    <div style="
        background: #171C28;
        border: 1px solid #2A3346;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #7B8CFF;
        margin: 1rem 0;
    ">
        <p style="font-size: 1.05rem; line-height: 1.6; margin: 0; color: #E6E9EF;">
            {text}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Regenerate button
    if st.button("🔄 Generate New Explanation", key="regen_explanation"):
        st.info("This would regenerate the explanation with the LLM. (Feature placeholder)")
