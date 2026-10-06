"""
Learning Path Component
========================
Visualize the personalized learning path as a timeline.
"""

import streamlit as st
import plotly.graph_objects as go


def render_learning_path(learning_path):
    """
    Render the learning path visualization.
    
    Args:
        learning_path: LearningPath object or dict
    """
    
    # Extract attributes
    if hasattr(learning_path, 'steps'):
        steps = learning_path.steps
        total_weeks = learning_path.total_weeks
        total_hours = learning_path.total_hours
        algorithm = learning_path.algorithm_used
        path_score = learning_path.path_score
        q_value = learning_path.q_value
    elif isinstance(learning_path, dict):
        steps = learning_path.get('steps', [])
        total_weeks = learning_path.get('total_weeks', 0)
        total_hours = learning_path.get('total_hours', 0)
        algorithm = learning_path.get('algorithm_used', 'unknown')
        path_score = learning_path.get('path_score', 0)
        q_value = learning_path.get('q_value', 0)
    else:
        st.info("No learning path available.")
        return
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Weeks", total_weeks)
    
    with col2:
        st.metric("Total Hours", f"{total_hours}h")
    
    with col3:
        st.metric("Algorithm", algorithm.upper())
    
    with col4:
        st.metric("Path Score", f"{path_score:.1f}")
    
    st.markdown("---")
    
    # Timeline visualization
    st.markdown("#### 📅 Learning Timeline")
    
    if not steps:
        st.info("No steps in learning path.")
        return
    
    # Create timeline figure
    fig = go.Figure()
    
    # Extract step data
    weeks = []
    names = []
    hours = []
    
    for step in steps:
        if hasattr(step, 'week'):
            weeks.append(step.week)
            names.append(step.node_name)
            hours.append(step.estimated_hours)
        elif isinstance(step, dict):
            weeks.append(step.get('week', 0))
            names.append(step.get('node_name', step.get('topic', 'Unknown')))
            hours.append(step.get('estimated_hours', step.get('hours', 0)))
    
    # Colors for steps
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4', '#E91E63', '#3F51B5']
    
    # Add timeline markers
    for i, (week, name, hour) in enumerate(zip(weeks, names, hours)):
        color = colors[i % len(colors)]
        
        # Add marker
        fig.add_trace(go.Scatter(
            x=[week],
            y=[0],
            mode='markers+text',
            marker=dict(size=30, color=color, symbol='circle'),
            text=[f"W{week}"],
            textposition='middle center',
            textfont=dict(color='white', size=10),
            hoverinfo='text',
            hovertext=f"Week {week}: {name}<br>{hour}h",
            showlegend=False
        ))
        
        # Add label above
        fig.add_annotation(
            x=week,
            y=0.3,
            text=f"<b>{name}</b><br>{hour}h",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=color,
            ax=0,
            ay=-40,
            font=dict(size=10, color='#E6E9EF'),
            align='center'
        )
    
    # Add connecting line
    fig.add_trace(go.Scatter(
        x=weeks,
        y=[0] * len(weeks),
        mode='lines',
        line=dict(color='#E0E0E0', width=3, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Update layout
    fig.update_layout(
        height=250,
        xaxis=dict(
            title="Week",
            showgrid=True,
            gridcolor='#F0F0F0',
            range=[min(weeks) - 1, max(weeks) + 2]
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            range=[-0.5, 1]
        ),
        margin=dict(l=20, r=20, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed step list
    with st.expander("📋 Detailed Course List"):
        for i, step in enumerate(steps):
            if hasattr(step, 'week'):
                week = step.week
                name = step.node_name
                hours_val = step.estimated_hours
                activities = getattr(step, 'activities', [])
                rationale = getattr(step, 'rationale', '')
            elif isinstance(step, dict):
                week = step.get('week', 0)
                name = step.get('node_name', step.get('topic', 'Unknown'))
                hours_val = step.get('estimated_hours', step.get('hours', 0))
                activities = step.get('activities', [])
                rationale = step.get('rationale', '')
            else:
                continue
            
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.markdown(f"**Week {week}**")
            
            with col2:
                st.markdown(f"📚 {name}")
                if rationale:
                    st.caption(rationale)
            
            with col3:
                st.markdown(f"⏱️ {hours_val}h")
            
            if activities:
                st.caption(f"Activities: {', '.join(activities[:3])}")
            
            st.markdown("---")
    
    # Q-Learning info
    if algorithm == 'qlearning':
        with st.expander("🤖 Q-Learning Details"):
            st.markdown(f"""
            **Q-Value:** {q_value:.2f}
            
            The Q-Learning agent explored multiple learning paths and selected this one
            based on:
            - Expected completion probability
            - Time efficiency
            - Prerequisite satisfaction
            - Student skill level alignment
            """)
            
            # Simple Q-value visualization
            st.progress(min(q_value / 500, 1.0), text=f"Q-Value: {q_value:.1f} / 500")


def render_path_comparison(path1, path2, label1="Path A", label2="Path B"):
    """Compare two learning paths side by side."""
    
    st.markdown(f"### Path Comparison: {label1} vs {label2}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{label1}**")
        if hasattr(path1, 'steps'):
            for step in path1.steps[:5]:
                st.markdown(f"- Week {step.week}: {step.node_name}")
        st.metric("Total Hours", getattr(path1, 'total_hours', 0))
    
    with col2:
        st.markdown(f"**{label2}**")
        if hasattr(path2, 'steps'):
            for step in path2.steps[:5]:
                st.markdown(f"- Week {step.week}: {step.node_name}")
        st.metric("Total Hours", getattr(path2, 'total_hours', 0))
