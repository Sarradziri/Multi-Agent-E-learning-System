"""
Agent Interaction Graph Component
==================================
Visualizes the flow of data between agents.
"""

import streamlit as st
import plotly.graph_objects as go


def render_agent_graph(agent_status: dict):
    """
    Render an interactive agent interaction graph.
    
    Args:
        agent_status: Dict with agent names as keys and status ('pending', 'running', 'complete') as values
    """
    
    # Define agent positions (x, y)
    agents = {
        'profiling': {'name': 'Profiling Agent', 'pos': (0, 1), 'icon': '👤', 'color': '#2196F3'},
        'recommendation': {'name': 'Recommendation Agent', 'pos': (1, 1), 'icon': '📚', 'color': '#4CAF50'},
        'xai': {'name': 'XAI Agent', 'pos': (2, 1), 'icon': '🔍', 'color': '#FF9800'},
        'path_planning': {'name': 'Path Planning Agent', 'pos': (1, 0), 'icon': '🛤️', 'color': '#9C27B0'},
        'content_gen': {'name': 'Content Generator', 'pos': (2, 0), 'icon': '📝', 'color': '#00BCD4'}
    }
    
    # Define connections (from, to, label)
    connections = [
        ('profiling', 'recommendation', 'Student Profile'),
        ('recommendation', 'xai', 'Recommendations'),
        ('profiling', 'path_planning', 'Learning Style'),
        ('path_planning', 'content_gen', 'Learning Path'),
        ('xai', 'content_gen', 'Explanations')
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Add edges (connections)
    for from_agent, to_agent, label in connections:
        from_pos = agents[from_agent]['pos']
        to_pos = agents[to_agent]['pos']
        
        # Determine edge color based on status
        from_status = agent_status.get(from_agent, 'pending')
        to_status = agent_status.get(to_agent, 'pending')
        
        if from_status == 'complete' and to_status in ['complete', 'running']:
            edge_color = '#2EC4A6'  # Teal - data transferred
            width = 3
        elif from_status == 'running':
            edge_color = '#FFB74D'  # Amber - in progress
            width = 2
        else:
            edge_color = '#3a4253'  # Slate - pending
            width = 1
        
        # Add edge
        fig.add_trace(go.Scatter(
            x=[from_pos[0], (from_pos[0] + to_pos[0]) / 2, to_pos[0]],
            y=[from_pos[1], (from_pos[1] + to_pos[1]) / 2 + 0.1, to_pos[1]],
            mode='lines',
            line=dict(color=edge_color, width=width),
            hoverinfo='text',
            hovertext=f"{label}",
            showlegend=False
        ))
        
        # Add arrow annotation
        fig.add_annotation(
            x=to_pos[0],
            y=to_pos[1],
            ax=(from_pos[0] + to_pos[0]) / 2,
            ay=(from_pos[1] + to_pos[1]) / 2 + 0.1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=edge_color
        )
    
    # Add nodes (agents)
    for key, agent in agents.items():
        status = agent_status.get(key, 'pending')
        
        # Determine node appearance based on status (dark-theme palette)
        if status == 'complete':
            node_color = agent['color']
            border_color = '#2EC4A6'
            symbol = 'circle'
        elif status == 'running':
            node_color = '#3a3320'
            border_color = '#FFB74D'
            symbol = 'circle'
        else:
            node_color = '#2A3346'
            border_color = '#5c6680'
            symbol = 'circle'

        fig.add_trace(go.Scatter(
            x=[agent['pos'][0]],
            y=[agent['pos'][1]],
            mode='markers+text',
            marker=dict(
                size=60,
                color=node_color,
                line=dict(color=border_color, width=3),
                symbol=symbol
            ),
            text=f"{agent['icon']}<br>{agent['name'].split()[0]}",
            textposition='middle center',
            textfont=dict(size=10, color='#FFFFFF'),
            hoverinfo='text',
            hovertext=f"{agent['name']}<br>Status: {status.upper()}",
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, 2.5]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, 1.5]
        ),
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Legend
    col1, col2, col3, col4, col5 = st.columns(5)
    
    for col, (key, agent) in zip([col1, col2, col3, col4, col5], agents.items()):
        status = agent_status.get(key, 'pending')
        status_icon = "✅" if status == 'complete' else "🔄" if status == 'running' else "⏳"
        col.markdown(f"{agent['icon']} **{agent['name'].split()[0]}** {status_icon}")


def render_sequence_diagram(agent_status: dict, messages: list = None):
    """
    Render a sequence diagram showing agent communication.
    
    Args:
        agent_status: Dict of agent statuses
        messages: List of message dicts with 'from', 'to', 'content', 'timestamp'
    """
    
    st.markdown("#### 📊 Agent Communication Sequence")
    
    if not messages:
        # Default messages based on pipeline
        messages = [
            {'from': 'User', 'to': 'Profiling', 'content': 'Student ID + Data', 'step': 1},
            {'from': 'Profiling', 'to': 'Recommendation', 'content': 'Student Profile', 'step': 2},
            {'from': 'Recommendation', 'to': 'XAI', 'content': 'Top Recommendations', 'step': 3},
            {'from': 'Profiling', 'to': 'Path Planning', 'content': 'Learning Style', 'step': 4},
            {'from': 'Path Planning', 'to': 'Content Gen', 'content': 'Learning Path', 'step': 5},
        ]
    
    # Create sequence visualization
    for msg in messages:
        step_status = "complete" if msg['step'] <= sum(1 for s in agent_status.values() if s == 'complete') else "pending"
        
        if step_status == 'complete':
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 0.5rem 0; opacity: 1;">
                <span style="background: #E3F2FD; padding: 0.3rem 0.6rem; border-radius: 5px; font-size: 0.8rem;">
                    {msg['from']}
                </span>
                <span style="margin: 0 0.5rem;">→</span>
                <span style="background: #E8F5E9; padding: 0.3rem 0.6rem; border-radius: 5px; font-size: 0.8rem;">
                    {msg['to']}
                </span>
                <span style="margin-left: 1rem; color: #666; font-size: 0.8rem;">
                    📨 {msg['content']}
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 0.5rem 0; opacity: 0.4;">
                <span style="background: #F5F5F5; padding: 0.3rem 0.6rem; border-radius: 5px; font-size: 0.8rem;">
                    {msg['from']}
                </span>
                <span style="margin: 0 0.5rem;">→</span>
                <span style="background: #F5F5F5; padding: 0.3rem 0.6rem; border-radius: 5px; font-size: 0.8rem;">
                    {msg['to']}
                </span>
                <span style="margin-left: 1rem; color: #999; font-size: 0.8rem;">
                    ⏳ {msg['content']}
                </span>
            </div>
            """, unsafe_allow_html=True)
