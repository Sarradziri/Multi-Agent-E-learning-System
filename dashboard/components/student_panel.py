"""
Student Panel Component
========================
Editable student profile panel for the sidebar.
"""

import streamlit as st
import pandas as pd
import numpy as np


def render_student_panel(data: dict) -> tuple:
    """
    Render the student panel with selection and custom profile options.
    
    Returns:
        tuple: (student_id, custom_profile, use_custom)
    """
    st.markdown("## 👤 Student Profile")
    
    # Get students with enough interactions
    if 'student_vle' in data and len(data['student_vle']) > 0:
        interaction_counts = data['student_vle'].groupby('id_student').size()
        active_students = interaction_counts[interaction_counts >= 50].index.tolist()
    else:
        active_students = []
    
    # Mode selection
    mode = st.radio(
        "Profile Mode",
        ["Select Existing", "Custom Profile"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    use_custom = mode == "Custom Profile"
    
    if not use_custom:
        # Existing student selection
        if active_students:
            student_id = st.selectbox(
                "Select Student",
                options=active_students[:100],  # Limit for performance
                index=0,
                help="Students with 50+ interactions"
            )
            
            # Show student info
            if student_id and 'student_info' in data:
                student_data = data['student_info'][data['student_info']['id_student'] == student_id]
                if len(student_data) > 0:
                    row = student_data.iloc[0]
                    
                    st.markdown("---")
                    
                    # Module info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Module", row.get('code_module', 'N/A'))
                    with col2:
                        st.metric("Result", row.get('final_result', 'N/A'))
                    
                    # Get interaction stats
                    if 'student_vle' in data:
                        vle_data = data['student_vle'][data['student_vle']['id_student'] == student_id]
                        total_clicks = vle_data['sum_click'].sum() if 'sum_click' in vle_data.columns else len(vle_data)
                        num_sessions = len(vle_data)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Clicks", f"{total_clicks:,}")
                        with col2:
                            st.metric("Sessions", f"{num_sessions:,}")
                    
                    # Get assessment stats
                    if 'student_assessment' in data and len(data['student_assessment']) > 0:
                        assess_data = data['student_assessment'][data['student_assessment']['id_student'] == student_id]
                        if len(assess_data) > 0:
                            avg_score = assess_data['score'].mean()
                            st.metric("Avg Score", f"{avg_score:.1f}%")
            
            custom_profile = None
        else:
            st.warning("No students with sufficient data found")
            student_id = 1
            custom_profile = None
    
    else:
        # Custom profile builder
        st.markdown("### Build Custom Profile")
        
        student_id = st.number_input("Student ID", value=99999, min_value=1)
        
        st.markdown("#### Learning Metrics")
        
        total_clicks = st.slider(
            "Total Clicks",
            min_value=50,
            max_value=5000,
            value=500,
            step=50,
            help="Total interactions with learning materials"
        )
        
        avg_score = st.slider(
            "Average Score (%)",
            min_value=0,
            max_value=100,
            value=65,
            step=5,
            help="Average assessment score"
        )
        
        num_sessions = st.slider(
            "Number of Sessions",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Total learning sessions"
        )
        
        st.markdown("#### Profile Attributes")
        
        learning_style = st.selectbox(
            "Learning Style",
            ["High Engagement Achiever", "Steady Learner", "At-Risk Student", "Visual Learner", "Interactive Learner"],
            index=0
        )
        
        engagement_level = st.selectbox(
            "Engagement Level",
            ["High", "Medium", "Low"],
            index=1
        )
        
        module = st.selectbox(
            "Module",
            ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"],
            index=0
        )
        
        custom_profile = {
            'total_clicks': total_clicks,
            'avg_score': avg_score,
            'num_sessions': num_sessions,
            'learning_style': learning_style,
            'engagement_level': engagement_level,
            'module': module,
            'final_result': 'Pass' if avg_score >= 50 else 'Fail'
        }
        
        # Show profile summary
        st.markdown("---")
        st.markdown("#### Profile Summary")
        
        # Visual representation
        st.progress(total_clicks / 5000, text=f"Clicks: {total_clicks}")
        st.progress(avg_score / 100, text=f"Score: {avg_score}%")
        st.progress(num_sessions / 1000, text=f"Sessions: {num_sessions}")
    
    # Compare mode
    st.markdown("---")
    if st.checkbox("🔄 Compare with another student"):
        compare_id = st.selectbox(
            "Compare with",
            options=[s for s in active_students[:50] if s != student_id],
            key="compare_student"
        )
        st.session_state['compare_student'] = compare_id
    
    return student_id, custom_profile, use_custom
