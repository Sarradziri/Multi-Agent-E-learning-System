"""
Quiz Player Component
======================
Interactive quiz with scoring and feedback.
"""

import streamlit as st


def render_quiz_player(quizzes: list):
    """
    Render an interactive quiz player.
    
    Args:
        quizzes: List of GeneratedQuiz objects or dicts
    """
    
    if not quizzes:
        st.info("No quizzes available. Run the pipeline to generate quizzes.")
        return
    
    # Initialize quiz state
    if 'quiz_state' not in st.session_state:
        st.session_state.quiz_state = {
            'current_quiz': 0,
            'current_question': 0,
            'answers': {},  # Format: {'quiz_0_q0': 'A', 'quiz_0_q1': 'B'}
            'submitted': {},  # Format: {'quiz_0_q0': True, 'quiz_0_q1': False}
            'score': 0,
            'total_answered': 0
        }
    
    state = st.session_state.quiz_state
    
    # Quiz selector
    quiz_topics = []
    for q in quizzes:
        if hasattr(q, 'topic'):
            quiz_topics.append(q.topic)
        elif isinstance(q, dict):
            quiz_topics.append(q.get('topic', 'Unknown Topic'))
        else:
            quiz_topics.append('Unknown Topic')
    
    selected_quiz_idx = st.selectbox(
        "Select Quiz",
        range(len(quiz_topics)),
        format_func=lambda x: f"📝 {quiz_topics[x]}",
        key="quiz_selector"
    )
    
    # Get selected quiz
    current_quiz = quizzes[selected_quiz_idx]
    
    if hasattr(current_quiz, 'questions'):
        questions = current_quiz.questions
        topic = current_quiz.topic
        difficulty = getattr(current_quiz, 'difficulty', 'medium')
    elif isinstance(current_quiz, dict):
        questions = current_quiz.get('questions', [])
        topic = current_quiz.get('topic', 'Unknown')
        difficulty = current_quiz.get('difficulty', 'medium')
    else:
        questions = []
        topic = "Unknown"
        difficulty = "medium"
    
    if not questions:
        st.warning("No questions in this quiz.")
        return
    
    # Quiz header
    st.markdown(f"### 📚 {topic}")
    
    # Difficulty badge
    diff_colors = {'easy': '#4CAF50', 'medium': '#FF9800', 'hard': '#F44336'}
    diff_color = diff_colors.get(difficulty.lower(), '#9E9E9E')
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <span style="background-color: {diff_color}; color: white; padding: 0.2rem 0.6rem; border-radius: 10px; font-size: 0.8rem;">
            {difficulty.upper()}
        </span>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"**Questions:** {len(questions)}")
    
    with col3:
        # Calculate score for this quiz
        quiz_key = f"quiz_{selected_quiz_idx}"
        answered, correct = calculate_quiz_score(quiz_key, questions, state)
        
        if answered > 0:
            st.markdown(f"**Score:** {correct}/{answered}")
    
    st.markdown("---")
    
    # Mode selector
    mode = st.radio(
        "Mode",
        ["Take Quiz", "Review All"],
        horizontal=True,
        key=f"mode_{selected_quiz_idx}"
    )
    
    if mode == "Take Quiz":
        render_quiz_interactive(questions, selected_quiz_idx, state)
    else:
        render_quiz_review(questions, selected_quiz_idx, state)


def calculate_quiz_score(quiz_key: str, questions: list, state: dict) -> tuple:
    """Calculate score for a specific quiz."""
    answered = 0
    correct = 0
    
    for i in range(len(questions)):
        question_key = f"{quiz_key}_q{i}"
        if question_key in state.get('submitted', {}):
            answered += 1
            if state['submitted'][question_key]:  # True if correct
                correct += 1
    
    return answered, correct


def extract_question_data(question):
    """Extract data from question object/dict."""
    if isinstance(question, dict):
        return {
            'text': question.get('question', 'No question text'),
            'options': question.get('options', []),
            'correct': question.get('correct', 'A'),
            'explanation': question.get('explanation', '')
        }
    else:
        return {
            'text': getattr(question, 'question', 'No question text'),
            'options': getattr(question, 'options', []),
            'correct': getattr(question, 'correct', 'A'),
            'explanation': getattr(question, 'explanation', '')
        }


def render_quiz_interactive(questions: list, quiz_idx: int, state: dict):
    """Render interactive quiz-taking mode."""
    
    quiz_key = f"quiz_{quiz_idx}"
    
    # Question navigation
    total_q = len(questions)
    
    # Progress bar
    answered, _ = calculate_quiz_score(quiz_key, questions, state)
    progress = answered / total_q if total_q > 0 else 0
    st.progress(progress, text=f"Progress: {answered}/{total_q}")
    
    # Question selector
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("⬅️ Prev", key=f"prev_{quiz_key}", disabled=state.get('current_question', 0) == 0):
            state['current_question'] = max(0, state.get('current_question', 0) - 1)
            st.rerun()
    
    with col2:
        q_idx = st.selectbox(
            "Jump to Question",
            range(total_q),
            index=state.get('current_question', 0),
            format_func=lambda x: f"Question {x + 1}",
            key=f"q_selector_{quiz_key}",
            label_visibility="collapsed"
        )
        if q_idx != state.get('current_question', 0):
            state['current_question'] = q_idx
            st.rerun()
    
    with col3:
        if st.button("Next ➡️", key=f"next_{quiz_key}", disabled=state.get('current_question', 0) >= total_q - 1):
            state['current_question'] = min(total_q - 1, state.get('current_question', 0) + 1)
            st.rerun()
    
    st.markdown("---")
    
    # Current question
    current_q_idx = state.get('current_question', 0)
    question_data = extract_question_data(questions[current_q_idx])
    
    # Question text
    st.markdown(f"### Q{current_q_idx + 1}: {question_data['text']}")
    
    # Check if already submitted
    submission_key = f"{quiz_key}_q{current_q_idx}"
    already_submitted = submission_key in state.get('submitted', {})
    
    # Answer options
    answer_key = f"{quiz_key}_answer_{current_q_idx}"
    
    if not already_submitted:
        # Show radio buttons for selection
        selected = st.radio(
            "Select your answer:",
            question_data['options'],
            key=answer_key,
            label_visibility="collapsed"
        )
        
        # Store the selected answer
        if 'answers' not in state:
            state['answers'] = {}
        state['answers'][submission_key] = selected
        
        # Submit button
        if st.button("✅ Submit Answer", key=f"submit_{submission_key}", type="primary"):
            # Check if correct
            selected_letter = selected[0] if selected else ''
            is_correct = selected_letter == question_data['correct']
            
            # Store submission
            if 'submitted' not in state:
                state['submitted'] = {}
            state['submitted'][submission_key] = is_correct
            
            # Update score
            state['total_answered'] = sum(1 for key in state['submitted'] if key.startswith('quiz_'))
            state['score'] = sum(1 for key, value in state['submitted'].items() if value)
            
            st.rerun()
    
    else:
        # Show results
        is_correct = state['submitted'][submission_key]
        selected_answer = state['answers'].get(submission_key, '')
        
        for opt in question_data['options']:
            opt_letter = opt[0] if opt else ''
            
            if opt_letter == question_data['correct']:
                # Correct answer
                st.markdown(f"""
                <div style="padding: 0.8rem; border-radius: 8px; border: 2px solid #2EC4A6; background-color: #14241d; color: #E6E9EF; margin: 0.3rem 0;">
                    ✅ {opt}
                </div>
                """, unsafe_allow_html=True)
            elif opt == selected_answer and not is_correct:
                # Wrong selection
                st.markdown(f"""
                <div style="padding: 0.8rem; border-radius: 8px; border: 2px solid #F2555A; background-color: #2a1618; color: #E6E9EF; margin: 0.3rem 0;">
                    ❌ {opt}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Other options
                st.markdown(f"""
                <div style="padding: 0.8rem; border-radius: 8px; border: 1px solid #2A3346; background-color: #171C28; color: #cfd6e6; margin: 0.3rem 0;">
                    {opt}
                </div>
                """, unsafe_allow_html=True)
        
        # Result message
        if is_correct:
            st.success("🎉 Correct!")
        else:
            st.error(f"❌ Incorrect. The correct answer was {question_data['correct']}.")
        
        # Explanation
        if question_data['explanation']:
            st.info(f"💡 **Explanation:** {question_data['explanation']}")
        
        # Try again button
        if st.button("🔄 Try Again", key=f"retry_{submission_key}"):
            # Remove this question from submitted and answers
            if submission_key in state['submitted']:
                del state['submitted'][submission_key]
            if submission_key in state.get('answers', {}):
                del state['answers'][submission_key]
            st.rerun()


def render_quiz_review(questions: list, quiz_idx: int, state: dict):
    """Render quiz review mode showing all questions and answers."""
    
    quiz_key = f"quiz_{quiz_idx}"
    
    st.markdown("### 📖 All Questions & Answers")
    
    for i, question in enumerate(questions):
        question_data = extract_question_data(question)
        submission_key = f"{quiz_key}_q{i}"
        is_submitted = submission_key in state.get('submitted', {})
        user_answer = state.get('answers', {}).get(submission_key, '')
        
        # Create status indicator
        if is_submitted:
            is_correct = state['submitted'][submission_key]
            status_icon = "✅" if is_correct else "❌"
            status_color = "#4CAF50" if is_correct else "#F44336"
        else:
            status_icon = "⏳"
            status_color = "#9E9E9E"
        
        with st.expander(
            f"{status_icon} Q{i+1}: {question_data['text'][:50]}...", 
            expanded=i == 0
        ):
            st.markdown(f"**{question_data['text']}**")
            
            # Display options with indicators
            for opt in question_data['options']:
                opt_letter = opt[0] if opt else ''
                
                if opt_letter == question_data['correct']:
                    st.markdown(f"✅ **{opt}** (Correct Answer)")
                elif opt == user_answer and not (opt_letter == question_data['correct']):
                    st.markdown(f"❌ **{opt}** (Your Answer)")
                else:
                    st.markdown(f"○ {opt}")
            
            # Explanation
            if question_data['explanation']:
                st.info(f"💡 **Explanation:** {question_data['explanation']}")
            
            # User performance info
            if is_submitted:
                if state['submitted'][submission_key]:
                    st.success("You answered correctly!")
                else:
                    st.error(f"You selected: {user_answer}")
    
    # Reset quiz button
    if st.button("🔄 Reset Quiz Progress", key=f"reset_{quiz_key}"):
        # Clear all answers for this quiz
        keys_to_remove = [k for k in state.get('submitted', {}).keys() if k.startswith(quiz_key)]
        for k in keys_to_remove:
            del state['submitted'][k]
            if k in state.get('answers', {}):
                del state['answers'][k]
        
        # Update totals
        state['total_answered'] = sum(1 for key in state.get('submitted', {}).keys() if key.startswith('quiz_'))
        state['score'] = sum(1 for key, value in state.get('submitted', {}).items() if value)
        
        st.success("Quiz progress reset!")
        st.rerun()


def render_quiz_summary(quizzes: list, state: dict):
    """Render overall quiz performance summary."""
    
    if not quizzes:
        return
    
    st.markdown("### 📊 Quiz Performance Summary")
    
    total_correct = 0
    total_answered = 0
    
    for idx, quiz in enumerate(quizzes):
        quiz_key = f"quiz_{idx}"
        
        if hasattr(quiz, 'questions'):
            questions = quiz.questions
            topic = quiz.topic
        elif isinstance(quiz, dict):
            questions = quiz.get('questions', [])
            topic = quiz.get('topic', 'Unknown')
        else:
            continue
        
        answered, correct = calculate_quiz_score(quiz_key, questions, state)
        
        total_correct += correct
        total_answered += answered
        
        if answered > 0:
            percentage = (correct / answered) * 100 if answered > 0 else 0
            st.markdown(f"**{topic}:** {correct}/{answered} ({percentage:.0f}%)")
            st.progress(correct / len(questions) if questions else 0)
    
    if total_answered > 0:
        st.markdown("---")
        overall = (total_correct / total_answered) * 100 if total_answered > 0 else 0
        st.metric("Overall Score", f"{total_correct}/{total_answered} ({overall:.0f}%)")