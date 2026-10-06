"""
🎓 E-Learning Multi-Agent System Dashboard
============================================
Interactive dashboard for the MAS recommendation system.

Run from the project root:  streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Agents log decorative Unicode (✓, ⚠) to the console. On Windows the default
# cp1252 stdout can't encode those and raises UnicodeEncodeError mid-init, which
# would silently kill an agent. Force UTF-8 so the pipeline runs everywhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Credentials: locally the agents read a .env file; on Streamlit Community Cloud
# they come from st.secrets. Mirror any secrets into the environment so the
# agents (which use os.getenv) find them in both places.
try:
    for _k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
               "AZURE_OPENAI_CHAT_DEPLOYMENT", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
               "AZURE_OPENAI_API_VERSION", "OPENAI_API_KEY"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets.toml present (normal for local runs)

# Page config
st.set_page_config(
    page_title="E-Learning MAS Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import components
from components.student_panel import render_student_panel
from components.agent_graph import render_agent_graph
from components.recommendations import render_recommendations
from components.xai_explorer import render_xai_explorer
from components.learning_path import render_learning_path
from components.quiz_player import render_quiz_player
from components.blackboard import build_comm_log, render_agent_timeline, render_blackboard
from utils.pipeline_runner import PipelineRunner

# ---- Agent registry (single source of truth for names / icons) ----------
AGENTS = [
    ("profiling", "Profiling", "👤"),
    ("recommendation", "Recommendation", "📚"),
    ("xai", "Explainability", "🔍"),
    ("path_planning", "Path Planning", "🛤️"),
    ("content_gen", "Content Gen", "📝"),
]

# Custom CSS — cohesive dark accent (pairs with .streamlit/config.toml)
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #7B8CFF 0%, #2EC4A6 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { text-align: center; color: #9aa4b8; margin-bottom: 1.5rem; }
    .chip {
        display:inline-block; background:#1B2130; border:1px solid #2A3346;
        color:#cfd6e6; padding:0.35rem 0.8rem; border-radius:20px;
        font-size:0.85rem; margin:0.2rem;
    }
    .hero {
        background:#171C28; border:1px solid #2A3346; border-radius:14px;
        padding:2rem; margin:0.5rem 0 1.5rem 0;
    }
    div[data-testid="stMetric"] {
        background:#171C28; border:1px solid #2A3346;
        border-radius:10px; padding:0.6rem 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = None
    if 'pipeline_running' not in st.session_state:
        st.session_state.pipeline_running = False
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = {k: 'pending' for k, _, _ in AGENTS}
    if 'selected_student' not in st.session_state:
        st.session_state.selected_student = None
    if 'custom_profile' not in st.session_state:
        st.session_state.custom_profile = None
    if 'quiz_state' not in st.session_state:
        st.session_state.quiz_state = {
            'current_question': 0, 'answers': {}, 'submitted': {},
            'score': 0, 'total_answered': 0
        }
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'data' not in st.session_state:
        st.session_state.data = None


@st.cache_data(show_spinner=False)
def create_synthetic_data(n_students=400, n_items=80, seed=42):
    """Generate small synthetic OULAD-shaped data so the demo runs without the
    443 MB dataset (e.g. on Streamlit Community Cloud). Includes `code_module`
    on student_vle so the recommendation agent's module filter works."""
    rng = np.random.default_rng(seed)
    modules = ['AAA', 'BBB', 'CCC']
    student_modules = rng.choice(modules, n_students)

    student_info = pd.DataFrame({
        'id_student': range(1, n_students + 1),
        'code_module': student_modules,
        'code_presentation': rng.choice(['2013J', '2014B'], n_students),
        'final_result': rng.choice(['Pass', 'Fail', 'Withdrawn', 'Distinction'],
                                   n_students, p=[0.4, 0.2, 0.2, 0.2]),
    })

    vle = pd.DataFrame({
        'id_site': range(1, n_items + 1),
        'activity_type': rng.choice(
            ['oucontent', 'quiz', 'forumng', 'resource', 'url', 'page'], n_items),
        'code_module': rng.choice(modules, n_items),
    })

    rows = []
    for sid, mod in zip(student_info['id_student'], student_modules):
        pool = vle[vle['code_module'] == mod]['id_site'].tolist() or vle['id_site'].tolist()
        for _ in range(int(rng.integers(60, 200))):
            rows.append((sid, int(rng.choice(pool)), int(rng.integers(1, 40)), mod))
    student_vle = pd.DataFrame(rows, columns=['id_student', 'id_site', 'sum_click', 'code_module'])

    assess = []
    for sid in student_info['id_student']:
        for _ in range(int(rng.integers(3, 10))):
            assess.append((sid, int(rng.integers(1, 50)), int(rng.integers(20, 100))))
    student_assessment = pd.DataFrame(assess, columns=['id_student', 'id_assessment', 'score'])

    return {'student_info': student_info, 'student_vle': student_vle,
            'student_assessment': student_assessment, 'vle': vle}


def load_data():
    """Load OULAD dataset, falling back to synthetic data when CSVs are absent."""
    if st.session_state.data_loaded:
        return st.session_state.data

    data_path = "data/oulad"
    files = {
        'student_info': 'studentInfo.csv',
        'student_vle': 'studentVle.csv',
        'student_assessment': 'studentAssessment.csv',
        'vle': 'vle.csv'
    }
    data = {}
    for key, filename in files.items():
        filepath = os.path.join(data_path, filename)
        data[key] = pd.read_csv(filepath) if os.path.exists(filepath) else pd.DataFrame()

    # Fall back to synthetic data if the real dataset isn't present.
    if data['student_info'] is None or len(data['student_info']) == 0:
        data = create_synthetic_data()
        st.session_state.synthetic = True
    else:
        st.session_state.synthetic = False

    st.session_state.data = data
    st.session_state.data_loaded = True
    return data


def render_sidebar_status():
    """Compact per-agent status indicators in the sidebar."""
    st.markdown("### Agent Status")
    cols = st.columns(len(AGENTS))
    for col, (key, label, icon) in zip(cols, AGENTS):
        status = st.session_state.agent_status[key]
        mark = "✅" if status == 'complete' else "🔄" if status == 'running' else "⏳"
        col.markdown(f"<div style='text-align:center'>{icon}<br>{mark}</div>", unsafe_allow_html=True)
        col.caption(label.split()[0])


def main():
    """Main application."""
    init_session_state()

    st.markdown('<p class="main-header">🎓 E-Learning Multi-Agent System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Six agents collaborate over a shared blackboard to recommend, '
                'explain, and plan personalized learning — on the OULAD dataset.</p>', unsafe_allow_html=True)

    with st.spinner("Loading dataset..."):
        data = load_data()

    has_data = data.get('student_info') is not None and len(data.get('student_info', [])) > 0
    if has_data and st.session_state.get('synthetic'):
        st.info("🧪 Running on **synthetic demo data** (the 443 MB OULAD dataset isn't bundled). "
                "Metrics are illustrative; drop the real CSVs in `data/oulad/` for genuine results.")

    # ---- Sidebar: student selection + run control ----
    with st.sidebar:
        if has_data:
            student_id, custom_profile, use_custom = render_student_panel(data)
        else:
            st.warning("OULAD dataset not found in `data/oulad/`.")
            st.caption("Download: https://analyse.kmi.open.ac.uk/open_dataset")
            student_id, custom_profile, use_custom = 1, None, False

        st.session_state.selected_student = student_id
        st.session_state.custom_profile = custom_profile if use_custom else None

        st.divider()
        if st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True,
                     disabled=not has_data):
            st.session_state.pipeline_running = True
            st.session_state.agent_status = {k: 'pending' for k, _, _ in AGENTS}

        st.divider()
        render_sidebar_status()

    # ---- Tabs ----
    tab_overview, tab_run, tab_agents, tab_recs, tab_xai, tab_path = st.tabs([
        "🏠 Overview", "▶️ Live Run", "🔗 Agents & Blackboard",
        "📚 Recommendations", "🔍 Explainability", "🛤️ Path & Quiz"
    ])

    with tab_overview:
        render_overview(has_data)

    with tab_run:
        if not has_data:
            st.info("Add the OULAD CSVs to `data/oulad/` to run the pipeline.")
        else:
            run_tab(data, student_id, custom_profile if use_custom else None)

    results = st.session_state.pipeline_results

    with tab_agents:
        if results:
            st.markdown("### 🔗 Agent Interaction Flow")
            render_agent_graph(st.session_state.agent_status)
            st.divider()
            st.markdown("### 🗂️ Shared Blackboard")
            render_blackboard(
                profile=results.get('profile'),
                recommendations=results.get('recommendations'),
                explanations=results.get('explanations'),
                learning_path=results.get('learning_path'),
                content=results.get('content'),
            )
            st.divider()
            st.markdown("### 💬 Agent Communication Log (FIPA-ACL)")
            st.caption("Derived from the actual sequential run. The negotiation/feedback loop "
                       "is implemented in the orchestrator scaffold — see the project README.")
            render_agent_timeline(results.get('comm_log', []))
        else:
            st.info("Run the pipeline (Live Run tab) to populate the blackboard and message log.")

    with tab_recs:
        if results and results.get('recommendations'):
            render_recommendations(results['recommendations'], results.get('rec_metrics', {}))
        else:
            st.info("No recommendations yet — run the pipeline first.")

    with tab_xai:
        if results and results.get('explanations'):
            render_xai_explorer(results['explanations'], results.get('profile'))
        else:
            st.info("No explanations yet — run the pipeline first.")

    with tab_path:
        if results and results.get('learning_path'):
            st.markdown("### 🛤️ Personalized Learning Path")
            render_learning_path(results['learning_path'])
            st.divider()
            st.markdown("### 📝 Interactive Quiz")
            if results.get('content') and results['content'].get('quizzes'):
                render_quiz_player(results['content']['quizzes'])
        else:
            st.info("No learning path yet — run the pipeline first.")


def render_overview(has_data: bool):
    """Static overview / 'what is this' tab."""
    st.markdown("""
    <div class="hero">
      <h3>What this is</h3>
      <p style="color:#cfd6e6; line-height:1.6;">
      A multi-agent system that turns raw e-learning clickstream data into
      <b>personalized, explainable</b> learning paths. Five specialized agents each own one
      sub-problem and write their results to a shared <b>blackboard</b>; the orchestrator routes
      work between them using FIPA-ACL messages (request / propose / confirm / reject).
      </p>
      <div>
        <span class="chip">👤 Profiling · embeddings + K-Means</span>
        <span class="chip">📚 Recommendation · SVD + TF-IDF + LLM</span>
        <span class="chip">🔍 Explainability · SHAP + LIME + DiCE</span>
        <span class="chip">🛤️ Path Planning · NetworkX + A* + Q-Learning</span>
        <span class="chip">📝 Content · GPT-4o</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🛠️ Under the hood")
        st.markdown("""
        - **Profiling:** Azure embeddings → K-Means → LLM cluster interpretation
        - **Recommendation:** SVD collaborative + TF-IDF content-based, LLM re-ranked
        - **Explainability:** real SHAP, LIME, and DiCE counterfactuals
        - **Path planning:** A* search, Q-Learning, topological sort (best of three)
        - **Content:** GPT-4o quiz / summary generation
        """)
    with col2:
        st.markdown("#### 📐 Architecture & honesty")
        st.markdown("""
        The five agents above run as a **sequential pipeline** here. The
        **negotiation / blackboard orchestrator** (LangGraph + FIPA-ACL) is implemented
        as a **scaffold** in `src/orchestrator.py` — the message log in *Agents & Blackboard*
        reflects the real run, not a live negotiation loop.

        *Planned (roadmap):* wiring the orchestrator to the live agents, AutoGen
        conversations, RAG for content, and a FastAPI service.
        """)

    if not has_data:
        st.warning("ℹ️ Running without the OULAD dataset — add CSVs to `data/oulad/` to enable the pipeline.")
    st.success("👉 Head to the **Live Run** tab, pick a student in the sidebar, and hit **Run Full Pipeline**.")


def run_tab(data, student_id, custom_profile):
    """Live Run tab: trigger + streaming agent feed."""
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"**Target student:** `{student_id}`"
                    + ("  ·  *custom profile*" if custom_profile else ""))
    with c2:
        if st.button("▶️ Run", type="primary", use_container_width=True, key="run_inline"):
            st.session_state.pipeline_running = True
            st.session_state.agent_status = {k: 'pending' for k, _, _ in AGENTS}

    if st.session_state.pipeline_running:
        run_pipeline(data, student_id, custom_profile)
        st.session_state.pipeline_running = False
    elif st.session_state.pipeline_results:
        st.divider()
        st.markdown("#### 💬 Last run — agent communication")
        render_agent_timeline(st.session_state.pipeline_results.get('comm_log', []))
    else:
        st.info("Press **Run** to watch the agents collaborate in real time.")


def run_pipeline(data, student_id, custom_profile=None):
    """Run the full agent pipeline, streaming the ACL feed as each agent finishes."""
    progress_bar = st.progress(0, text="Starting pipeline...")
    st.divider()
    st.markdown("#### 💬 Agents collaborating…")
    feed = st.empty()

    runner = PipelineRunner(data)
    acc = {}  # accumulating results for incremental comm-log

    def step(key, label, pct, fn):
        st.session_state.agent_status[key] = 'running'
        out = fn()
        st.session_state.agent_status[key] = 'complete'
        progress_bar.progress(pct, text=f"✅ {label} complete")
        with feed.container():
            render_agent_timeline(build_comm_log(**acc))
        return out

    try:
        profile = step('profiling', 'Profiling Agent', 20,
                       lambda: runner.run_profiling(student_id, custom_profile))
        acc['profile'] = profile

        rec = runner.run_recommendations(student_id, profile)
        recommendations, rec_metrics = rec if isinstance(rec, tuple) else (rec, {})
        st.session_state.agent_status['recommendation'] = 'complete'
        acc['recommendations'] = recommendations
        acc['rec_metrics'] = rec_metrics
        progress_bar.progress(40, text="✅ Recommendation complete")
        with feed.container():
            render_agent_timeline(build_comm_log(**acc))

        explanations = step('xai', 'XAI Agent', 60,
                            lambda: runner.run_xai(student_id, profile, recommendations))
        acc['explanations'] = explanations

        learning_path = step('path_planning', 'Path Planning Agent', 80,
                             lambda: runner.run_path_planning(student_id, profile))
        acc['learning_path'] = learning_path

        content = step('content_gen', 'Content Generator', 100,
                      lambda: runner.run_content_generation(learning_path))
        acc['content'] = content

        st.session_state.pipeline_results = {
            'profile': profile,
            'recommendations': recommendations,
            'rec_metrics': rec_metrics,
            'explanations': explanations,
            'learning_path': learning_path,
            'content': content,
            'comm_log': build_comm_log(**acc),
        }
        st.success("All agents completed. See **Agents & Blackboard**, **Recommendations**, "
                   "**Explainability**, and **Path & Quiz** tabs for results.")

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
