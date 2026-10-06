"""
Blackboard & Agent-Communication Component
==========================================
Makes the multi-agent collaboration visible:

- `render_agent_timeline`  -> a FIPA-ACL style message feed (orchestrator <-> agents)
- `render_blackboard`      -> the shared state object filling up slot by slot

The messages are derived from the *actual* sequential pipeline run (real
confidences, counts, chosen algorithm), so nothing here is fabricated. The
negotiation / feedback loop itself lives in the orchestrator scaffold
(src/orchestrator.py) — see the project README for the implemented-vs-roadmap
status.
"""

import streamlit as st

# ---- Visual vocabulary ---------------------------------------------------

AGENT_ICON = {
    "Orchestrator": "🧭",
    "ProfilingAgent": "👤",
    "RecommendationAgent": "📚",
    "XAIAgent": "🔍",
    "PathPlanningAgent": "🛤️",
    "ContentGenerator": "📝",
}

# performative -> (label color, kind)  kind drives left/right alignment
PERFORMATIVE_COLOR = {
    "REQUEST": "#7B8CFF",
    "QUERY": "#7B8CFF",
    "CALL-FOR-PROPOSAL": "#7B8CFF",
    "INFORM": "#2EC4A6",
    "PROPOSE": "#2EC4A6",
    "CONFIRM": "#3DD68C",
    "REJECT": "#F2555A",
    "FAILURE": "#F2555A",
}


def _get(obj, attr, default=None):
    """Read attr from object or dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


# ---- Build the communication log from real results ----------------------

def build_comm_log(profile=None, recommendations=None, rec_metrics=None,
                   explanations=None, learning_path=None, content=None):
    """
    Translate the actual agent outputs into an ordered FIPA-ACL message log.
    Every value shown comes straight from the agents that just ran.
    """
    log = []

    def msg(performative, sender, receiver, content):
        log.append({
            "performative": performative,
            "sender": sender,
            "receiver": receiver,
            "content": content,
        })

    # Each agent's request + response are emitted together, only once that
    # agent has produced output — so the feed grows one block per stage when
    # called incrementally during a live run.

    # 1. Profiling
    if profile is not None:
        msg("REQUEST", "Orchestrator", "ProfilingAgent", "Profile this learner from their VLE activity.")
        style = _get(profile, "learning_style", "?")
        eng = _get(profile, "engagement_level", "?")
        cluster = _get(profile, "cluster_id", "?")
        msg("INFORM", "ProfilingAgent", "Orchestrator",
            f"Cluster {cluster} · “{style}” · engagement: {eng}.")

    # 2. Recommendation (Contract-Net: call for proposal -> propose)
    if recommendations is not None:
        msg("CALL-FOR-PROPOSAL", "Orchestrator", "RecommendationAgent",
            "Propose learning items for this profile.")
        recs = _get(recommendations, "recommendations", []) or []
        mrr = (rec_metrics or {}).get("mrr")
        detail = f"{len(recs)} items via hybrid SVD + TF-IDF, LLM-reranked"
        if mrr is not None:
            detail += f" · MRR {mrr:.3f}"
        msg("PROPOSE", "RecommendationAgent", "Orchestrator", detail + ".")

    # 3. XAI validation (this is where negotiation *would* trigger on reject)
    if explanations is not None:
        msg("QUERY", "Orchestrator", "XAIAgent",
            "Can these recommendations be explained? (SHAP / LIME / DiCE)")
        conf = _get(explanations, "confidence", None)
        if conf is not None and conf >= 0.4:
            msg("CONFIRM", "XAIAgent", "Orchestrator",
                f"Explainable — confidence {conf:.2f}. Proceed.")
        elif conf is not None:
            msg("REJECT", "XAIAgent", "Orchestrator",
                f"Low explanation confidence ({conf:.2f}) — would request alternatives.")
        else:
            msg("CONFIRM", "XAIAgent", "Orchestrator", "Explanations generated.")

    # 4. Path planning
    if learning_path is not None:
        msg("REQUEST", "Orchestrator", "PathPlanningAgent", "Plan a learning path toward the goal.")
        algo = str(_get(learning_path, "algorithm_used", "?")).upper()
        weeks = _get(learning_path, "total_weeks", "?")
        msg("INFORM", "PathPlanningAgent", "Orchestrator",
            f"Path chosen via {algo} · {weeks} weeks.")

    # 5. Content generation
    if content is not None:
        msg("REQUEST", "Orchestrator", "ContentGenerator", "Generate materials for the first topics.")
        quizzes = content.get("quizzes", []) if isinstance(content, dict) else []
        summaries = content.get("summaries", []) if isinstance(content, dict) else []
        msg("INFORM", "ContentGenerator", "Orchestrator",
            f"{len(quizzes)} quiz(zes) + {len(summaries)} summary(ies) generated.")

    return log


# ---- Render the timeline -------------------------------------------------

def _acl_bubble_html(m):
    color = PERFORMATIVE_COLOR.get(m["performative"], "#7B8CFF")
    from_orch = m["sender"] == "Orchestrator"
    align = "flex-start" if from_orch else "flex-end"
    s_icon = AGENT_ICON.get(m["sender"], "•")
    r_icon = AGENT_ICON.get(m["receiver"], "•")
    sender_short = m["sender"].replace("Agent", "")
    return f"""
    <div style="display:flex; justify-content:{align}; margin:0.35rem 0;">
      <div style="max-width:82%; background:#1B2130; border:1px solid #2A3346;
                  border-left:3px solid {color}; border-radius:10px; padding:0.55rem 0.8rem;">
        <div style="font-size:0.7rem; color:#9aa4b8; margin-bottom:0.2rem;">
          {s_icon} <b style="color:#cfd6e6;">{sender_short}</b>
          <span style="color:#5c6680;">&nbsp;→&nbsp;</span> {r_icon} {m['receiver'].replace('Agent','')}
          <span style="background:{color}22; color:{color}; border:1px solid {color}55;
                       padding:0.05rem 0.4rem; border-radius:8px; font-size:0.62rem;
                       margin-left:0.4rem; letter-spacing:0.04em;">{m['performative']}</span>
        </div>
        <div style="font-size:0.9rem; color:#e6e9ef; line-height:1.4;">{m['content']}</div>
      </div>
    </div>
    """


def render_agent_timeline(comm_log):
    """Render the full ACL message feed."""
    if not comm_log:
        st.info("Run the pipeline to see the agents communicate.")
        return
    html = "".join(_acl_bubble_html(m) for m in comm_log)
    st.markdown(html, unsafe_allow_html=True)


# ---- Render the shared-state blackboard ----------------------------------

def render_blackboard(profile=None, recommendations=None, explanations=None,
                      learning_path=None, content=None):
    """Show the shared state object as slots that fill as agents write to it."""
    st.caption(
        "The orchestrator's shared state (the *blackboard*). Each agent writes "
        "its result here; downstream agents read from it."
    )

    recs = _get(recommendations, "recommendations", []) or []
    quizzes = content.get("quizzes", []) if isinstance(content, dict) else []
    conf = _get(explanations, "confidence", None)

    slots = [
        ("👤", "student_profile",
         _get(profile, "learning_style") and f"“{_get(profile, 'learning_style')}” · {_get(profile,'engagement_level','?')}",
         profile is not None),
        ("📚", "recommendations",
         f"{len(recs)} items (hybrid)" if recs else None,
         bool(recs)),
        ("🔍", "explanations",
         f"SHAP+LIME+DiCE · conf {conf:.2f}" if conf is not None else None,
         explanations is not None),
        ("🛤️", "learning_path",
         f"{str(_get(learning_path,'algorithm_used','?')).upper()} · {_get(learning_path,'total_weeks','?')} wks" if learning_path is not None else None,
         learning_path is not None),
        ("📝", "generated_content",
         f"{len(quizzes)} quiz(zes)" if quizzes else None,
         bool(quizzes)),
    ]

    cols = st.columns(len(slots))
    for col, (icon, key, summary, filled) in zip(cols, slots):
        accent = "#2EC4A6" if filled else "#3a4253"
        body = summary if filled else "— empty —"
        check = "✓" if filled else "○"
        col.markdown(f"""
        <div style="background:#171C28; border:1px solid #2A3346; border-top:3px solid {accent};
                    border-radius:10px; padding:0.7rem 0.6rem; min-height:120px; text-align:center;">
          <div style="font-size:1.5rem;">{icon}</div>
          <div style="font-family:monospace; font-size:0.72rem; color:#9aa4b8; margin:0.3rem 0;">{key}</div>
          <div style="color:{accent}; font-weight:bold;">{check}</div>
          <div style="font-size:0.72rem; color:#cfd6e6; margin-top:0.25rem; line-height:1.3;">{body}</div>
        </div>
        """, unsafe_allow_html=True)
