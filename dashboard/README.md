# 🎓 E-Learning MAS Dashboard

Interactive dashboard for the Multi-Agent E-Learning Recommendation System.

## Features

- **👤 Student Profile Panel** - Select existing students or create custom profiles
- **🔗 Agent Interaction Graph** - Visualize how agents communicate
- **📚 Recommendations** - View personalized recommendations with metrics
- **🔍 XAI Explorer** - Interactive SHAP, LIME, and counterfactual explanations
- **🛤️ Learning Path** - Visual timeline of personalized learning journey
- **📝 Interactive Quizzes** - Take quizzes with instant feedback

## Installation

1. Copy the `dashboard` folder into your project root:
   ```
   elearning-mas-v2/
   ├── dashboard/          <-- Put it here
   │   ├── app.py
   │   ├── components/
   │   ├── utils/
   │   └── ...
   ├── src/
   ├── data/
   └── main.py
   ```

2. Install additional requirements:
   ```bash
   pip install streamlit plotly
   ```

## Running the Dashboard

From the project root directory:

```bash
cd elearning-mas-v2
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Usage

1. **Select a Student** - Use the sidebar to pick an existing student or create a custom profile
2. **Run Pipeline** - Click "🚀 Run Full Pipeline" to execute all agents
3. **Explore Results** - View recommendations, explanations, learning path, and quizzes
4. **Take Quizzes** - Switch to "Take Quiz" mode to test your knowledge

## Components

| Component | Description |
|-----------|-------------|
| `student_panel.py` | Editable student profile with sliders |
| `agent_graph.py` | Animated agent interaction visualization |
| `recommendations.py` | Recommendation cards with metrics |
| `xai_explorer.py` | Interactive SHAP/LIME/Counterfactual panels |
| `learning_path.py` | Timeline visualization of learning path |
| `quiz_player.py` | Interactive quiz with scoring |

## Screenshots

(Add screenshots here after running)

## Requirements

- Python 3.9+
- Streamlit 1.28+
- Plotly 5.18+
- All main project dependencies
