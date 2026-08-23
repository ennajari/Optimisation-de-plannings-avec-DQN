# Architecture

## Vue d'ensemble

Ce projet a deux chemins de code qui **ne communiquent pas entre eux** :

```text
┌─────────────────────────────┐        ┌──────────────────────────────┐
│      Training pipeline      │        │      Demo application         │
│      (notebooks/, offline)  │        │      (ui/app.py, deployed)    │
├─────────────────────────────┤        ├──────────────────────────────┤
│ 1_data_exploration.ipynb    │        │ Streamlit UI                  │
│   → explore ATUS dataset    │        │   ├── SimpleScheduleOptimizer │
│ 2_data_preprocessing.ipynb  │        │   │   (rule-based scoring,    │
│   → cleaned_data, encoders  │        │   │    hardcoded preferences, │
│ ScheduleEnv (Gym env)       │        │   │    + randomness)          │
│   environment/schedule_env  │        │   ├── Plotly / Matplotlib     │
│ 3_dqn_training.ipynb        │        │   │   visualizations          │
│   → trains DQN (TensorFlow) │        │   └── CSV export              │
│   → models/dqn_schedule_    │        │                                │
│     model.h5                │        │  (does NOT load the .h5 model)│
└─────────────────────────────┘        └──────────────────────────────┘
        no connection between the two paths ───────────────────►
```

## Training pipeline

- **Data**: [ATUS](https://www.bls.gov/tus/) (American Time Use Survey), not included in the repo.
- **Environment**: `environment/schedule_env.py` defines `ScheduleEnv(gym.Env)` — 24 hourly time slots, an action space over activities/slots, and a reward based on how well the schedule matches historical preferences and avoids conflicts.
- **Agent**: trained inside `notebooks/3_dqn_training.ipynb` using TensorFlow, saved to `models/dqn_schedule_model.h5`.
- This path requires the extra dependencies listed in [Getting-Started.md](Getting-Started.md) (TensorFlow, Gym, scikit-learn, seaborn, tqdm) — they are intentionally **not** in the root `requirements.txt`.

## Demo application

- `ui/app.py` is explicitly documented in its own docstring as *"a simplified version that works without external model files"*.
- Its core class, `SimpleScheduleOptimizer`, scores each (activity, hour) pair using hardcoded `ACTIVITY_PREFERENCES` (preferred hours, duration, priority) plus a small random perturbation, and a weekend adjustment. It picks the highest-scoring combination for each remaining slot.
- It never imports TensorFlow, never references `models/dqn_schedule_model.h5`, and can run entirely on the lightweight `requirements.txt`.
- This is why the app can be deployed cheaply (e.g. Streamlit Community Cloud) without bundling TensorFlow — but it also means the "DQN" in the project name does not describe the running demo, only the offline training notebooks.

## Deployment

- `Dockerfile` + `docker-compose.yml` package `ui/app.py` behind Streamlit on port 8501.
- `.devcontainer/devcontainer.json` provides a Codespaces environment that installs `requirements.txt` and launches the same app.
- `Jenkinsfile` describes a build/test/push-to-DockerHub/deploy pipeline, but it references an external Jenkins server and Docker Hub credentials that are not part of this repository — it is a template, not an active pipeline here.

## Closing the gap (roadmap)

To make the demo actually reflect the project name, `ui/app.py` would need to load `models/dqn_schedule_model.h5` (e.g. via `tensorflow.keras.models.load_model`) and use the trained agent's Q-values to pick actions from the same observation space defined in `ScheduleEnv`, instead of (or alongside, as a comparison) the current heuristic.
