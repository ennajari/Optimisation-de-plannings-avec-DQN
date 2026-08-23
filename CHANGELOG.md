# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project has no tagged releases yet — everything below is reconstructed
from the real commit history (`git log`) and grouped under `[Unreleased]`.

## [Unreleased]

### Added
- Custom OpenAI Gym environment (`environment/schedule_env.py`, `ScheduleEnv`) modeling a 24-hour daily schedule, built on top of the [ATUS](https://www.bls.gov/tus/) (American Time Use Survey) dataset.
- Data exploration and preprocessing notebooks (`notebooks/1_data_exploration.ipynb`, `notebooks/2_data_preprocessing.ipynb`).
- DQN training notebook (`notebooks/3_dqn_training.ipynb`, TensorFlow) producing a trained checkpoint (`models/dqn_schedule_model.h5`).
- Standalone Streamlit UI (`ui/app.py`) with a rule-based schedule generator, Plotly/Matplotlib visualizations, and CSV export.
- Docker support: `Dockerfile` and `docker-compose.yml` to run the Streamlit app in a container (port 8501).
- Dev Container configuration (`.devcontainer/devcontainer.json`) for GitHub Codespaces.
- GitHub Actions workflows: `python-package.yml` (lint + test matrix), `python-publish.yml` (PyPI publish template), `generator-generic-ossf-slsa3-publish.yml` (SLSA provenance template).
- `Jenkinsfile` as an alternative CI/CD pipeline template (build/test/push/deploy via Docker Hub — requires external Jenkins + registry credentials that are not part of this repo).

### Changed
- `requirements.txt` was progressively trimmed from a full training environment (TensorFlow, Gym, scikit-learn, seaborn, stable-baselines3) down to just what the deployed Streamlit app needs (`streamlit`, `numpy`, `pandas`, `matplotlib`, `plotly`) — the heavier ML dependencies are only needed to run the notebooks, not the shipped app (see README "Getting Started" for both install paths).
- Default port and packaging fixed up across several commits (`system packages`, `tensorflow-cpu`, `gym openai`, `deploy`).

### Fixed
- `.gitignore` was corrupted (mixed UTF-16/UTF-8 encoding from a Windows shell redirect) and only matched one literal path; rewritten as plain UTF-8 with standard Python/data ignore rules.
- Removed an accidentally committed `.pyc` build artifact (`environment/__pycache__/schedule_env.cpython-312.pyc`).

### Known limitations (documented, not fixed in this PR)
- The shipped `ui/app.py` does **not** load `models/dqn_schedule_model.h5` or run any TensorFlow inference. Its own docstring describes it as "a simplified version that works without external model files" — it uses a hand-written scoring heuristic (`SimpleScheduleOptimizer`) that "mimics DQN behavior" rather than the trained agent. See README for details.
- No automated test suite exists yet, so the `unit-tests`/`pytest` step in `python-package.yml` and the `pytest tests/` step in `Jenkinsfile` currently have nothing to collect.
