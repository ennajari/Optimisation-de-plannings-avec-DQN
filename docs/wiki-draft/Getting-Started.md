# Getting Started

## Prérequis

- Python 3.11 (voir `.python-version`) ou Docker
- Pour reproduire l'entraînement : le jeu de données ATUS (non fourni dans ce dépôt)

## Option A — Lancer la démo Streamlit

C'est le chemin le plus rapide ; il ne nécessite ni TensorFlow ni le jeu de données ATUS.

```bash
git clone https://github.com/ennajari/Optimisation-de-plannings-avec-DQN.git
cd Optimisation-de-plannings-avec-DQN
pip install -r requirements.txt
streamlit run ui/app.py
```

Ouvre `http://localhost:8501`. Dans la barre latérale :
1. Choisis un planning quotidien ou hebdomadaire.
2. Sélectionne/personnalise les activités.
3. Ajoute éventuellement des contraintes horaires (ex. "Travail" de 9h à 12h).
4. Génère le planning et exporte-le en CSV si besoin.

### Avec Docker

```bash
docker compose up --build
```

## Option B — Reproduire l'entraînement DQN

Le fichier `requirements.txt` du dépôt est volontairement limité aux dépendances de l'app déployée. Pour les notebooks, installe en plus :

```bash
pip install tensorflow-cpu gym scikit-learn seaborn tqdm
```

Place les fichiers ATUS bruts dans `Data/raw/` (dossier ignoré par git, à créer toi-même), puis exécute dans l'ordre :

1. `notebooks/1_data_exploration.ipynb`
2. `notebooks/2_data_preprocessing.ipynb`
3. `notebooks/3_dqn_training.ipynb` → produit `models/dqn_schedule_model.h5`

## Où sont les tests ?

Il n'y a pas encore de suite de tests dans ce dépôt. La CI GitHub Actions (`python-package.yml`) exécute déjà une étape `pytest`, mais elle ne trouve actuellement aucun test à collecter — c'est un point d'amélioration listé dans le README.
