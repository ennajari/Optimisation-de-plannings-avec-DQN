# Optimisation de plannings avec DQN — Wiki

> Ceci est un brouillon de wiki (`docs/wiki-draft/`), destiné à être copié dans le Wiki GitHub du dépôt si le mainteneur le souhaite. Il n'a pas été publié automatiquement.

## Qu'est-ce que ce projet ?

Un projet académique en deux parties :

1. **Pipeline d'entraînement** (`notebooks/`) : exploration et prétraitement du jeu de données [ATUS](https://www.bls.gov/tus/) (American Time Use Survey), puis entraînement d'un agent **Deep Q-Network** sur un environnement `gym` personnalisé (`environment/schedule_env.py`) qui apprend à placer des activités dans les 24 créneaux horaires d'une journée.
2. **Application de démonstration** (`ui/app.py`) : une interface Streamlit qui génère et visualise des plannings journaliers/hebdomadaires. Important : cette application **n'utilise pas** le modèle DQN entraîné — elle utilise un optimiseur à base de règles qui imite le comportement attendu (voir [Architecture](Architecture.md)).

## Pages

- [Getting Started](Getting-Started.md) — installer et lancer l'app, ou reproduire l'entraînement.
- [Architecture](Architecture.md) — comment les notebooks, l'environnement Gym et l'application s'articulent (et où elles ne sont *pas* connectées).
- [FAQ](FAQ.md) — questions fréquentes, notamment sur l'écart entre le nom du dépôt et le comportement réel de la démo.

## Liens utiles

- [README du dépôt](../../README.md)
- [CHANGELOG](../../CHANGELOG.md)
- [Code source de l'environnement Gym](../../environment/schedule_env.py)
- [Code source de l'application Streamlit](../../ui/app.py)
