# Assistant Personnel pour la Gestion du Temps

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.24.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Une application intelligente conçue pour optimiser votre emploi du temps quotidien ou hebdomadaire en utilisant un algorithme d'apprentissage par renforcement basé sur Deep Q-Network (DQN). Ce projet analyse vos habitudes et préférences pour suggérer des plannings qui maximisent votre productivité et votre satisfaction.

## 📋 Table des matières

- [Aperçu du projet](#aperçu-du-projet)
- [Caractéristiques principales](#caractéristiques-principales)
- [Technologies utilisées](#technologies-utilisées)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Méthodologie](#méthodologie)
- [Résultats attendus](#résultats-attendus)
- [Limitations](#limitations)
- [Améliorations futures](#améliorations-futures)
- [Contribution](#contribution)
- [Licence](#licence)

## 🎯 Aperçu du projet

L'objectif de ce projet est de créer un assistant personnel qui aide les utilisateurs à planifier leurs journées ou semaines de manière optimale. En exploitant les données historiques des activités et un environnement de simulation basé sur OpenAI Gym, le modèle DQN propose des emplois du temps adaptés aux préférences de l'utilisateur tout en respectant des contraintes horaires spécifiques.

## ✨ Caractéristiques principales

- **Optimisation intelligente** : Utilisation d'un algorithme DQN pour générer des plannings optimisés
- **Personnalisation** : Possibilité de définir des activités personnalisées et des contraintes horaires
- **Visualisation interactive** : Interface utilisateur Streamlit avec des graphiques Plotly pour visualiser les plannings quotidiens et hebdomadaires
- **Analyse des performances** : Statistiques sur la productivité et la répartition des activités
- **Exportation** : Téléchargement des plannings sous forme de fichiers CSV

## 🛠️ Technologies utilisées

- **Python** : Langage principal pour le développement
- **TensorFlow** : Pour l'implémentation et l'entraînement du modèle DQN
- **OpenAI Gym** : Pour la création d'un environnement de simulation personnalisé
- **Streamlit** : Pour l'interface utilisateur interactive
- **Pandas/NumPy** : Pour le prétraitement et la manipulation des données
- **Plotly/Matplotlib/Seaborn** : Pour les visualisations de données
- **Scikit-learn** : Pour l'encodage des données catégoriques

## 📁 Structure du projet

```
📁 Optimisation-de-plannings-avec-DQN/
├── 📁 Data/                  # Données du projet
│   ├── 📁 Processed/        # Données prétraitées
│   │   ├── activity_encoder.pkl
│   │   └── cleaned_data.csv
│   └── 📁 raw/              # Données brutes
│       └── atus_full_selected.csv
├── 📁 environment/          # Environnement RL personnalisé
│   ├── 📄 schedule_env.py  # Implémentation de l'environnement Gym
├── 📁 models/              # Modèles entraînés
│   ├── dqn_final.h5
│   └── dqn_schedule_model.h5
├── 📁 notebooks/           # Notebooks Jupyter
│   ├── 2_data_preprocessing.ipynb  # Prétraitement des données
│   └── 3_dqn_training.ipynb       # Entraînement du DQN
└── 📁 ui/                  # Interface utilisateur
    └── 📄 app.py          # Application Streamlit
```

### Description des fichiers

#### 📂 Data/
Contient les données brutes (`raw/`) et prétraitées (`Processed/`).
- `atus_full_selected.csv` : Données brutes des activités (extraites de l'American Time Use Survey ou synthétiques pour les tests)
- `cleaned_data.csv` : Données nettoyées et prêtes pour l'entraînement
- `activity_encoder.pkl` : Encodeur pour les noms d'activités

#### 📂 environment/
- `schedule_env.py` : Implémentation de l'environnement Gym pour simuler la planification des activités

#### 📂 models/
Modèles DQN entraînés.
- `dqn_schedule_model.h5` : Modèle principal utilisé par l'application
- `dqn_final.h5` : Modèle final sauvegardé après l'entraînement

#### 📂 notebooks/
- `2_data_preprocessing.ipynb` : Prétraitement des données, incluant le nettoyage, la conversion des heures, et l'encodage des activités
- `3_dqn_training.ipynb` : Entraînement du modèle DQN avec suivi des récompenses et des pertes

#### 📂 ui/
- `app.py` : Application Streamlit pour interagir avec l'utilisateur, générer des plannings, et visualiser les résultats

## 🚀 Installation

### Prérequis

- Python 3.8+
- pip (gestionnaire de paquets Python)
- Environnement virtuel (recommandé)

### Étapes d'installation

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/votre-nom/Optimisation-de-plannings-avec-DQN.git
   cd Optimisation-de-plannings-avec-DQN
   ```

2. **Créer et activer un environnement virtuel :**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```

3. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

4. **Créer un fichier `requirements.txt` avec les dépendances suivantes :**
   ```txt
   tensorflow==2.12.0
   gym==0.23.1
   streamlit==1.24.0
   pandas==1.5.3
   numpy==1.24.3
   matplotlib==3.7.1
   seaborn==0.12.2
   plotly==5.14.1
   scikit-learn==1.2.2
   tqdm==4.65.0
   ```

5. **Vérifier les données :**
   Placez vos données brutes dans `Data/raw/atus_full_selected.csv` ou utilisez les données synthétiques générées automatiquement par l'environnement.

## 💻 Utilisation

### 1. Prétraitement des données

Exécutez le notebook `2_data_preprocessing.ipynb` pour nettoyer et transformer les données brutes :

```bash
jupyter notebook notebooks/2_data_preprocessing.ipynb
```

Ce notebook génère `cleaned_data.csv` et `activity_encoder.pkl` dans `Data/Processed/`.

### 2. Entraînement du modèle DQN

Exécutez le notebook `3_dqn_training.ipynb` pour entraîner le modèle DQN :

```bash
jupyter notebook notebooks/3_dqn_training.ipynb
```

Le modèle entraîné est sauvegardé dans `models/dqn_schedule_model.h5`.

### 3. Lancer l'application Streamlit

Lancez l'application utilisateur avec la commande suivante :

```bash
streamlit run ui/app.py
```

L'application s'ouvre dans votre navigateur par défaut. Vous pouvez :
- Choisir entre un planning quotidien ou hebdomadaire
- Personnaliser les activités et ajouter des contraintes horaires
- Visualiser les plannings et leurs statistiques
- Exporter les plannings en CSV

### Exemple d'utilisation

1. Sélectionnez "Planning quotidien" dans la barre latérale
2. Choisissez un jour (ex. : Lundi)
3. Activez ou personnalisez les activités
4. Ajoutez des contraintes horaires (ex. : "Travail" de 9h à 12h)
5. Cliquez sur "Générer planning quotidien" pour voir le planning optimisé avec des visualisations interactives

![Optimized Schedule](..\ui\1.png)
![Optimized Schedule](..\ui\2.png)
![Optimized Schedule](..\ui\3.png)
![Optimized Schedule](..\ui\33.png)
![Optimized Schedule](..\ui\/44.png)
![Optimized Schedule](..\ui\222.png)
![Optimized Schedule](..ui\Sans-titre-2025-05-06-1603.png)

## 🧠 Méthodologie

### Algorithme

Le projet utilise un **Deep Q-Network (DQN)**, un algorithme d'apprentissage par renforcement qui apprend à maximiser une récompense cumulative en planifiant des activités. Le DQN :

- Analyse les habitudes historiques des utilisateurs (heures préférées, durées, fréquences)
- Évite les conflits horaires et respecte les contraintes
- Optimise la cohérence et la satisfaction en regroupant les activités similaires

### Environnement

L'environnement (`schedule_env.py`) est construit avec OpenAI Gym :

- **Espace d'actions** : Choix d'une activité et d'un créneau horaire
- **Espace d'observation** : Inclut l'agenda actuel, le jour de la semaine, et le temps restant
- **Récompense** : Basée sur la proximité des activités planifiées avec les préférences historiques et la cohérence des plannings

### Données

Les données sont issues de l'American Time Use Survey (ATUS) ou de données synthétiques générées pour les tests. Les colonnes clés incluent :

- `ACTIVITY_NAME` : Nom de l'activité
- `TUACTDUR24` : Durée de l'activité (en minutes)
- `TUSTARTTIM` : Heure de début
- `TUDIARYDAY` : Jour de la semaine

### Évaluation

L'évaluation repose sur :

- **Productivité** : Mesurée par la récompense totale du DQN
- **Satisfaction** : Basée sur l'alignement des plannings avec les préférences historiques
- **Visualisations** : Graphiques interactifs pour analyser la répartition des activités et les scores de productivité

## 📊 Résultats attendus

- **Plannings optimisés** : Horaires cohérents et adaptés aux préférences de l'utilisateur
- **Visualisations claires** : Graphiques Plotly pour une compréhension facile des plannings
- **Statistiques utiles** : Identification des heures les plus productives et de la répartition des activités

## ⚠️ Limitations

- Les données synthétiques peuvent ne pas refléter parfaitement les comportements réels
- Le modèle DQN nécessite un entraînement suffisant pour converger vers des plannings optimaux
- Les contraintes horaires complexes peuvent réduire la flexibilité du modèle

## 🔮 Améliorations futures

- Intégration de données utilisateur en temps réel via une API
- Prise en charge de contraintes multi-jours et de dépendances entre activités
- Optimisation de l'algorithme avec des variantes avancées (Double DQN, Dueling DQN)
- Amélioration de l'interface avec des options de personnalisation avancées

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le dépôt
2. Créez une branche pour vos modifications (`git checkout -b feature/amélioration`)
3. Soumettez une pull request avec une description claire des changements

## 📄 Licence

Ce projet est sous licence MIT. Consultez le fichier [LICENSE](LICENSE) pour plus de détails.
---

⭐ N'hésitez pas à donner une étoile au projet si vous l'avez trouvé utile !



 


