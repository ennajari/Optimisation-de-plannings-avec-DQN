le voila mon projet :
algorithme:Deep Q-Network 
enviroments donnes : Données d'agenda personnel et historiques d'activités mrthode utiliser :- Analyse des habitudes et préférences de l'utilisateur - Application de DQN pour suggérer des plannings optimaux - Évaluation basée sur l'amélioration de la productivité et la satisfaction de l'utilisateur .

le la struture de .
├── LICENSE (Licence MIT)
├── README.md (Description du projet)
├── requirements.txt (Dépendances Python)
├── .vscode/
│   └── settings.json (Configurations VS Code)
├── dashboard/
│   └── app.py (Application dashboard)
├── Data/
│   └── raw/
│       └── atus_full_selected.csv (Données brutes)
├── notebooks/
│   ├── 1_data_exploration.ipynb (Exploration des données)
│   ├── 2_data_preprocessing.ipynb (Prétraitement)
│   ├── environment_setup.py (Configuration)
│   ├── 4_dqn_implementation.ipynb (Implémentation DQN)
│   ├── 5_model_training.ipynb (Entraînement)
│   └── 6_evaluation.ipynb (Évaluation)

et le voila le code de :
│   ├── 1_data_exploration.ipynb:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Définir le style des visualisations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
# Charger les données
file_path = "../Data/raw/atus_full_selected.csv"
df = pd.read_csv(file_path)
# Afficher les premières lignes du dataset
print("Aperçu des données :")
display(df.head())
df.info()
# Afficher la forme du dataset
print(f"\nDimensions du dataset : {df.shape}")
# Afficher les types de données
print("\nTypes de données :")
print(df.dtypes)
# Vérifier les valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())
# Statistiques descriptives
print("\n✅ Statistiques descriptives des colonnes numériques :")
display(df.describe())
# Distribution de la durée des activités
plt.figure(figsize=(12, 6))
sns.histplot(df['TUACTDUR24'], bins=50, kde=True)
plt.title("Distribution de la durée des activités (TUACTDUR24)", fontsize=14)
plt.xlabel("Durée (minutes)", fontsize=12)
plt.ylabel("Fréquence", fontsize=12)
plt.axvline(df['TUACTDUR24'].mean(), color='red', linestyle='--', label=f"Moyenne: {df['TUACTDUR24'].mean():.2f}")
plt.axvline(df['TUACTDUR24'].median(), color='green', linestyle='--', label=f"Médiane: {df['TUACTDUR24'].median():.2f}")
plt.legend()
plt.tight_layout()
plt.show()
# Top 10 des activités les plus fréquentes
plt.figure(figsize=(14, 8))
top_activities = df['ACTIVITY_NAME'].value_counts().head(10)
sns.barplot(x=top_activities.values, y=top_activities.index, palette="viridis")
plt.title("Top 10 des activités les plus fréquentes", fontsize=14)
plt.xlabel("Nombre d'occurrences", fontsize=12)
plt.ylabel("Nom de l'activité", fontsize=12)
plt.tight_layout()
plt.show()
# Répartition des activités selon le lieu
plt.figure(figsize=(12, 6))
sns.countplot(x='TEWHERE', data=df, palette="Set2")
plt.title("Répartition des activités selon TEWHERE (lieu)", fontsize=14)
plt.xlabel("Code lieu (TEWHERE)", fontsize=12)
plt.ylabel("Nombre d'activités", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Distribution des activités par heure de la journée
plt.figure(figsize=(14, 6))
# Convertir TUSTARTTIM en heure du jour (0-23)
df['hour'] = df['TUSTARTTIM'].apply(lambda x: int(str(x).zfill(4)[:2]) if pd.notnull(x) else np.nan)
hour_counts = df.groupby('hour').size()
sns.barplot(x=hour_counts.index, y=hour_counts.values, palette="rocket")
plt.title("Distribution des activités par heure de la journée", fontsize=14)
plt.xlabel("Heure", fontsize=12)
plt.ylabel("Nombre d'activités", fontsize=12)
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()
# Matrice de corrélation
plt.figure(figsize=(12, 10))
corr_matrix = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, vmin=-1, vmax=1)
plt.title("Matrice de corrélation entre variables numériques", fontsize=14)
plt.tight_layout()
plt.show()
# Analyse temporelle : durée moyenne des activités par heure
plt.figure(figsize=(14, 6))
hour_durations = df.groupby('hour')['TUACTDUR24'].mean()
sns.lineplot(x=hour_durations.index, y=hour_durations.values, marker='o')
plt.title("Durée moyenne des activités par heure de la journée", fontsize=14)
plt.xlabel("Heure", fontsize=12)
plt.ylabel("Durée moyenne (minutes)", fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True)
plt.tight_layout()
plt.show()
   les rsulta de ceficheirs :
           <class 'pandas.core.frame.DataFrame'>
RangeIndex: 3347093 entries, 0 to 3347092
Data columns (total 12 columns):
 #   Column         Dtype  
---  ------         -----  
 0   TUCASEID       int64  
 1   TUACTIVITY_N   int64  
 2   TUACTDUR24     int64  
 3   TUSTARTTIM     object 
 4   TEWHERE        int64  
 5   ACTIVITY_NAME  object 
 6   TUDIARYDAY     int64  
 7   TUFNWGTP001    float64
 8   TUFNWGTP002    float64
 9   GEMETSTA       int64  
 10  GTMETSTA       int64  
 11  hour           int64  
dtypes: float64(2), int64(8), object(2)
memory usage: 306.4+ MB    
Dimensions du dataset : (3347093, 11) Types de données :
TUCASEID           int64
TUACTIVITY_N       int64
TUACTDUR24         int64
TUSTARTTIM        object
TEWHERE            int64
ACTIVITY_NAME     object
TUDIARYDAY         int64
TUFNWGTP001      float64
TUFNWGTP002      float64
GEMETSTA           int64
GTMETSTA           int64
dtype: object
Valeurs manquantes par colonne :
TUCASEID              0
TUACTIVITY_N          0
TUACTDUR24            0
TUSTARTTIM            0
TEWHERE               0
ACTIVITY_NAME    763110
TUDIARYDAY            0
TUFNWGTP001           0
TUFNWGTP002           0
GEMETSTA              0
GTMETSTA              0
dtype: int64

│   ├── 2_data_preprocessing.ipynb (Prétraitement):
     import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# 1. Chargement des données
df = pd.read_csv("../Data/raw/atus_full_selected.csv", nrows=50000)
# 2. Traitement des valeurs manquantes
print(f"\n5. Traitement des valeurs manquantes...")
print(f"Nombre de lignes avant suppression : {len(df)}")
df = df.dropna(subset=['ACTIVITY_NAME'])
print(f"Nombre de lignes après suppression : {len(df)}")
df = df.reset_index(drop=True)
# 3. Encodage des variables catégorielles
print("\n6. Encodage des variables catégorielles...")
label_encoders = {}
# Encodage de ACTIVITY_NAME
le_activity = LabelEncoder()
df['ACTIVITY_NAME_ENC'] = le_activity.fit_transform(df['ACTIVITY_NAME'])
label_encoders['ACTIVITY_NAME'] = le_activity
print(f"Nombre d'activités uniques : {len(le_activity.classes_)}")
# Encodage de TEWHERE si elle existe
if 'TEWHERE' in df.columns:
    le_location = LabelEncoder()
    df['TEWHERE_ENC'] = le_location.fit_transform(df['TEWHERE'].astype(str))
    label_encoders['TEWHERE'] = le_location
else:
    print("Colonne 'TEWHERE' non trouvée. Remplissage avec 0.")
    df['TEWHERE_ENC'] = 0
# 4. Transformation de START_TIME en minutes
def parse_start_time(val):
    try:
        val_str = str(val).zfill(4)
        hour = int(val_str[:2])
        minute = int(val_str[2:])
        return hour * 60 + minute
    except:
        return 0
df['START_TIME_MINUTES'] = df['TUSTARTTIM'].apply(parse_start_time)
df['START_TIME_MINUTES'] = df['START_TIME_MINUTES'].fillna(0)
# 5. Création de colonnes supplémentaires
df['DAY_OF_WEEK'] = df['TUDIARYDAY'] - 1  # 0=dimanche, 6=samedi
df['IS_WEEKEND'] = df['DAY_OF_WEEK'].apply(lambda x: 1 if x in [0, 6] else 0)
df['hour'] = df['START_TIME_MINUTES'] // 60
# 6. Sélection des colonnes utiles
cols_to_keep = [
    'TUCASEID',
    'TUACTIVITY_N',
    'TUACTDUR24',
    'START_TIME_MINUTES',
    'TEWHERE_ENC',
    'ACTIVITY_NAME_ENC',
    'DAY_OF_WEEK',
    'IS_WEEKEND',
    'TUFNWGTP001',
    'TUFNWGTP002',
    'hour'
]
# Vérification que toutes les colonnes existent
missing_cols = [col for col in cols_to_keep if col not in df.columns]
if missing_cols:
    raise KeyError(f"Colonnes manquantes : {missing_cols}")
df_selected = df[cols_to_keep]
# 7. Normalisation des variables numériques
print("\n9. Normalisation des variables numériques...")
cols_to_scale = ['TUACTDUR24', 'START_TIME_MINUTES', 'TUFNWGTP001', 'TUFNWGTP002', 'hour']
scaler = StandardScaler()
df_scaled = df_selected.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df_selected[cols_to_scale])

# 8. Enregistrement du dataset prétraité
output_dir = "../Data/processed/"
os.makedirs(output_dir, exist_ok=True)

df_scaled.to_csv(os.path.join(output_dir, "preprocessed_data.csv"), index=False)
# 9. Sauvegarde des encodeurs et du scaler
with open(os.path.join(output_dir, "label_encoders.pkl"), "wb") as f:
    pickle.dump(label_encoders, f)

with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
# 10. Visualisations
plt.figure(figsize=(10, 5))
sns.histplot(df['TUACTDUR24'], bins=50, kde=True)
plt.title("Distribution de la durée des activités (TUACTDUR24)")
plt.xlabel("Durée (minutes)")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 5))
sns.histplot(df['START_TIME_MINUTES'], bins=48, kde=True)
plt.title("Distribution des heures de début d'activités")
plt.xlabel("Minutes depuis minuit")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 6))
top_activities = df['ACTIVITY_NAME'].value_counts().head(10)
sns.barplot(x=top_activities.values, y=top_activities.index, palette="viridis")
plt.title("Top 10 des activités les plus fréquentes")
plt.xlabel("Nombre d'occurrences")
plt.ylabel("Nom de l'activité")
plt.tight_layout()
plt.show()
   les rsultat de ce fichiers:
              5. Traitement des valeurs manquantes...
Nombre de lignes avant suppression : 50000
Nombre de lignes après suppression : 37933  
Nombre d'activités uniques : 18
Index(['TUCASEID', 'TUACTIVITY_N', 'TUACTDUR24', 'TUSTARTTIM', 'TEWHERE',
       'ACTIVITY_NAME', 'TUDIARYDAY', 'TUFNWGTP001', 'TUFNWGTP002', 'GEMETSTA',
       'GTMETSTA', 'ACTIVITY_NAME_ENC', 'TEWHERE_ENC', 'START_TIME_MINUTES',
       'DAY_OF_WEEK', 'IS_WEEKEND', 'hour'],
      dtype='object')
le code de ficheirs :
# 1. Import des bibliothèques
import numpy as np
import pandas as pd
import random
from gym import Env
from gym.spaces import Discrete, Box

# 2. Charger les données prétraitées
data_path = "../Data/processed/preprocessed_data.csv"
df = pd.read_csv(data_path)

print("✅ Aperçu des données :")
display(df.head())

# 3. Paramètres de l’environnement
NUM_TIME_SLOTS = 24  # Par exemple : 24 slots d'1h
NUM_ACTIVITIES = df['ACTIVITY_NAME_ENC'].nunique()  # Utilisation de la colonne encodée

# 4. Création de l’environnement
class ScheduleEnv(Env):
    def __init__(self, data, num_slots=24):
        super(ScheduleEnv, self).__init__()
        
        self.data = data
        self.num_slots = num_slots
        
        # Action : choisir une activité pour chaque slot
        self.action_space = Discrete(NUM_ACTIVITIES)
        
        # Observation : état actuel (slot courant + historique simple)
        self.observation_space = Box(low=0, high=1, shape=(num_slots,), dtype=np.float32)
        
        self.reset()

    def reset(self):
        self.current_slot = 0
        self.schedule = np.zeros(self.num_slots)
        return self.schedule

    def step(self, action):
        done = False
        
        # Enregistrer l'action (activité choisie) dans le slot courant
        self.schedule[self.current_slot] = action

        # Simuler une "récompense"
        reward = self._calculate_reward(action, self.current_slot)

        # Avancer d’un slot
        self.current_slot += 1

        if self.current_slot >= self.num_slots:
            done = True

        return self.schedule, reward, done, {}

    def _calculate_reward(self, action, time_slot):
        """
        Simule une récompense en fonction de l'heure et du type d'activité :
        - Par exemple, dormir la nuit, travailler en journée = bon
        """
        activity_name_enc = self.data[self.data['ACTIVITY_NAME_ENC'] == action]['ACTIVITY_NAME_ENC'].mode()
        if len(activity_name_enc) > 0:
            activity = activity_name_enc.iloc[0]
        else:
            activity = "Unknown"

        # Exemple de règles simples
        if "sleep" in str(activity).lower() and (time_slot >= 22 or time_slot < 6):
            return 1.0
        elif "work" in str(activity).lower() and (9 <= time_slot <= 17):
            return 1.0
        else:
            return 0.2

    def render(self):
        print(f"Planning courant : {self.schedule}")
le code ficheirs │   ├── 4_dqn_implementation.ipynb (Implémentation DQN):
# 4_dqn_implementation.ipynb
# Implémentation du Deep Q-Network pour l'optimisation de planning

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from collections import deque
import random
import os
import sys

# Ajouter le chemin du dossier parent pour importer environment_setup.py
sys.path.append('..')
from environment_setup import ScheduleEnv

# Vérifier la version de TensorFlow
print(f"TensorFlow version: {tf.__version__}")

# Définir les chemins de fichiers
DATA_PATH = "../Data/processed/preprocessed_data.csv"
MODEL_PATH = "../models/dqn_model"
os.makedirs("../models", exist_ok=True)

# 1. Charger les données prétraitées
print("Chargement des données prétraitées...")
df = pd.read_csv(DATA_PATH)
print(f"Données chargées: {df.shape[0]} entrées")

# Configurer l'environnement
env = ScheduleEnv(df)
print(f"Environnement créé avec {env.action_space.n} actions possibles")

# 2. Implémentation de la mémoire d'expérience (Experience Replay)
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def size(self):
        return len(self.buffer)

# 3. Implémentation de l'agent DQN
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Paramètres de l'algorithme
        self.gamma = 0.95  # Facteur d'actualisation
        self.epsilon = 1.0  # Exploration initiale
        self.epsilon_min = 0.01  # Exploration minimale
        self.epsilon_decay = 0.995  # Taux de décroissance de l'exploration
        self.learning_rate = 0.001  # Taux d'apprentissage
        self.batch_size = 64  # Taille des batchs d'apprentissage
        
        # Mémoire d'expérience
        self.memory = ReplayBuffer(capacity=10000)
        
        # Réseaux de neurones (principal et cible)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        # Réseau de neurones pour approximer la fonction Q
        model = Sequential([
            Flatten(input_shape=(self.state_size,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        # Copier les poids du modèle principal vers le modèle cible
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        # Stocker l'expérience dans la mémoire
        self.memory.add(state, action, reward, next_state, done)
    
    def act(self, state, training=True):
        # Stratégie epsilon-greedy pour l'exploration/exploitation
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Prédire les valeurs Q pour toutes les actions
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        # Apprentissage par expérience replay
        if self.memory.size() < self.batch_size:
            return
        
        # Échantillonner un batch de la mémoire
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Calculer les valeurs Q cibles
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Entraîner le modèle
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Mettre à jour epsilon pour réduire l'exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, path):
        self.model.load_weights(path)
        self.update_target_model()
    
    def save(self, path):
        self.model.save_weights(path)

# 4. Fonction d'entraînement
def train_dqn(env, agent, episodes=1000, update_target_every=10, max_steps=24):
    """
    Entraîne l'agent DQN sur l'environnement spécifié
    """
    rewards_history = []
    
    for episode in range(episodes):
        # Réinitialiser l'environnement
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Choisir une action
            action = agent.act(state)
            
            # Exécuter l'action
            next_state, reward, done, _ = env.step(action)
            
            # Stocker l'expérience
            agent.remember(state, action, reward, next_state, done)
            
            # Mettre à jour l'état
            state = next_state
            total_reward += reward
            
            # Apprentissage
            agent.replay()
            
            if done:
                break
        
        # Mettre à jour le modèle cible périodiquement
        if episode % update_target_every == 0:
            agent.update_target_model()
        
        rewards_history.append(total_reward)
        
        # Afficher la progression
        if episode % 100 == 0:
            print(f"Episode: {episode}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    return rewards_history

# 5. Visualisation des résultats
def plot_rewards(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Récompenses par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense totale')
    plt.grid(True)
    plt.savefig('../outputs/dqn_rewards.png')
    plt.show()

# 6. Évaluation du modèle
def evaluate_agent(env, agent, episodes=10):
    """
    Évalue les performances de l'agent entraîné
    """
    print("\n--- Évaluation du modèle ---")
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
        
        total_rewards.append(total_reward)
        print(f"Épisode {episode+1}: Récompense = {total_reward:.2f}")
    
    print(f"\nRécompense moyenne: {np.mean(total_rewards):.2f}")
    return total_rewards

# 7. Exécution de l'entraînement et de l'évaluation
if __name__ == "__main__":
    # Paramètres
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f"État: {state_size} dimensions, Actions: {action_size} possibilités")
    
    # Créer l'agent
    agent = DQNAgent(state_size, action_size)
    
    # Entraîner l'agent
    print("\n--- Début de l'entraînement ---")
    rewards = train_dqn(env, agent, episodes=500, update_target_every=5)
    
    # Sauvegarder le modèle
    agent.save(MODEL_PATH)
    print(f"\nModèle sauvegardé dans {MODEL_PATH}")
    
    # Visualiser les résultats
    plot_rewards(rewards)
    
    # Évaluer l'agent
    evaluate_agent(env, agent)

# 8. Visualisation des plannings générés
def visualize_schedule(env, agent):
    """
    Génère et visualise un planning optimisé par l'agent
    """
    state = env.reset()
    done = False
    
    activities = []
    slots = []
    
    for slot in range(24):
        action = agent.act(state, training=False)
        next_state, reward, done, _ = env.step(action)
        
        # Récupérer le nom de l'activité
        activity_name = df[df['ACTIVITY_NAME_ENC'] == action]['ACTIVITY_NAME'].mode()
        if len(activity_name) > 0:
            activity = activity_name.iloc[0]
        else:
            activity = f"Activity {action}"
        
        activities.append(activity)
        slots.append(slot)
        
        state = next_state
        
        if done:
            break
    
    # Créer un DataFrame pour la visualisation
    schedule_df = pd.DataFrame({
        'Heure': slots,
        'Activité': activities
    })
    
    # Visualisation
    plt.figure(figsize=(12, 8))
    plt.barh(schedule_df['Heure'], [1] * len(schedule_df), color='skyblue')
    
    # Ajouter les noms des activités
    for i, (hour, activity) in enumerate(zip(schedule_df['Heure'], schedule_df['Activité'])):
        plt.text(0.5, hour, activity, ha='center', va='center')
    
    plt.yticks(slots, [f"{h}:00" for h in slots])
    plt.xlabel('Activité')
    plt.ylabel('Heure')
    plt.title('Planning journalier optimisé')
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig('../outputs/optimized_schedule.png')
    plt.show()

# Créer le dossier de sortie si nécessaire
os.makedirs("../outputs", exist_ok=True)

# Générer un exemple de planning
print("\n--- Génération d'un planning optimisé ---")
visualize_schedule(env, agent)

# Afficher les informations sur la taille de la mémoire d'expérience
print(f"\nTaille de la mémoire d'expérience: {agent.memory.size()} échantillons")
print(f"Taux d'exploration final: {agent.epsilon:.4f}")
errur,:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
File c:\Users\Abdel\anaconda3\Lib\site-packages\pandas\core\indexes\base.py:3805, in Index.get_loc(self, key)
   3804 try:
-> 3805     return self._engine.get_loc(casted_key)
   3806 except KeyError as err:

File index.pyx:167, in pandas._libs.index.IndexEngine.get_loc()

File index.pyx:196, in pandas._libs.index.IndexEngine.get_loc()

File pandas\\_libs\\hashtable_class_helper.pxi:7081, in pandas._libs.hashtable.PyObjectHashTable.get_item()

File pandas\\_libs\\hashtable_class_helper.pxi:7089, in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'ACTIVITY_NAME'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
Cell In[76], line 13
     11 # Entraîner l'agent
     12 print("\n--- Début de l'entraînement ---")
---> 13 rewards = train_dqn(env, agent, episodes=500, update_target_every=5)
     15 # Sauvegarder le modèle
     16 agent.save(MODEL_PATH)

Cell In[73], line 18, in train_dqn(env, agent, episodes, update_target_every, max_steps)
     15 action = agent.act(state)
     17 # Exécuter l'action
---> 18 next_state, reward, done, _ = env.step(action)
     20 # Stocker l'expérience
     21 agent.remember(state, action, reward, next_state, done)

File c:\Users\Abdel\Desktop\Assistant-Personnel-pour-la-Gestion-du-Temps\notebooks\environment_setup.py:121, in step(self, action)
    120 def reset(self):
--> 121     self.current_slot = 0
    122     self.schedule = np.zeros(self.num_slots, dtype=np.int32)
    123     self.day_of_week = random.randint(0, 6)

File c:\Users\Abdel\Desktop\Assistant-Personnel-pour-la-Gestion-du-Temps\notebooks\environment_setup.py:148, in _calculate_reward(self, action, time_slot)
    147 def _calculate_reward(self, action, time_slot):
--> 148     reward = 0.0
    149     time_pref_score = self.activity_patterns.get_time_preference_score(action, time_slot)
    150     reward += time_pref_score * 2.0

File c:\Users\Abdel\Desktop\Assistant-Personnel-pour-la-Gestion-du-Temps\notebooks\environment_setup.py:177, in get_activity_name(self, activity_code)
    176 def render(self, mode='human'):
--> 177     if mode == 'human':
    178         print(f"🕒 Slot actuel: {self.current_slot}/{self.num_slots}")
    179         print(f"📅 Jour: {['Dim', 'Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam'][self.day_of_week]}")

File c:\Users\Abdel\anaconda3\Lib\site-packages\pandas\core\frame.py:4102, in DataFrame.__getitem__(self, key)
   4100 if self.columns.nlevels > 1:
   4101     return self._getitem_multilevel(key)
-> 4102 indexer = self.columns.get_loc(key)
   4103 if is_integer(indexer):
   4104     indexer = [indexer]

File c:\Users\Abdel\anaconda3\Lib\site-packages\pandas\core\indexes\base.py:3812, in Index.get_loc(self, key)
   3807     if isinstance(casted_key, slice) or (
   3808         isinstance(casted_key, abc.Iterable)
   3809         and any(isinstance(x, slice) for x in casted_key)
   3810     ):
   3811         raise InvalidIndexError(key)
-> 3812     raise KeyError(key) from err
   3813 except TypeError:
   3814     # If we have a listlike key, _check_indexing_error will raise
   3815     #  InvalidIndexError. Otherwise we fall through and re-raise
   3816     #  the TypeError.
   3817     self._check_indexing_error(key)

KeyError: 'ACTIVITY_NAME'#   O p t i m i s a t i o n - d e - p l a n n i n g s - a v e c - D Q N  
 