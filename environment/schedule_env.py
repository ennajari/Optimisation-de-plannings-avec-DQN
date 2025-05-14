"""
Environnement de planification pour l'Assistant Personnel de Gestion du Temps
Utilise un format compatible avec OpenAI Gym pour l'apprentissage par renforcement
"""
import numpy as np
import pandas as pd
import gym
from gym import spaces
from datetime import datetime, timedelta
import os

class ScheduleEnv(gym.Env):
    """
    Environnement de simulation pour la planification d'agenda personnel
    utilisant le format OpenAI Gym pour l'apprentissage par renforcement.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data_path=None, user_id=None, n_time_slots=24, max_activities=10):
        super(ScheduleEnv, self).__init__()
        
        # Paramètres de l'environnement
        self.n_time_slots = n_time_slots  # 24 créneaux horaires (un par heure)
        self.max_activities = max_activities  # Nombre maximum d'activités à planifier
        self.days_of_week = 7  # Jour de la semaine (1-7)
        self.activity_types = None  # Sera défini lors du chargement des données
        
        # Charger et préprocesser les données
        self.load_user_data(data_path, user_id)
        
        # Définir l'espace d'actions : 
        # Pour chaque activité, on peut la planifier à n'importe quel créneau horaire
        # Actions: (activité_id, créneau_horaire)
        self.action_space = spaces.MultiDiscrete([
            len(self.activity_types), 
            self.n_time_slots
        ])
        
        # Définir l'espace d'observation
        # État: matrice [n_time_slots x n_activity_types] + vecteur jour de la semaine (one-hot)
        self.observation_space = spaces.Dict({
            'schedule': spaces.Box(
                low=0, 
                high=1, 
                shape=(self.n_time_slots, len(self.activity_types)), 
                dtype=np.float32
            ),
            'day_of_week': spaces.Box(
                low=0, 
                high=1, 
                shape=(self.days_of_week,), 
                dtype=np.float32
            ),
            'time_remaining': spaces.Box(
                low=0, 
                high=self.n_time_slots, 
                shape=(1,), 
                dtype=np.float32
            )
        })
        
        # État actuel de l'environnement
        self.current_schedule = None
        self.current_day = None
        self.available_time = None
        self.scheduled_activities = None
        
        # Historique des activités pour l'apprentissage des préférences utilisateur
        self.activity_history = None
        
        # Reset pour initialiser l'environnement
        self.reset()
    
    def load_user_data(self, data_path, user_id=None):
        """
        Charge les données d'activités d'un utilisateur spécifique ou d'un ensemble d'utilisateurs.
        Ne conserve que les colonnes essentielles pour le projet.
        """
        if data_path is None:
            # Créer des données synthétiques pour les tests
            self._create_synthetic_data()
            return
        
        try:
            # Charger les données
            data = pd.read_csv(data_path)
            
            # Filtrer uniquement les colonnes essentielles
            essential_cols = ['TUCASEID', 'TUACTIVITY_N', 'TUACTDUR24', 
                             'TUSTARTTIM', 'ACTIVITY_NAME', 'TUDIARYDAY']
            data = data[essential_cols]
            
            # Filtrer pour un utilisateur spécifique si fourni
            if user_id:
                data = data[data['TUCASEID'] == user_id]
            
            # Extraire les types d'activités uniques
            self.activity_types = data['ACTIVITY_NAME'].unique()
            
            # Convertir les heures de début en format numérique (minutes depuis minuit)
            data['start_time_minutes'] = data['TUSTARTTIM'].apply(self._convert_time_to_minutes)
            
            # Calculer les heures de fin
            data['end_time_minutes'] = data['start_time_minutes'] + data['TUACTDUR24']
            
            # Stocker les données traitées
            self.user_data = data
            
            # Calculer les statistiques des activités pour l'apprentissage des préférences
            self._compute_activity_statistics()
            
        except Exception as e:
            print(f"Erreur lors du chargement des données: {str(e)}")
            # Créer des données synthétiques en cas d'erreur
            self._create_synthetic_data()
    
    def _convert_time_to_minutes(self, time_str):
        """Convertit une chaîne de temps HH:MM en minutes depuis minuit."""
        try:
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        except:
            # Imputer la moyenne des heures pour l'activité si disponible
            return 8 * 60  # Fallback: 8h du matin
    
    def _create_synthetic_data(self):
        """Crée des données synthétiques pour les tests."""
        # Définir des types d'activités de base
        self.activity_types = np.array([
            'Travail', 'Repas', 'Transport', 'Loisirs', 'Sport', 
            'Sommeil', 'Tâches ménagères', 'Courses', 'Socialisation', 'Apprentissage'
        ])
        
        # Créer un dataframe avec des données synthétiques
        n_samples = 100
        synthetic_data = {
            'TUCASEID': np.ones(n_samples),
            'TUACTIVITY_N': np.arange(n_samples),
            'ACTIVITY_NAME': np.random.choice(self.activity_types, n_samples),
            'TUACTDUR24': np.random.randint(15, 240, n_samples),  # 15min à 4h
            'start_time_minutes': np.random.randint(0, 24*60, n_samples),
            'TUDIARYDAY': np.random.randint(1, 8, n_samples)  # 1-7 (lundi-dimanche)
        }
        
        # Calculer les heures de fin
        synthetic_data['end_time_minutes'] = synthetic_data['start_time_minutes'] + synthetic_data['TUACTDUR24']
        
        # Convertir en dataframe
        self.user_data = pd.DataFrame(synthetic_data)
        
        # Calculer les statistiques des activités
        self._compute_activity_statistics()
    
    def _compute_activity_statistics(self):
        """Calcule les statistiques des activités pour l'apprentissage des préférences."""
        # Grouper par type d'activité et jour de la semaine
        grouped = self.user_data.groupby(['ACTIVITY_NAME', 'TUDIARYDAY'])
        
        # Calculer la durée moyenne par activité et par jour
        self.avg_duration = grouped['TUACTDUR24'].mean().reset_index()
        
        # Calculer l'heure de début moyenne par activité et par jour
        self.avg_start_time = grouped['start_time_minutes'].mean().reset_index()
        
        # Calculer la fréquence des activités par jour
        self.activity_frequency = grouped.size().reset_index(name='frequency')
    
    def reset(self):
        """Réinitialise l'environnement et retourne l'état initial."""
        # Choisir un jour aléatoire de la semaine (1-7)
        self.current_day = np.random.randint(1, 8)
        
        # Initialiser un agenda vide
        self.current_schedule = np.zeros((self.n_time_slots, len(self.activity_types)))
        
        # Initialiser le temps disponible (24 heures = 24 créneaux)
        self.available_time = self.n_time_slots
        
        # Initialiser la liste des activités planifiées
        self.scheduled_activities = []
        
        # Créer un vecteur one-hot pour le jour de la semaine
        day_of_week = np.zeros(self.days_of_week)
        day_of_week[self.current_day - 1] = 1  # -1 car l'indexation commence à 0
        
        # Retourner l'état initial
        return {
            'schedule': self.current_schedule.astype(np.float32),
            'day_of_week': day_of_week.astype(np.float32),
            'time_remaining': np.array([self.available_time], dtype=np.float32)
        }
    
    def step(self, action):
        """
        Exécute une action dans l'environnement et retourne le nouvel état,
        la récompense, le statut de fin et des informations supplémentaires.
        
        Action: (activité_id, créneau_horaire)
        """
        activity_id, time_slot = action
        activity_name = self.activity_types[activity_id]
        
        # Vérifier si le créneau horaire est disponible
        if self.current_schedule[time_slot].sum() > 0:
            reward = -1.0  # Pénalité pour conflit horaire
            done = False
            info = {'status': 'conflict', 'activity': activity_name, 'time_slot': time_slot}
        else:
            # Planifier l'activité
            self.current_schedule[time_slot, activity_id] = 1
            self.available_time -= 1
            self.scheduled_activities.append((activity_name, time_slot))
            
            # Calculer la récompense basée sur les préférences utilisateur
            reward = self._compute_reward(activity_name, time_slot)
            
            # Vérifier si toutes les activités ont été planifiées ou si le temps est écoulé
            done = len(self.scheduled_activities) >= self.max_activities or self.available_time <= 0
            
            info = {'status': 'scheduled', 'activity': activity_name, 'time_slot': time_slot}
        
        # Construire le nouvel état
        day_of_week = np.zeros(self.days_of_week)
        day_of_week[self.current_day - 1] = 1
        
        state = {
            'schedule': self.current_schedule.astype(np.float32),
            'day_of_week': day_of_week.astype(np.float32),
            'time_remaining': np.array([self.available_time], dtype=np.float32)
        }
        
        return state, reward, done, info
    
    def _compute_reward(self, activity_name, time_slot):
        """
        Calcule la récompense pour une activité planifiée à un créneau horaire spécifique,
        basé sur les préférences historiques de l'utilisateur.
        """
        # Convertir le créneau horaire en minutes (1 créneau = 1 heure = 60 minutes)
        scheduled_time = time_slot * 60
        
        # Trouver les statistiques pour cette activité et ce jour de la semaine
        act_stats = self.avg_start_time[
            (self.avg_start_time['ACTIVITY_NAME'] == activity_name) & 
            (self.avg_start_time['TUDIARYDAY'] == self.current_day)
        ]
        
        if not act_stats.empty:
            # Calculer la différence entre l'heure planifiée et l'heure préférée (en minutes)
            preferred_time = act_stats['start_time_minutes'].values[0]
            time_diff = abs(scheduled_time - preferred_time)
            
            # La récompense diminue avec l'écart par rapport à l'heure préférée
            # Normaliser pour que la récompense soit entre 0 et 1
            time_reward = max(0, 1 - (time_diff / (12 * 60)))  # max 12h de différence
        else:
            # Si nous n'avons pas d'information sur cette activité pour ce jour
            time_reward = 0.5  # Récompense neutre
        
        # Bonus pour une planification cohérente (activités similaires regroupées)
        coherence_reward = 0
        if len(self.scheduled_activities) > 0:
            for prev_activity, prev_slot in self.scheduled_activities:
                # Récompense pour les activités similaires regroupées
                if prev_activity == activity_name and abs(prev_slot - time_slot) <= 1:
                    coherence_reward += 0.2
                # Pénalité pour les activités qui devraient être espacées
                elif prev_activity == activity_name and abs(prev_slot - time_slot) < 3:
                    coherence_reward -= 0.1
        
        # La récompense finale est une combinaison de récompenses basées sur le temps et la cohérence
        reward = time_reward + coherence_reward
        
        return reward
    
    def render(self, mode='human'):
        """Affiche l'état actuel de l'environnement pour le débogage."""
        if mode != 'human':
            return
        
        print("\n===== ÉTAT ACTUEL DE L'AGENDA =====")
        print(f"Jour de la semaine: {self.current_day}")
        print(f"Temps restant: {self.available_time} créneaux")
        print("\nActivités planifiées:")
        
        for i, (activity, time_slot) in enumerate(self.scheduled_activities):
            print(f"{i+1}. {activity} à {time_slot}:00")
        
        print("\nGrille horaire:")
        for slot in range(self.n_time_slots):
            activities = [self.activity_types[i] for i in range(len(self.activity_types)) 
                         if self.current_schedule[slot, i] > 0]
            print(f"{slot}:00 - {activities if activities else 'Libre'}")
        
        print("====================================\n")
    
    def close(self):
        """Libère les ressources."""
        pass

    def get_optimal_schedule(self):
        """
        Génère un emploi du temps optimal basé sur les préférences utilisateur.
        Utilisé pour comparer les performances du modèle DQN.
        """
        # Créer un agenda vide
        optimal_schedule = np.zeros((self.n_time_slots, len(self.activity_types)))
        
        # Filtrer les activités pour le jour actuel
        day_activities = self.avg_start_time[self.avg_start_time['TUDIARYDAY'] == self.current_day]
        
        # Trier par fréquence (priorité aux activités les plus fréquentes)
        day_activities = day_activities.merge(self.activity_frequency, 
                                          on=['ACTIVITY_NAME', 'TUDIARYDAY'])
        day_activities = day_activities.sort_values('frequency', ascending=False)
        
        # Planifier les activités à leur heure préférée
        for _, row in day_activities.iterrows():
            activity_name = row['ACTIVITY_NAME']
            preferred_time_minutes = row['start_time_minutes']
            
            # Convertir en créneau horaire (arrondi à l'heure la plus proche)
            preferred_slot = int(round(preferred_time_minutes / 60)) % self.n_time_slots
            
            # Vérifier si le créneau est disponible
            if optimal_schedule[preferred_slot].sum() == 0:
                # Trouver l'index de l'activité
                activity_idx = np.where(self.activity_types == activity_name)[0][0]
                
                # Planifier l'activité
                optimal_schedule[preferred_slot, activity_idx] = 1
            else:
                # Chercher le créneau disponible le plus proche
                for offset in range(1, self.n_time_slots // 2):
                    # Essayer le créneau avant
                    before_slot = (preferred_slot - offset) % self.n_time_slots
                    if optimal_schedule[before_slot].sum() == 0:
                        activity_idx = np.where(self.activity_types == activity_name)[0][0]
                        optimal_schedule[before_slot, activity_idx] = 1
                        break
                    
                    # Essayer le créneau après
                    after_slot = (preferred_slot + offset) % self.n_time_slots
                    if optimal_schedule[after_slot].sum() == 0:
                        activity_idx = np.where(self.activity_types == activity_name)[0][0]
                        optimal_schedule[after_slot, activity_idx] = 1
                        break
        
        return optimal_schedule