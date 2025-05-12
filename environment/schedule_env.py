import numpy as np
import pandas as pd
from gym import Env, spaces

class ScheduleEnv(Env):
    def __init__(self, df):
        super(ScheduleEnv, self).__init__()
        
        self.df = df
        self.activities = df['ACTIVITY_CODE'].unique()
        self.num_activities = len(self.activities)
        self.current_step = 0
        self.max_steps = 24  # Simule une journée de 24 heures
        
        # Espace d'action = choisir une activité
        self.action_space = spaces.Discrete(self.num_activities)
        
        # Espace d'observation = [heure, jour de la semaine, week-end ou non]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([24, 6, 1]),
            dtype=np.float32
        )
        
        self.current_hour = 8  # Début de la journée à 8h
        self.day_of_week = np.random.randint(0, 7)
        self.is_weekend = int(self.day_of_week >= 5)
        self.schedule = []

    def reset(self):
        """Réinitialise l'environnement au début d'une journée."""
        self.current_step = 0
        self.current_hour = 8
        self.day_of_week = np.random.randint(0, 7)
        self.is_weekend = int(self.day_of_week >= 5)
        self.schedule = []

        return self._get_obs()
    
    def _get_obs(self):
        """Retourne l'état actuel sous forme de vecteur."""
        return np.array([self.current_hour, self.day_of_week, self.is_weekend], dtype=np.float32)

    def step(self, action):
        """Exécute une action choisie (activité) et retourne l'état suivant, la récompense, et si la journée est terminée."""
        if self.current_step >= self.max_steps:
            done = True
            return self._get_obs(), 0, done, {}

        # Filtrer les données de l'activité sélectionnée
        activity_data = self.df[self.df['ACTIVITY_CODE'] == action]
        
        if activity_data.empty:
            duration = 1  # activité inconnue, faible durée
        else:
            duration = activity_data['TUACTDUR24'].mean() / 60  # conversion en heures
            duration = max(0.25, min(duration, 3.0))  # clamp entre 15min et 3h

        self.current_hour += duration
        self.current_step += 1
        self.schedule.append((self.current_hour, action))

        # Définir si la journée est terminée
        done = self.current_hour >= 24

        # Récompense basée sur la variété et la progression de la journée
        reward = 1.0 - abs(self.current_hour - 12) / 12.0  # meilleure répartition sur la journée
        reward -= 0.05 * self.schedule.count((self.current_hour, action))  # pénaliser la répétition

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        """Affiche le planning courant."""
        print(f"📅 Jour: {self.day_of_week} | Planning:")
        for i, (hour, act) in enumerate(self.schedule):
            print(f" - {hour:.2f}h : Activité {act}")
