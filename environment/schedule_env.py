import numpy as np
import pandas as pd
from gym import Env, spaces

class ScheduleEnv(Env):
    def __init__(self, df):
        super(ScheduleEnv, self).__init__()
        
        self.df = df
        self.current_step = 0
        self.max_steps = 24  # Planification horaire sur 24h
        
        # Définir l'espace d'action (toutes les activités possibles)
        self.action_space = spaces.Discrete(len(df['ACTIVITY_CODE'].unique()))
        
        # Définir l'espace d'observation
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([24, 6, 1]),  # [heure, jour, weekend]
            dtype=np.float32
        )
        
    def reset(self):
        """Réinitialiser l'environnement"""
        self.current_step = 0
        current_time = 8 * 60  # Commence à 8h du matin (en minutes)
        day_of_week = np.random.randint(0, 6)  # Jour aléatoire
        is_weekend = 1 if day_of_week in [5, 6] else 0
        
        self.state = np.array([
            current_time / 60,  # Convertir en heures
            day_of_week,
            is_weekend
        ])
        
        return self.state
    
    def step(self, action):
        """Exécuter une action"""
        # Récupérer la durée moyenne de l'activité choisie
        activity_duration = self.df[self.df['ACTIVITY_CODE'] == action]['TUACTDUR24'].mean()
        
        # Calculer la récompense (à personnaliser)
        reward = self._calculate_reward(action, activity_duration)
        
        # Mettre à jour l'état
        self.current_step += 1
        new_time = (self.state[0] * 60 + activity_duration) / 60  # Avancer le temps
        
        done = (new_time >= 24) or (self.current_step >= self.max_steps)
        
        self.state = np.array([
            new_time,
            self.state[1],  # Même jour
            self.state[2]   # Même weekend
        ])
        
        return self.state, reward, done, {}
    
    def _calculate_reward(self, action, duration):
        """Fonction de récompense personnalisée"""
        # Exemple simple : récompenser les activités productives
        productive_activities = [3, 5, 7]  # À adapter avec vos codes d'activités
        
        if action in productive_activities:
            return min(duration / 60, 1.0)  # Récompense basée sur la durée
        else:
            return -0.1  # Pénalité pour les activités non productives