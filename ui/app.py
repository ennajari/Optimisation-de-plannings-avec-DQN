"""
Application Streamlit pour l'Assistant Personnel de Gestion du Temps
Utilise un modèle DQN pour optimiser les plannings personnels des utilisateurs
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import os
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Vérifier la version de TensorFlow
print(f"TensorFlow version: {tf.__version__}")

# Ajouter le répertoire parent au chemin pour importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.schedule_env import ScheduleEnv

# Configuration de la page
st.set_page_config(
    page_title="Assistant Personnel de Gestion du Temps",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chemins de fichiers
MODEL_PATH = '../models/dqn_schedule_model.h5'
DATA_PATH = '../Data/processed/cleaned_data.csv'

# Vérification des fichiers
if not os.path.exists(MODEL_PATH):
    st.error(f"Le fichier modèle n'existe pas : {MODEL_PATH}")
if not os.path.exists(DATA_PATH):
    st.warning(f"Le fichier de données n'existe pas : {DATA_PATH}. Utilisation de données synthétiques.")

DEFAULT_ACTIVITIES = [
    'Travail', 'Repas', 'Transport', 'Loisirs', 'Sport', 
    'Sommeil', 'Tâches ménagères', 'Courses', 'Socialisation', 'Apprentissage'
]
ACTIVITY_COLORS = {
    'Travail': '#FF6B6B',
    'Repas': '#4ECDC4',
    'Transport': '#FFD166',
    'Loisirs': '#6B5B95',
    'Sport': '#88D8B0',
    'Sommeil': '#5D535E',
    'Tâches ménagères': '#F7B801',
    'Courses': '#F18701',
    'Socialisation': '#7BDFF2',
    'Apprentissage': '#B2DBBF'
}
DEFAULT_COLOR = '#CCCCCC'

# Fonction pour charger le modèle
@st.cache_resource
def load_dqn_model():
    """Charge le modèle DQN pré-entraîné"""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return None

# Fonction pour charger les données
@st.cache_data
def load_data():
    """Charge les données prétraitées"""
    try:
        data = pd.read_csv(DATA_PATH)
        return data
    except Exception as e:
        st.warning(f"Erreur lors du chargement des données: {str(e)}")
        st.info("Utilisation de données par défaut")
        return None

# Créer un environnement Gym pour l'optimisation de planning
def create_environment(data=None, user_activities=None):
    """Crée l'environnement de simulation pour l'optimisation de planning"""
    if user_activities is None:
        user_activities = DEFAULT_ACTIVITIES
    
    env = ScheduleEnv(data_path=DATA_PATH if data is not None else None)
    
    if user_activities:
        env.activity_types = np.array(user_activities)
    
    return env

# Fonction pour générer un planning optimisé avec DQN
def generate_optimized_schedule(env, model, user_constraints=None, day_of_week=None):
    """
    Génère un planning optimisé en utilisant le modèle DQN
    
    Args:
        env: Environnement de simulation
        model: Modèle DQN pré-entraîné
        user_constraints: Dictionnaire de contraintes utilisateur
        day_of_week: Jour de la semaine spécifique (1-7, lundi-dimanche)
        
    Returns:
        schedule: Planning généré (matrice)
        activities: Liste des activités planifiées (avec heures et durées)
        rewards: Récompenses obtenues
    """
    state = env.reset()
    
    if day_of_week is not None:
        env.current_day = day_of_week
        day_of_week_vector = np.zeros(env.days_of_week)
        day_of_week_vector[env.current_day - 1] = 1
        state['day_of_week'] = day_of_week_vector
    
    if user_constraints:
        for activity, time_slots in user_constraints.items():
            if activity in env.activity_types:
                activity_idx = np.where(env.activity_types == activity)[0][0]
                for slot in time_slots:
                    if 0 <= slot < env.n_time_slots:
                        env.current_schedule[slot, activity_idx] = 1
                        env.available_time -= 1
    
    done = False
    total_reward = 0
    activities_planned = []
    
    while not done and env.available_time > 0:
        schedule_flat = state['schedule'].flatten()
        day_of_week = state['day_of_week']
        time_remaining = state['time_remaining']
        state_tensor = np.concatenate([schedule_flat, day_of_week, time_remaining])
        state_tensor = np.expand_dims(state_tensor, 0)
        
        act_values = model.predict(state_tensor, verbose=0)[0]
        
        n_activities = len(env.activity_types)
        action_idx = np.argmax(act_values)
        activity_id = action_idx // env.n_time_slots
        time_slot = action_idx % env.n_time_slots
        action = np.array([activity_id, time_slot])
        
        while env.current_schedule[action[1]].sum() > 0:
            act_values[action_idx] = -np.inf
            action_idx = np.argmax(act_values)
            activity_id = action_idx // env.n_time_slots
            time_slot = action_idx % env.n_time_slots
            action = np.array([activity_id, time_slot])
            
            if np.all(act_values == -np.inf):
                done = True
                break
        
        if not done:
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            total_reward += reward
            
            activity_name = env.activity_types[action[0]]
            duration_stats = env.avg_duration[
                (env.avg_duration['ACTIVITY_NAME'] == activity_name) & 
                (env.avg_duration['TUDIARYDAY'] == env.current_day)
            ]
            duration = float(duration_stats['TUACTDUR24'].values[0]) if not duration_stats.empty else 60.0
            duration = min(duration, 240.0)  # Cap duration at 4 hours
            
            time_slot = action[1]
            activities_planned.append((activity_name, time_slot, reward, duration))
    
    return env.current_schedule, activities_planned, total_reward

# Fonction pour visualiser le planning avec Plotly
def visualize_schedule_plotly(schedule, activities, activity_types, reference_date="2025-05-12"):
    """Crée une visualisation interactive du planning avec Plotly"""
    from datetime import datetime, timedelta
    
    schedule_data = []
    
    for activity_name, time_slot, reward, duration in activities:
        # Calculate start time as a datetime
        start_hour = time_slot
        start_time = datetime.strptime(f"{reference_date} {start_hour:02d}:00", "%Y-%m-%d %H:%M")
        
        # Calculate end time
        end_time_minutes = time_slot * 60 + duration
        days_offset = int(end_time_minutes // 1440)  # Number of days to add if exceeding 24 hours
        end_time_minutes = end_time_minutes % 1440  # Minutes within the day
        end_hour = int(end_time_minutes // 60)
        end_minute = int(end_time_minutes % 60)
        
        # Parse end time, adjusting date if necessary
        end_date = datetime.strptime(reference_date, "%Y-%m-%d") + timedelta(days=days_offset)
        end_time = datetime.strptime(
            f"{end_date.strftime('%Y-%m-%d')} {end_hour:02d}:{end_minute:02d}", 
            "%Y-%m-%d %H:%M"
        )
        
        schedule_data.append({
            'Heure_Début': start_time,
            'Heure_Fin': end_time,
            'Activité': activity_name,
            'Valeur': 1,
            'Couleur': ACTIVITY_COLORS.get(activity_name, DEFAULT_COLOR)
        })
    
    if not schedule_data:
        st.warning("Aucune activité n'a pu être planifiée.")
        return go.Figure()
    
    schedule_df = pd.DataFrame(schedule_data)
    
    fig = px.timeline(
        schedule_df, 
        x_start="Heure_Début", 
        x_end="Heure_Fin", 
        y="Activité", 
        color="Activité",
        color_discrete_map=ACTIVITY_COLORS,
        title="Planning Journalier Optimisé"
    )
    
    fig.update_layout(
        xaxis_title="Heure de la journée",
        yaxis_title="Activité",
        height=600,
        showlegend=True,
        xaxis=dict(
            tickformat="%H:%M",  # Display only time (HH:MM)
            tickangle=45
        )
    )
    
    return fig

# Fonction pour créer un calendrier hebdomadaire
def create_weekly_calendar(daily_schedules, activity_types, all_activities_planned):
    """Crée un calendrier hebdomadaire à partir des plannings quotidiens"""
    from datetime import datetime, timedelta
    
    days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    day_to_date = {
        'Lundi': '2025-05-12',
        'Mardi': '2025-05-13',
        'Mercredi': '2025-05-14',
        'Jeudi': '2025-05-15',
        'Vendredi': '2025-05-16',
        'Samedi': '2025-05-17',
        'Dimanche': '2025-05-18'
    }
    calendar_data = []
    
    for act, time_slot, reward, duration, day in all_activities_planned:
        day_name = days[day - 1]
        reference_date = day_to_date[day_name]
        
        start_time = datetime.strptime(f"{reference_date} {time_slot:02d}:00", "%Y-%m-%d %H:%M")
        
        end_time_minutes = time_slot * 60 + duration
        days_offset = int(end_time_minutes // 1440)
        end_time_minutes = end_time_minutes % 1440
        end_hour = int(end_time_minutes // 60)
        end_minute = int(end_time_minutes % 60)
        
        end_date = datetime.strptime(reference_date, "%Y-%m-%d") + timedelta(days=days_offset)
        end_time = datetime.strptime(
            f"{end_date.strftime('%Y-%m-%d')} {end_hour:02d}:{end_minute:02d}", 
            "%Y-%m-%d %H:%M"
        )
        
        calendar_data.append({
            'Jour': day_name,
            'Heure_Début': start_time,
            'Heure_Fin': end_time,
            'Activité': act,
            'Couleur': ACTIVITY_COLORS.get(act, DEFAULT_COLOR)
        })
    
    if not calendar_data:
        st.warning("Aucune activité n'a pu être planifiée pour la semaine.")
        return go.Figure()
    
    calendar_df = pd.DataFrame(calendar_data)
    
    fig = px.timeline(
        calendar_df, 
        x_start="Heure_Début", 
        x_end="Heure_Fin", 
        y="Jour", 
        color="Activité",
        color_discrete_map=ACTIVITY_COLORS,
        title="Planning Hebdomadaire Optimisé"
    )
    
    fig.update_layout(
        xaxis_title="Heure de la journée",
        yaxis_title="Jour de la semaine",
        height=600,
        showlegend=True,
        xaxis=dict(
            tickformat="%H:%M",
            tickangle=45
        )
    )
    
    return fig

# Fonction pour calculer des statistiques sur le planning
def calculate_schedule_stats(activities):
    """Calcule des statistiques sur le planning généré"""
    if not activities:
        return {}
    
    activity_counts = {}
    for activity, time_slot, reward, duration in activities:
        if activity in activity_counts:
            activity_counts[activity] += 1
        else:
            activity_counts[activity] = 1
    
    rewards_by_hour = {}
    for activity, time_slot, reward, duration in activities:
        if time_slot in rewards_by_hour:
            rewards_by_hour[time_slot] += reward
        else:
            rewards_by_hour[time_slot] = reward
    
    most_productive_hour = max(rewards_by_hour, key=rewards_by_hour.get) if rewards_by_hour else None
    
    return {
        'activity_counts': activity_counts,
        'rewards_by_hour': rewards_by_hour,
        'most_productive_hour': most_productive_hour
    }

# Interface utilisateur avec Streamlit
def main():
    st.title("📅 Assistant Personnel de Gestion du Temps")
    st.subheader("Optimisez votre planning quotidien avec l'intelligence artificielle")
    
    model = load_dqn_model()
    data = load_data()
    
    if model is None:
        st.error("Le modèle DQN n'a pas pu être chargé. Veuillez vérifier le fichier modèle.")
        return
    
    st.sidebar.title("Paramètres")
    
    planning_type = st.sidebar.radio(
        "Type de planning",
        ["📆 Planning quotidien", "🗓️ Planning hebdomadaire"]
    )
    
    day_options = {
        "Lundi": 1, "Mardi": 2, "Mercredi": 3, "Jeudi": 4, 
        "Vendredi": 5, "Samedi": 6, "Dimanche": 7
    }
    
    if planning_type == "📆 Planning quotidien":
        selected_day_name = st.sidebar.selectbox(
            "Jour de la semaine",
            list(day_options.keys())
        )
        selected_day = day_options[selected_day_name]
    else:
        selected_day = None
    
    st.sidebar.subheader("Activités personnalisées")
    
    use_default_activities = st.sidebar.checkbox("Utiliser les activités par défaut", value=True)
    
    if use_default_activities:
        activities = DEFAULT_ACTIVITIES
    else:
        custom_activities_input = st.sidebar.text_area(
            "Entrez vos activités (une par ligne)",
            value="\n".join(DEFAULT_ACTIVITIES)
        )
        activities = [act.strip() for act in custom_activities_input.split("\n") if act.strip()]
    
    st.sidebar.subheader("Liste des activités")
    for i, activity in enumerate(activities):
        color = ACTIVITY_COLORS.get(activity, DEFAULT_COLOR)
        st.sidebar.markdown(
            f"<div style='background-color:{color}; padding:5px; border-radius:5px; margin:2px 0;'>"
            f"{i+1}. {activity}</div>",
            unsafe_allow_html=True
        )
    
    st.sidebar.subheader("Contraintes (optionnel)")
    add_constraints = st.sidebar.checkbox("Ajouter des contraintes horaires")
    
    user_constraints = {}
    if add_constraints:
        constraint_activity = st.sidebar.selectbox(
            "Activité",
            activities
        )
        
        constraint_time = st.sidebar.multiselect(
            "Horaires (obligatoires)",
            [f"{h:02d}:00" for h in range(24)]
        )
        
        if st.sidebar.button("Ajouter cette contrainte"):
            time_slots = [int(t.split(":")[0]) for t in constraint_time]
            if constraint_activity in user_constraints:
                user_constraints[constraint_activity].extend(time_slots)
            else:
                user_constraints[constraint_activity] = time_slots
    
    if user_constraints:
        st.sidebar.subheader("Contraintes ajoutées")
        for activity, slots in user_constraints.items():
            st.sidebar.write(f"{activity}: {', '.join([f'{s:02d}:00' for s in slots])}")
    
    if planning_type == "📆 Planning quotidien":
        generate_button = st.sidebar.button("Générer planning quotidien")
    else:
        generate_button = st.sidebar.button("Générer planning hebdomadaire")
    
    if generate_button:
        with st.spinner("Génération du planning en cours..."):
            env = create_environment(data, activities)
            
            if planning_type == "📆 Planning quotidien":
                schedule, activities_planned, total_reward = generate_optimized_schedule(
                    env, model, user_constraints, selected_day
                )
                
                st.subheader(f"Planning optimisé pour {selected_day_name}")
                
                # Map selected day to a date (week of 2025-05-12)
                day_to_date = {
                    "Lundi": "2025-05-12",
                    "Mardi": "2025-05-13",
                    "Mercredi": "2025-05-14",
                    "Jeudi": "2025-05-15",
                    "Vendredi": "2025-05-16",
                    "Samedi": "2025-05-17",
                    "Dimanche": "2025-05-18"
                }
                reference_date = day_to_date[selected_day_name]
                
                fig = visualize_schedule_plotly(schedule, activities_planned, env.activity_types, reference_date)
                st.plotly_chart(fig, use_container_width=True)
                
                if activities_planned:
                    st.subheader("Détail des activités")
                    
                    activities_df = pd.DataFrame([
                        (act, f"{slot:02d}:00", f"{reward:.2f}", f"{duration:.0f} min") 
                        for act, slot, reward, duration in activities_planned
                    ], columns=["Activité", "Heure", "Score", "Durée"])
                    
                    st.dataframe(activities_df.sort_values(by="Heure"), use_container_width=True)
                    
                    stats = calculate_schedule_stats(activities_planned)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Distribution des activités")
                        if stats['activity_counts']:
                            fig = px.pie(
                                values=list(stats['activity_counts'].values()),
                                names=list(stats['activity_counts'].keys()),
                                title="Distribution des activités",
                                color=list(stats['activity_counts'].keys()),
                                color_discrete_map=ACTIVITY_COLORS
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Productivité par heure")
                        if stats['rewards_by_hour']:
                            hours = [f"{h:02d}:00" for h in stats['rewards_by_hour'].keys()]
                            rewards = list(stats['rewards_by_hour'].values())
                            
                            fig = px.bar(
                                x=hours,
                                y=rewards,
                                title="Score de productivité par heure",
                                labels={'x': 'Heure', 'y': 'Score de productivité'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if stats['most_productive_hour'] is not None:
                        st.info(f"✨ Votre heure la plus productive est {stats['most_productive_hour']:02d}:00")
                
                st.subheader("Évaluation du planning")
                st.write(f"Score total du planning: {total_reward:.2f}")
                
                quality = min(max(total_reward / 10, 0), 1)
                
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=quality * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Qualité du planning"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 33], 'color': "red"},
                            {'range': [33, 66], 'color': "yellow"},
                            {'range': [66, 100], 'color': "green"}
                        ]
                    }
                ))
                
                st.plotly_chart(gauge, use_container_width=True)
                
                st.subheader("Recommandations")
                
                if quality < 0.4:
                    st.warning("Votre planning pourrait être amélioré. Essayez d'ajouter plus de contraintes ou de modifier vos activités.")
                elif quality < 0.7:
                    st.info("Votre planning est bon, mais il pourrait encore être optimisé.")
                else:
                    st.success("Excellent planning ! Votre journée est optimisée pour une productivité maximale.")
            
            else:
                st.subheader("Planning hebdomadaire optimisé")
                
                daily_schedules = []
                all_activities_planned = []
                total_weekly_reward = 0
                
                for day in range(1, 8):
                    day_schedule, day_activities, day_reward = generate_optimized_schedule(
                        env, model, user_constraints, day
                    )
                    daily_schedules.append(day_schedule)
                    all_activities_planned.extend([(act, slot, reward, duration, day) for act, slot, reward, duration in day_activities])
                    total_weekly_reward += day_reward
                
                fig = create_weekly_calendar(daily_schedules, env.activity_types, all_activities_planned)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Statistiques hebdomadaires")
                
                activities_by_day = {}
                for day_idx in range(7):
                    day_name = list(day_options.keys())[day_idx]
                    day_activities = [act for act, _, _, _, day in all_activities_planned if day == day_idx + 1]
                    activities_by_day[day_name] = len(day_activities)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Activités par jour")
                    fig = px.bar(
                        x=list(activities_by_day.keys()),
                        y=list(activities_by_day.values()),
                        title="Nombre d'activités par jour",
                        labels={'x': 'Jour', 'y': 'Nombre d\'activités'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Score hebdomadaire")
                    st.write(f"Score total de la semaine: {total_weekly_reward:.2f}")
                    
                    quality = min(max(total_weekly_reward / 70, 0), 1)
                    
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=quality * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Qualité du planning hebdomadaire"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 33], 'color': "red"},
                                {'range': [33, 66], 'color': "yellow"},
                                {'range': [66, 100], 'color': "green"}
                            ]
                        }
                    ))
                    
                    st.plotly_chart(gauge, use_container_width=True)
                
                activity_counts = {}
                for act, _, _, _, _ in all_activities_planned:
                    if act in activity_counts:
                        activity_counts[act] += 1
                    else:
                        activity_counts[act] = 1
                
                st.subheader("Répartition des activités sur la semaine")
                fig = px.pie(
                    values=list(activity_counts.values()),
                    names=list(activity_counts.keys()),
                    title="Distribution des activités sur la semaine",
                    color=list(activity_counts.keys()),
                    color_discrete_map=ACTIVITY_COLORS
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Exporter le planning")
                
                export_data = []
                days = list(day_options.keys())
                
                for day_idx, day_name in enumerate(days):
                    day_activities = [(act, slot, duration) for act, slot, _, duration, day in all_activities_planned if day == day_idx + 1]
                    for activity, time_slot, duration in day_activities:
                        export_data.append({
                            'Jour': day_name,
                            'Heure_Début': f"{time_slot:02d}:00",
                            'Durée': f"{duration:.0f} min",
                            'Activité': activity
                        })
                
                export_df = pd.DataFrame(export_data)
                
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Télécharger le planning (CSV)",
                    data=csv,
                    file_name=f"planning_hebdomadaire_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    else:
        st.write("""
        Bienvenue dans votre Assistant Personnel de Gestion du Temps ! Cette application utilise l'intelligence artificielle
        pour vous aider à optimiser votre emploi du temps quotidien ou hebdomadaire.
        
        ### Comment ça marche ?
        
        Notre application utilise un algorithme d'apprentissage par renforcement appelé **Deep Q-Network (DQN)** pour analyser vos habitudes
        et préférences, puis générer un planning optimisé qui maximise votre productivité et votre satisfaction.
        
        ###Fonctionnalités
        
        - **Planning quotidien**: Obtenez un emploi du temps optimisé pour une journée spécifique
        - **Planning hebdomadaire**: Visualisez un calendrier complet pour toute la semaine
        - **Activités personnalisées**: Définissez vos propres types d'activités
        - **Contraintes horaires**: Ajoutez des activités obligatoires à des heures précises
        - **Statistiques**: Analysez la qualité de votre planning et identifiez vos heures les plus productives
        
        ###Commencer
        
        Utilisez les options dans la barre latérale pour personnaliser votre planning, puis cliquez sur "Générer planning" pour voir les résultats.
        """)
        
        st.image("../logo/4406306.png", 
                 caption="Optimisez votre temps avec l'IA")

if __name__ == "__main__":
    main()