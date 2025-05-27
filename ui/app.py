"""
Standalone Schedule Optimization Application
A simplified version that works without external model files
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Configuration de la page
st.set_page_config(
    page_title="Assistant Personnel de Gestion du Temps",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
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

# Activity preferences and typical durations (in minutes)
ACTIVITY_PREFERENCES = {
    'Travail': {'preferred_hours': [8, 9, 10, 11, 13, 14, 15, 16], 'duration': 180, 'priority': 9},
    'Repas': {'preferred_hours': [7, 12, 19], 'duration': 45, 'priority': 8},
    'Transport': {'preferred_hours': [7, 8, 17, 18], 'duration': 30, 'priority': 6},
    'Loisirs': {'preferred_hours': [18, 19, 20, 21], 'duration': 90, 'priority': 5},
    'Sport': {'preferred_hours': [6, 7, 17, 18, 19], 'duration': 60, 'priority': 7},
    'Sommeil': {'preferred_hours': [22, 23, 0, 1, 2, 3, 4, 5, 6], 'duration': 480, 'priority': 10},
    'Tâches ménagères': {'preferred_hours': [9, 10, 15, 16, 17], 'duration': 60, 'priority': 4},
    'Courses': {'preferred_hours': [9, 10, 11, 15, 16, 17], 'duration': 75, 'priority': 5},
    'Socialisation': {'preferred_hours': [18, 19, 20, 21], 'duration': 120, 'priority': 6},
    'Apprentissage': {'preferred_hours': [9, 10, 11, 20, 21], 'duration': 90, 'priority': 7}
}

class SimpleScheduleOptimizer:
    """A simplified schedule optimizer that mimics DQN behavior"""
    
    def __init__(self, activities, day_of_week=1):
        self.activities = activities
        self.day_of_week = day_of_week
        self.schedule = {}
        self.available_hours = list(range(24))
        
    def calculate_activity_score(self, activity, hour, constraints=None):
        """Calculate a score for placing an activity at a specific hour"""
        if activity not in ACTIVITY_PREFERENCES:
            return random.uniform(0.3, 0.7)
        
        prefs = ACTIVITY_PREFERENCES[activity]
        score = prefs['priority'] / 10
        
        # Boost score if hour is in preferred hours
        if hour in prefs['preferred_hours']:
            score += 0.3
        
        # Weekend adjustments
        if self.day_of_week in [6, 7]:  # Saturday, Sunday
            if activity in ['Travail']:
                score *= 0.3  # Reduce work on weekends
            elif activity in ['Loisirs', 'Sport', 'Socialisation']:
                score *= 1.2  # Boost leisure activities
        
        # Constraint bonus
        if constraints and activity in constraints:
            if hour in constraints[activity]:
                score += 0.5
        
        return max(0, min(1, score + random.uniform(-0.1, 0.1)))
    
    def optimize_schedule(self, constraints=None, max_activities=8):
        """Generate an optimized schedule"""
        schedule_items = []
        used_hours = set()
        
        # Handle constraints first
        if constraints:
            for activity, hours in constraints.items():
                for hour in hours:
                    if hour not in used_hours and activity in self.activities:
                        duration = ACTIVITY_PREFERENCES.get(activity, {}).get('duration', 60)
                        score = self.calculate_activity_score(activity, hour, constraints)
                        schedule_items.append((activity, hour, score, duration))
                        used_hours.add(hour)
        
        # Fill remaining slots
        remaining_activities = [a for a in self.activities if not any(a == item[0] for item in schedule_items)]
        
        for _ in range(max_activities - len(schedule_items)):
            if not remaining_activities:
                break
                
            best_score = -1
            best_combo = None
            
            for activity in remaining_activities:
                for hour in range(24):
                    if hour not in used_hours:
                        score = self.calculate_activity_score(activity, hour, constraints)
                        if score > best_score:
                            best_score = score
                            best_combo = (activity, hour, score)
            
            if best_combo:
                activity, hour, score = best_combo
                duration = ACTIVITY_PREFERENCES.get(activity, {}).get('duration', 60)
                schedule_items.append((activity, hour, score, duration))
                used_hours.add(hour)
                remaining_activities.remove(activity)
        
        return sorted(schedule_items, key=lambda x: x[1])  # Sort by hour

def visualize_schedule_plotly(activities, reference_date="2025-05-12"):
    """Create an interactive schedule visualization with Plotly"""
    if not activities:
        st.warning("Aucune activité n'a pu être planifiée.")
        return go.Figure()
    
    schedule_data = []
    
    for activity_name, time_slot, reward, duration in activities:
        # Calculate start time
        start_time = datetime.strptime(f"{reference_date} {time_slot:02d}:00", "%Y-%m-%d %H:%M")
        
        # Calculate end time
        end_time = start_time + timedelta(minutes=duration)
        
        schedule_data.append({
            'Heure_Début': start_time,
            'Heure_Fin': end_time,
            'Activité': activity_name,
            'Score': reward,
            'Couleur': ACTIVITY_COLORS.get(activity_name, DEFAULT_COLOR)
        })
    
    schedule_df = pd.DataFrame(schedule_data)
    
    fig = px.timeline(
        schedule_df, 
        x_start="Heure_Début", 
        x_end="Heure_Fin", 
        y="Activité", 
        color="Activité",
        color_discrete_map=ACTIVITY_COLORS,
        title="Planning Journalier Optimisé",
        hover_data=["Score"]
    )
    
    fig.update_layout(
        xaxis_title="Heure de la journée",
        yaxis_title="Activité",
        height=600,
        showlegend=True,
        xaxis=dict(
            tickformat="%H:%M",
            tickangle=45
        )
    )
    
    return fig

def create_weekly_calendar(all_activities_planned):
    """Create a weekly calendar from daily schedules"""
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
        end_time = start_time + timedelta(minutes=duration)
        
        calendar_data.append({
            'Jour': day_name,
            'Heure_Début': start_time,
            'Heure_Fin': end_time,
            'Activité': act,
            'Score': reward,
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
        title="Planning Hebdomadaire Optimisé",
        hover_data=["Score"]
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

def calculate_schedule_stats(activities):
    """Calculate statistics about the generated schedule"""
    if not activities:
        return {}
    
    activity_counts = {}
    rewards_by_hour = {}
    total_reward = 0
    
    for activity, time_slot, reward, duration in activities:
        # Count activities
        activity_counts[activity] = activity_counts.get(activity, 0) + 1
        
        # Track rewards by hour
        rewards_by_hour[time_slot] = rewards_by_hour.get(time_slot, 0) + reward
        total_reward += reward
    
    most_productive_hour = max(rewards_by_hour, key=rewards_by_hour.get) if rewards_by_hour else None
    
    return {
        'activity_counts': activity_counts,
        'rewards_by_hour': rewards_by_hour,
        'most_productive_hour': most_productive_hour,
        'total_reward': total_reward
    }

def main():
    st.title("📅 Assistant Personnel de Gestion du Temps")
    st.subheader("Optimisez votre planning quotidien avec l'intelligence artificielle")
    
    # Sidebar configuration
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
    
    # Activity configuration
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
    
    # Display activities with colors
    st.sidebar.subheader("Liste des activités")
    for i, activity in enumerate(activities):
        color = ACTIVITY_COLORS.get(activity, DEFAULT_COLOR)
        st.sidebar.markdown(
            f"<div style='background-color:{color}; padding:5px; border-radius:5px; margin:2px 0; color:white; font-weight:bold;'>"
            f"{i+1}. {activity}</div>",
            unsafe_allow_html=True
        )
    
    # Constraints
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
            st.sidebar.success(f"Contrainte ajoutée pour {constraint_activity}")
    
    if user_constraints:
        st.sidebar.subheader("Contraintes ajoutées")
        for activity, slots in user_constraints.items():
            st.sidebar.write(f"**{activity}**: {', '.join([f'{s:02d}:00' for s in slots])}")
    
    # Generate button
    if planning_type == "📆 Planning quotidien":
        generate_button = st.sidebar.button("🚀 Générer planning quotidien", type="primary")
    else:
        generate_button = st.sidebar.button("🚀 Générer planning hebdomadaire", type="primary")
    
    # Main content
    if generate_button:
        with st.spinner("Génération du planning en cours..."):
            if planning_type == "📆 Planning quotidien":
                # Daily schedule
                optimizer = SimpleScheduleOptimizer(activities, selected_day)
                activities_planned = optimizer.optimize_schedule(user_constraints)
                
                st.subheader(f"Planning optimisé pour {selected_day_name}")
                
                # Map selected day to a date
                day_to_date = {
                    "Lundi": "2025-05-12", "Mardi": "2025-05-13", "Mercredi": "2025-05-14",
                    "Jeudi": "2025-05-15", "Vendredi": "2025-05-16", "Samedi": "2025-05-17",
                    "Dimanche": "2025-05-18"
                }
                reference_date = day_to_date[selected_day_name]
                
                # Visualize schedule
                fig = visualize_schedule_plotly(activities_planned, reference_date)
                st.plotly_chart(fig, use_container_width=True)
                
                if activities_planned:
                    # Activity details
                    st.subheader("📋 Détail des activités")
                    
                    activities_df = pd.DataFrame([
                        {
                            "Activité": act,
                            "Heure": f"{slot:02d}:00",
                            "Score": f"{reward:.2f}",
                            "Durée": f"{duration:.0f} min"
                        }
                        for act, slot, reward, duration in activities_planned
                    ])
                    
                    st.dataframe(activities_df.sort_values(by="Heure"), use_container_width=True)
                    
                    # Statistics
                    stats = calculate_schedule_stats(activities_planned)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Distribution des activités")
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
                        st.subheader("⚡ Productivité par heure")
                        if stats['rewards_by_hour']:
                            hours = [f"{h:02d}:00" for h in sorted(stats['rewards_by_hour'].keys())]
                            rewards = [stats['rewards_by_hour'][int(h.split(':')[0])] for h in hours]
                            
                            fig = px.bar(
                                x=hours,
                                y=rewards,
                                title="Score de productivité par heure",
                                labels={'x': 'Heure', 'y': 'Score de productivité'},
                                color=rewards,
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if stats['most_productive_hour'] is not None:
                        st.info(f"✨ **Votre heure la plus productive est {stats['most_productive_hour']:02d}:00**")
                
                # Schedule evaluation
                st.subheader("📈 Évaluation du planning")
                total_reward = stats.get('total_reward', 0)
                st.write(f"**Score total du planning**: {total_reward:.2f}")
                
                quality = min(max(total_reward / len(activities_planned) if activities_planned else 0, 0), 1)
                
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=quality * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Qualité du planning (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                st.plotly_chart(gauge, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Recommandations")
                
                if quality < 0.5:
                    st.warning("⚠️ Votre planning pourrait être amélioré. Essayez d'ajouter plus de contraintes ou de modifier vos activités.")
                elif quality < 0.8:
                    st.info("ℹ️ Votre planning est bon, mais il pourrait encore être optimisé.")
                else:
                    st.success("🎉 Excellent planning ! Votre journée est optimisée pour une productivité maximale.")
            
            else:
                # Weekly schedule
                st.subheader("🗓️ Planning hebdomadaire optimisé")
                
                all_activities_planned = []
                total_weekly_reward = 0
                
                for day in range(1, 8):
                    optimizer = SimpleScheduleOptimizer(activities, day)
                    day_activities = optimizer.optimize_schedule(user_constraints)
                    all_activities_planned.extend([
                        (act, slot, reward, duration, day) 
                        for act, slot, reward, duration in day_activities
                    ])
                    total_weekly_reward += sum(reward for _, _, reward, _ in day_activities)
                
                # Weekly calendar visualization
                fig = create_weekly_calendar(all_activities_planned)
                st.plotly_chart(fig, use_container_width=True)
                
                # Weekly statistics
                st.subheader("📊 Statistiques hebdomadaires")
                
                activities_by_day = {}
                for day_idx in range(7):
                    day_name = list(day_options.keys())[day_idx]
                    day_activities = [act for act, _, _, _, day in all_activities_planned if day == day_idx + 1]
                    activities_by_day[day_name] = len(day_activities)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📅 Activités par jour")
                    fig = px.bar(
                        x=list(activities_by_day.keys()),
                        y=list(activities_by_day.values()),
                        title="Nombre d'activités par jour",
                        labels={'x': 'Jour', 'y': 'Nombre d\'activités'},
                        color=list(activities_by_day.values()),
                        color_continuous_scale='blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("🏆 Score hebdomadaire")
                    st.metric("Score total de la semaine", f"{total_weekly_reward:.2f}")
                    
                    quality = min(max(total_weekly_reward / 50, 0), 1)
                    
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=quality * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Qualité hebdomadaire (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkgreen"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ]
                        }
                    ))
                    
                    st.plotly_chart(gauge, use_container_width=True)
                
                # Activity distribution for the week
                activity_counts = {}
                for act, _, _, _, _ in all_activities_planned:
                    activity_counts[act] = activity_counts.get(act, 0) + 1
                
                st.subheader("🥧 Répartition des activités sur la semaine")
                fig = px.pie(
                    values=list(activity_counts.values()),
                    names=list(activity_counts.keys()),
                    title="Distribution des activités sur la semaine",
                    color=list(activity_counts.keys()),
                    color_discrete_map=ACTIVITY_COLORS
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Export functionality
                st.subheader("💾 Exporter le planning")
                
                export_data = []
                days = list(day_options.keys())
                
                for day_idx, day_name in enumerate(days):
                    day_activities = [
                        (act, slot, duration) 
                        for act, slot, _, duration, day in all_activities_planned 
                        if day == day_idx + 1
                    ]
                    for activity, time_slot, duration in day_activities:
                        export_data.append({
                            'Jour': day_name,
                            'Heure_Début': f"{time_slot:02d}:00",
                            'Heure_Fin': f"{(time_slot + duration//60):02d}:{duration%60:02d}",
                            'Durée': f"{duration:.0f} min",
                            'Activité': activity
                        })
                
                if export_data:
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger le planning (CSV)",
                        data=csv,
                        file_name=f"planning_hebdomadaire_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🎯 Bienvenue dans votre Assistant Personnel de Gestion du Temps ! 
        
        Cette application utilise l'intelligence artificielle pour vous aider à optimiser votre emploi du temps 
        quotidien ou hebdomadaire.
        
        ### 🔧 Comment ça marche ?
        
        Notre application utilise un algorithme d'optimisation intelligent pour analyser vos préférences
        et générer un planning optimisé qui maximise votre productivité et votre satisfaction.
        
        ### ✨ Fonctionnalités
        
        - **📆 Planning quotidien**: Obtenez un emploi du temps optimisé pour une journée spécifique
        - **🗓️ Planning hebdomadaire**: Visualisez un calendrier complet pour toute la semaine
        - **🎨 Activités personnalisées**: Définissez vos propres types d'activités
        - **⏰ Contraintes horaires**: Ajoutez des activités obligatoires à des heures précises
        - **📊 Statistiques**: Analysez la qualité de votre planning et identifiez vos heures les plus productives
        - **💾 Export**: Téléchargez votre planning au format CSV
        
        ### 🚀 Commencer
        
        Utilisez les options dans la barre latérale pour personnaliser votre planning, puis cliquez sur 
        **"Générer planning"** pour voir les résultats.
        
        ---
        *Optimisez votre temps avec l'IA ! ⚡*
        """)
        
        # Display activity preferences as a helpful reference
        with st.expander("📋 Voir les préférences par défaut des activités"):
            pref_data = []
            for activity, prefs in ACTIVITY_PREFERENCES.items():
                preferred_times = ", ".join([f"{h:02d}:00" for h in prefs['preferred_hours'][:5]])
                if len(prefs['preferred_hours']) > 5:
                    preferred_times += "..."
                    
                pref_data.append({
                    'Activité': activity,
                    'Heures préférées': preferred_times,
                    'Durée typique': f"{prefs['duration']} min",
                    'Priorité': f"{prefs['priority']}/10"
                })
            
            pref_df = pd.DataFrame(pref_data)
            st.dataframe(pref_df, use_container_width=True)

if __name__ == "__main__":
    main()