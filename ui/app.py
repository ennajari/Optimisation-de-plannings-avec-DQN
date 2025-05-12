import streamlit as st
import numpy as np
import tensorflow as tf

# Chargement du modèle
model = tf.keras.models.load_model('../models/dqn_final.h5', compile=False)


st.set_page_config(page_title="Optimisation de Planning", layout="centered")
st.title("🧠 Optimisation de Planning avec DQN")
st.markdown("Entrez les caractéristiques de la tâche pour prédire l'action optimale.")

# Entrée utilisateur pour les 3 features (adaptées à l'entrée du modèle)
feature1 = st.number_input("Durée de la tâche (en heures)", min_value=0.0, value=1.0)
feature2 = st.number_input("Priorité (ex : 1=faible, 5=élevée)", min_value=1, max_value=5, value=3)
feature3 = st.number_input("Complexité (1 à 10)", min_value=1, max_value=10, value=5)

# Former un tableau d'entrée compatible avec le modèle (1, 3)
input_array = np.array([[feature1, feature2, feature3]])

# Faire la prédiction
prediction = model.predict(input_array)
predicted_action = np.argmax(prediction[0])

# Affichage du résultat
st.success(f"✅ Action optimale prédite : **{predicted_action}**")
