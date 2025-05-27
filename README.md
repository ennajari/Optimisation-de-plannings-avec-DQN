# Personal Scheduler: An AI-Based Daily Schedule Optimization Project

This repository contains a reinforcement learning solution for optimizing daily schedules using Deep Q-Learning (DQN). The project predicts and suggests optimal activities based on various factors such as the time of day, weekend status, and activity duration.

## Table of Contents

* [Overview](#overview)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
  * [Data Exploration](#data-exploration)
  * [Data Preprocessing](#data-preprocessing)
  * [Training the DQN Model](#training-the-dqn-model)
  * [Running the Streamlit App](#running-the-streamlit-app)
* [Models and Environment](#models-and-environment)
* [Acknowledgements](#acknowledgements)

## Overview

The **Personal Scheduler** aims to automate the daily scheduling process based on historical activity data. Using Deep Q-Learning, it learns to schedule activities in a way that maximizes productivity while minimizing wasted time.

The model is built using TensorFlow, and the environment is created using OpenAI's `gym` library to simulate a daily schedule. The project also provides a simple Streamlit app for interactive testing of the model.

## Project Structure

```bash
personal_scheduler/
├── Data/
│   ├── raw/atus_full_selected.csv          # Raw activity data
│   └── processed/preprocessed_data.csv    # Cleaned and preprocessed data
├── environment/
│   └── schedule_env.py                    # Custom Gym environment for scheduling
├── models/
│   └── dqn_model.h5                       # Trained DQN model
├── outputs/                                # Model outputs (logs, results)
├── requirements.txt                       # Python dependencies
├── notebooks/
│   ├── 1_data_exploration.ipynb           # Exploratory data analysis (EDA)
│   ├── 2_data_preprocessing.ipynb         # Data cleaning and preprocessing
│   └── 3_dqn_training.ipynb               # Training the Deep Q-Network
└── streamlit_app/
    └── app.py                             # Streamlit app for interactive testing
```
Data
The data used in this project is stored in the Data/raw/ folder. The raw dataset atus_full_selected.csv contains activity records with details like activity duration, activity names, start time, and the day of the week.

The Data/processed/ folder contains the cleaned and preprocessed version of the data that is used for training the model (preprocessed_data.csv).

Environment
The environment/ folder contains the custom scheduling environment for the reinforcement learning agent. This environment simulates the process of scheduling activities and provides feedback (reward) based on the actions taken.

Models
The models/ folder stores the trained DQN model (dqn_model.h5), which can be used to predict optimal actions (activities) based on the current state (time of day, day of the week, weekend status).

Notebooks
The notebooks/ directory contains Jupyter notebooks for:

Data Exploration (1_data_exploration.ipynb): Understanding the structure of the dataset.
Data Preprocessing (2_data_preprocessing.ipynb): Cleaning the dataset and preparing it for training.
DQN Training (3_dqn_training.ipynb): Training the reinforcement learning agent using DQN.
Streamlit App
The streamlit_app/ folder contains the app (app.py), which allows users to interact with the trained model through a web interface.

Installation
To run this project, make sure you have Python 3.x installed. You can create a virtual environment and install the necessary dependencies by running the following commands:
# Create and activate a virtual environment
```bash
python -m venv schedule_env
source schedule_env/bin/activate  # On Windows use `schedule_env\Scripts\activate`
```
# Install required dependencies
```bash
pip install -r requirements.txt
```
# Usage
Data Exploration
Run the 1_data_exploration.ipynb notebook to perform exploratory data analysis (EDA) on the raw activity data. This will give you an understanding of the dataset's structure, activity types, and potential patterns.
```bash
jupyter notebook notebooks/1_data_exploration.ipynb
```
# Data Preprocessing
To clean and preprocess the data, run the 2_data_preprocessing.ipynb notebook. This notebook will handle missing values, convert time data, and perform feature engineering.
```bash
jupyter notebook notebooks/2_data_preprocessing.ipynb
```
# Training the DQN Model
To train the Deep Q-Network (DQN), run the 3_dqn_training.ipynb notebook. This will train the model on the processed data and save the trained model to models/dqn_model.h5.
```bash
jupyter notebook notebooks/3_dqn_training.ipynb
```
# Running the Streamlit App
After training the model, you can use the streamlit_app/app.py to launch an interactive web app where you can test the model by providing different inputs.
```bash
streamlit run ui/app.py
```
# Models and Environment
Schedule Environment
The scheduling environment is implemented in the environment/schedule_env.py file using the gym library. It models the process of scheduling activities, where each action corresponds to selecting an activity, and the reward is calculated based on how productive the chosen activity is.

# Deep Q-Network (DQN)
The DQN model is implemented using TensorFlow in the 3_dqn_training.ipynb notebook. The model consists of a neural network with two hidden layers, and it learns to map states (time of day, day of the week, weekend status) to actions (activities) using Q-learning.

# Acknowledgements
This project uses OpenAI's Gym for building the environment.
TensorFlow is used for creating and training the DQN model.
The Streamlit app is powered by the Streamlit library for building interactive web applications.


### Additional Enhancements

1. **Project Title and Description**: Clearly state the title and a brief description of the project at the beginning.

2. **Table of Contents**: Include a table of contents to make it easier for users to navigate through the document.

3. **Detailed Sections**: Break down the content into detailed sections such as Overview, Project Structure, Installation, Usage, Models and Environment, and Acknowledgements.

4. **Code Snippets**: Include code snippets for installation and usage instructions to make it easier for users to follow along.

5. **Visual Aids**: Use visual aids like code blocks and lists to make the information more digestible.

6. **Acknowledgements**: Acknowledge the tools and libraries used in the project.

This enhanced `README.md` should provide a comprehensive guide to your project, making it easier for users to understand and use.
