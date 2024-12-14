import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px

st.set_page_config(
    page_title="Hockey Visualization App",
    layout="wide"
)
st.title("Hockey Visualization App")

st.sidebar.header("Model Settings")
workspace = st.sidebar.text_input("Workspace", placeholder="Enter workspace name")
model = st.sidebar.text_input("Model", placeholder="Enter model name")
version = st.sidebar.text_input("Version", placeholder="Enter model version")

if st.sidebar.button("Get model"):
    # Appel de l'API pour charger le modèle
    response = requests.post(
        "http://127.0.0.1:<PORT>/download_registry_model",
        json={"workspace": workspace, "model": model, "version": version}
    )
    if response.status_code == 200:
        st.sidebar.success("Model downloaded and loaded successfully!")
    else:
        st.sidebar.error("Failed to download the model.")

game_id = st.text_input("Game ID", placeholder="Enter game ID (e.g., 2021020329)")

if st.button("Ping game"):
    # Appel de l'API pour récupérer les données du jeu
    response = requests.get(f"http://127.0.0.1:<PORT>/game_data/{game_id}")
    if response.status_code == 200:
        game_data = response.json()
        st.write(f"Game {game_id}: {game_data['teams']['home']} vs {game_data['teams']['away']}")
        st.write(f"Period: {game_data['period']} - Time left: {game_data['time_left']}")
        
        # Affichage des scores
        st.metric(
            label=f"{game_data['teams']['home']} xG (actual)",
            value=f"{game_data['xG']['home']} ({game_data['score']['home']})",
            delta=game_data['xG']['home'] - game_data['score']['home']
        )
        st.metric(
            label=f"{game_data['teams']['away']} xG (actual)",
            value=f"{game_data['xG']['away']} ({game_data['score']['away']})",
            delta=game_data['xG']['away'] - game_data['score']['away']
        )
        
        # Affichage des données pour les prédictions
        events_df = pd.DataFrame(game_data["events"])
        st.subheader("Data used for predictions (and predictions)")
        st.dataframe(events_df)
    else:
        st.error("Failed to fetch game data.")

if "game_data" in locals() and game_data:
    # Filtrer les nouveaux événements
    new_events = [event for event in game_data["events"] if not event["processed"]]
    
    if new_events:
        df = pd.DataFrame(new_events)
        predictions = requests.post(
            "http://127.0.0.1:<PORT>/predict",
            json=json.loads(df.to_json(orient="records"))
        )
        
        if predictions.status_code == 200:
            predictions_df = pd.DataFrame(predictions.json())
            predictions_df["Model output"] = predictions_df["xG"]
            st.write(predictions_df)
        else:
            st.error("Prediction service failed.")

if "game_data" in locals() and game_data:
    fig = px.line(events_df, x="time", y="xG", color="team", title="Expected Goals Over Time")
    st.plotly_chart(fig)
