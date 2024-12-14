import streamlit as st
import pandas as pd
import requests
import json
import wandb
import plotly.express as px

# config streamlit
st.set_page_config(
    page_title="Hockey Visualization App",
    layout="wide"
)
st.title("Hockey Visualization App")

# init de WandB
st.sidebar.header("WandB Configuration")
workspace = st.sidebar.text_input("Workspace", placeholder="Enter WandB workspace name")
model = st.sidebar.text_input("Model", placeholder="Enter model name")
version = st.sidebar.text_input("Version", placeholder="Enter model version")

# télécharger le modèle via WandB
if st.sidebar.button("Get model"):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/download_registry_model",
            json={"workspace": workspace, "model": model, "version": version}
        )
        if response.status_code == 200:
            st.sidebar.success("Model downloaded and loaded successfully!")
            st.sidebar.write(f"Loaded model: {model} from workspace: {workspace}, version: {version}")
        else:
            st.sidebar.error(f"Failed to download the model. Error: {response.json().get('error', 'Unknown error')}")
    except Exception as e:
        st.sidebar.error(f"An error occurred: {e}")

# ID du match
game_id = st.text_input("Game ID", placeholder="Enter Game ID (e.g., 2021020329)")

if st.button("Ping game"):
    try:
        # API pour recup les données du match
        response = requests.get(f"http://127.0.0.1:8000/game_data/{game_id}")
        if response.status_code == 200:
            game_data = response.json()

            home_team = game_data['teams']['home']
            away_team = game_data['teams']['away']
            period = game_data['period']
            time_left = game_data['time_left']
            score = game_data['score']
            xg = game_data['xG']

            st.subheader(f"Game {game_id}: {home_team} vs {away_team}")
            st.write(f"**Period**: {period} - **Time Left**: {time_left}")

            col1, col2 = st.columns(2)
            col1.metric(
                f"{home_team} xG (actual)",
                value=f"{xg['home']} ({score['home']})",
                delta=round(xg['home'] - score['home'], 2)
            )
            col2.metric(
                f"{away_team} xG (actual)",
                value=f"{xg['away']} ({score['away']})",
                delta=round(xg['away'] - score['away'], 2)
            )

            events_df = pd.DataFrame(game_data["events"])
            st.subheader("Data used for predictions (and predictions)")
            st.dataframe(events_df)

            new_events = events_df[~events_df['processed']]
            if not new_events.empty:
                prediction_response = requests.post(
                    "http://127.0.0.1:0>/predict",
                    json=json.loads(new_events.to_json(orient="records"))
                )
                if prediction_response.status_code == 200:
                    predictions_df = pd.DataFrame(prediction_response.json())
                    predictions_df["Model output"] = predictions_df["xG"]
                    st.write("Predictions:")
                    st.dataframe(predictions_df)

                    # save les preds dans WandB
                    run = wandb.init(project="hockey-xg", name=f"predictions_game_{game_id}", reinit=True)
                    table = wandb.Table(dataframe=predictions_df)
                    run.log({"predictions": table})
                    run.finish()
                    st.success("Predictions logged to WandB successfully!")
                else:
                    st.error("Prediction service failed.")
            else:
                st.info("No new events to process.")
        else:
            st.error(f"Failed to fetch game data. Error: {response.status_code}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# visualisation graphique (bonus)
if "game_data" in locals() and game_data:
    try:
        fig = px.line(events_df, x="time", y="xG", color="team", title="Expected Goals Over Time")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Failed to generate visualization: {e}")
