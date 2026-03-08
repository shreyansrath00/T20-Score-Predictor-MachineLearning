import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("t20_model.pkl", "rb"))

st.set_page_config(page_title="T20 Score Predictor", page_icon="🏏")

st.title("🏏 T20 Score Predictor")

st.caption("Predicts the final T20 score based on current match conditions using a machine learning model.")
st.caption("Machine Learning model trained on 1,100,000+ ball-by-ball T20 deliveries")

# Team selection
teams = [
    "India","Australia","England","New Zealand",
    "Pakistan","South Africa","West Indies",
    "Sri Lanka","Bangladesh","Afghanistan"
]

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select the batting team", teams)

with col2:
    bowling_team = st.selectbox("Select the bowling team", teams)

# Prevent same team selection
if batting_team == bowling_team:
    st.warning("Batting and bowling team cannot be the same.")
    st.stop()

st.markdown("---")

# Match state inputs
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input("Score", min_value=0, max_value=400, value=100)

with col4:
    overs_completed = st.number_input(
        "Overs completed",
        min_value=0.0,
        max_value=20.0,
        value=10.0,
        step=0.1
    )

with col5:
    wickets_out = st.number_input("Wickets out", min_value=0, max_value=10, value=2)

runs_last_5 = st.number_input(
    "Runs in last 5 overs",
    min_value=0,
    max_value=100,
    value=40
)

st.markdown("")

# Prediction button
if st.button("Predict Score", key="predict_button"):

    # Convert overs to balls
    over_int = int(overs_completed)
    balls_part = int(round((overs_completed - over_int) * 10))

    if balls_part > 5:
        st.error("Invalid overs format. Use values like 10.2 (10 overs and 2 balls).")
        st.stop()

    balls_bowled = over_int * 6 + balls_part

    # Total balls in T20
    balls_left = max(0, 120 - balls_bowled)

    # Current run rate
    if overs_completed > 0:
        current_run_rate = score / overs_completed
    else:
        current_run_rate = 0

    # ML prediction
    prediction = model.predict([[score, balls_left, current_run_rate]])
    predicted_score = int(prediction[0])

    # Wicket impact adjustment
    wickets_left = 10 - wickets_out

    if wickets_left <= 2:
        predicted_score -= 25
    elif wickets_left <= 4:
        predicted_score -= 15
    elif wickets_left <= 6:
        predicted_score -= 5

    # Momentum adjustment
    if runs_last_5 > 60:
        predicted_score += 15
    elif runs_last_5 > 50:
        predicted_score += 10
    elif runs_last_5 < 25:
        predicted_score -= 10

    # Realistic scoring caps
    max_possible = score + (balls_left * 6)

    if predicted_score > max_possible:
        predicted_score = max_possible

    if predicted_score > score + 120:
        predicted_score = score + 120

    if predicted_score < score:
        predicted_score = score

    # Show prediction
    st.success(f"🏏 Predicted Final Score: {predicted_score}")

    # Additional analytics
    runs_remaining = predicted_score - score

    st.info(
        f"""
Match Projection

Current Score: {score}

Predicted Final Score: {predicted_score}

Expected Runs Remaining: {runs_remaining}

Balls Remaining: {balls_left}

Current Run Rate: {current_run_rate:.2f}
"""
    )