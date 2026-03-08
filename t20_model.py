import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_model():

    df = pd.read_csv("t20_demo_data.csv")

    X = df[['current_score','balls_left','current_run_rate']]
    y = df['final_score']

    model = RandomForestRegressor()

    model.fit(X,y)

    return model
