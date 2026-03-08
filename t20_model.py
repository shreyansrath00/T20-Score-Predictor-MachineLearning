import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

# Load dataset
df = pd.read_csv("t20_matches.csv")

# Convert innings column from string → Python object
df['innings'] = df['innings'].apply(ast.literal_eval)

rows = []

for i in range(len(df)):
    match_id = df.iloc[i]['match_id']
    innings = df.iloc[i]['innings']
    
    for inning in innings:
        inning_key = list(inning.keys())[0]
        deliveries = inning[inning_key]['deliveries']
        
        for delivery in deliveries:
            over = list(delivery.keys())[0]
            data = delivery[over]
            runs = data['runs']['total']
            
            rows.append({
                "match_id": match_id,
                "over": over,
                "runs": runs
            })

# Convert to dataframe
deliveries_df = pd.DataFrame(rows)

print(deliveries_df.head())
print(deliveries_df.shape)

# Convert over to float
deliveries_df['over'] = deliveries_df['over'].astype(float)

# Calculate cumulative score
deliveries_df['current_score'] = deliveries_df.groupby('match_id')['runs'].cumsum()

# Calculate balls bowled
deliveries_df['balls_bowled'] = deliveries_df['over'].apply(
    lambda x: int(x) * 6 + int((x - int(x)) * 10)
)

# Total balls in T20 = 120
deliveries_df['balls_left'] = 120 - deliveries_df['balls_bowled']

# Current run rate
deliveries_df['current_run_rate'] = deliveries_df['current_score'] / (
    deliveries_df['balls_bowled'] / 6
)

# Calculate final score for each match
final_scores = deliveries_df.groupby('match_id')['runs'].sum().reset_index()
final_scores.rename(columns={'runs': 'final_score'}, inplace=True)

# Merge final score
deliveries_df = deliveries_df.merge(final_scores, on='match_id')

print(deliveries_df.head())

# Features for training
X = deliveries_df[['current_score', 'balls_left', 'current_run_rate']]
y = deliveries_df['final_score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model training complete")

# Evaluate model
predictions = model.predict(X_test)

accuracy = r2_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("t20_model.pkl", "wb"))

print("Model saved successfully")