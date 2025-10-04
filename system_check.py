import pandas as pd
import joblib
import numpy as np

# ------------------------------
# Load your saved model objects
# ------------------------------
model_objects = joblib.load("C:/Users/Nethmi Niwarthana/Desktop/Mini_project/random_forest_model.sav")
model = model_objects['model']
scaler = model_objects['scaler']
label_encoders = model_objects['label_encoders']
grade_encoder = model_objects['grade_encoder']

# ------------------------------
# Helper functions for messages
# ------------------------------
def get_grade_message(grade):
    messages = {
        "A": "🟢 Excellent! Safe to dine.",
        "B": "🟡 Good. Some issues.",
        "C": "🔴 Risky! Extra caution advised.",
        "N": "⚪ Not yet graded.",
        "Z": "⚪ Grade pending.",
        "P": "⚪ Grade pending appeal."
    }
    return messages.get(grade, "No info.")

def owner_recommendations(grade):
    tips = {
        "A": "🍕 Keep up the great work!",
        "B": "🍕 Improve food handling and training.",
        "C": "🍕 Immediate action needed!",
        "N": "🍕 Prepare for first inspection.",
        "Z": "🍕 Pending results: maintain standards.",
        "P": "🍕 Prepare compliance evidence."
    }
    return tips.get(grade, "No recommendations.")

def authority_actions(grade):
    actions = {
        "A": "🏛️ Low priority. Routine inspection.",
        "B": "🏛️ Medium priority. Follow-up needed.",
        "C": "🏛️ High priority. Immediate action required.",
        "N": "🏛️ Schedule initial inspection.",
        "Z": "🏛️ Pending. Monitor closely.",
        "P": "🏛️ Under appeal. Review evidence."
    }
    return actions.get(grade, "No actions.")

# ------------------------------
# Sample test data
# ------------------------------
data = [
    {"INSPECTION TYPE": "Pre-permit (Operational) / Initial Inspection", "CRITICAL FLAG": "Critical", "VIOLATION CODE": "08A", "SCORE": 25, "inspection_year": 2024, "inspection_month": 6, "inspection_day_of_week": 1},
    {"INSPECTION TYPE": "Cycle Inspection / Initial Inspection", "CRITICAL FLAG": "Critical", "VIOLATION CODE": "06C", "SCORE": 18, "inspection_year": 2024, "inspection_month": 7, "inspection_day_of_week": 4},
    {"INSPECTION TYPE": "Cycle Inspection / Re-inspection", "CRITICAL FLAG": "Not Applicable", "VIOLATION CODE": "10D", "SCORE": 12, "inspection_year": 2024, "inspection_month": 7, "inspection_day_of_week": 0},
    {"INSPECTION TYPE": "Administrative Miscellaneous / Second Compliance Inspection", "CRITICAL FLAG": "Critical", "VIOLATION CODE": "11B", "SCORE": 15, "inspection_year": 2024, "inspection_month": 7, "inspection_day_of_week": 2},  # unseen type
]

df = pd.DataFrame(data)

# ------------------------------
# Encode categorical features safely
# ------------------------------
for feature in ['INSPECTION TYPE', 'CRITICAL FLAG', 'VIOLATION CODE']:
    le = label_encoders[feature]
    # Map unseen labels to "Other" if exists, else first class
    safe_values = []
    for val in df[feature]:
        if val in le.classes_:
            safe_values.append(val)
        else:
            safe_values.append(le.classes_[0])  # fallback to first class
    df[feature] = le.transform(safe_values)

# ------------------------------
# Scale features
# ------------------------------
X_scaled = scaler.transform(df)

# ------------------------------
# Predict grades
# ------------------------------
pred_encoded = model.predict(X_scaled)
pred_grades = grade_encoder.inverse_transform(pred_encoded)
df['Predicted Grade'] = pred_grades

# ------------------------------
# Show messages for each role
# ------------------------------
df['Customer Msg'] = df['Predicted Grade'].apply(get_grade_message)
df['Owner Rec'] = df['Predicted Grade'].apply(owner_recommendations)
df['Authority Act'] = df['Predicted Grade'].apply(authority_actions)

# ------------------------------
# Print results
# ------------------------------
print(df)

# ------------------------------
# Optional: Save to CSV
# ------------------------------
df.to_csv("predicted_results.csv", index=False)
print("✅ Predictions saved to predicted_results.csv")

import os

# Save to CSV
df.to_csv("predicted_results.csv", index=False)
print("✅ Predictions saved to predicted_results.csv")

# Open the CSV automatically (Windows)
os.startfile("predicted_results.csv")

