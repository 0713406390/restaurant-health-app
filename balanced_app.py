import streamlit as st
import pandas as pd
import joblib
import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ------------------------------
# Load saved model
# ------------------------------
# NEW LINE - USE THIS ONE
model_objects = joblib.load("random_forest_model.sav")

model = model_objects['model']
scaler = model_objects['scaler']
label_encoders = model_objects['label_encoders']
features = model_objects['features']
grade_encoder = model_objects['grade_encoder']

# ------------------------------
# Streamlit App Config
# ------------------------------
st.set_page_config(page_title="Restaurant Grade Predictor", layout="wide")

st.title("🍽️ NYC Restaurant Health Grade Predictor")

# ------------------------------
# Role Selection
# ------------------------------
role = st.sidebar.selectbox("Select Your Role", ["Customer", "Restaurant Owner", "Health Authority"])

# ------------------------------
# Helper: Messages for Customer
# ------------------------------
def get_grade_message(grade):
    messages = {
        "A": "🟢 Excellent! This restaurant meets the highest standards of food safety and cleanliness. Customers can dine here with confidence.",
        "B": "🟡 Good. The restaurant is fairly safe but improvements are needed in hygiene or food handling.",
        "C": "🔴 Risky! This grade indicates significant health and safety concerns. Extra caution is advised when dining.",
        "N": "⚪ Not Yet Graded. This restaurant hasn’t received a final grade yet.",
        "Z": "⚪ Grade Pending. Awaiting inspection or administrative processing.",
        "P": "⚪ Grade Pending Appeal. The grade may change after review."
    }
    return messages.get(grade, "No additional information available.")

# ------------------------------
# Helper: Recommendations for Owner
# ------------------------------
def owner_recommendations(grade):
    tips = {
        "A": "🍕 Keep up the great work! Continue regular cleaning, safe food storage, and staff hygiene training.",
        "B": "🍕 Focus on improving food handling, staff training, and addressing minor violations before the next inspection.",
        "C": "🍕 Immediate action needed: deep cleaning, pest control, and strict hygiene enforcement. Consider staff retraining.",
        "N": "🍕 Prepare thoroughly for your first inspection: ensure cleanliness and compliance with safety standards.",
        "Z": "🍕 Pending results: maintain high standards and prepare documentation in case of re-evaluation.",
        "P": "🍕 Since this is under appeal, prepare strong evidence of compliance and corrective actions."
    }
    return tips.get(grade, "🍕 No specific recommendations available.")

# ------------------------------
# Helper: Recommendations for Health Authority
# ------------------------------
def authority_actions(grade):
    actions = {
        "A": "🏛️ Low priority. Routine inspection scheduling is sufficient.",
        "B": "🏛️ Medium priority. Schedule a follow-up inspection to monitor compliance.",
        "C": "🏛️ High priority. Immediate inspection and enforcement required to protect public health.",
        "N": "🏛️ Not graded yet. Schedule initial inspection soon.",
        "Z": "🏛️ Pending. Monitor status and ensure inspection is completed.",
        "P": "🏛️ Under appeal. Review case details and verify compliance evidence."
    }
    return actions.get(grade, "🏛️ No specific actions available.")

# ------------------------------
# Helper: Grade Badge
# ------------------------------
def grade_badge(grade):
    colors = {
        "A": "#4CAF50",   # Green
        "B": "#FFC107",   # Yellow
        "C": "#F44336",   # Red
        "N": "#9E9E9E",   # Gray
        "Z": "#9E9E9E",
        "P": "#9E9E9E"
    }
    color = colors.get(grade, "#9E9E9E")
    return f"<span style='background-color:{color}; color:white; padding:6px 15px; border-radius:8px; font-size:18px; font-weight:bold;'>{grade}</span>"

# ------------------------------
# Helper: Summary Box
# ------------------------------
def summary_box(grade, role):
    colors = {
        "A": "#4CAF50",
        "B": "#FFC107",
        "C": "#F44336",
        "N": "#9E9E9E",
        "Z": "#9E9E9E",
        "P": "#9E9E9E"
    }
    messages = {
        "Customer": {
            "A": "✅ Safe choice! You can dine here with confidence.",
            "B": "⚠️ Some issues noted, but still acceptable.",
            "C": "❌ Risky! Consider alternatives for safety.",
            "N": "ℹ️ Not graded yet – no final evaluation available.",
            "Z": "⏳ Pending inspection – grade not final.",
            "P": "⏳ Under appeal – grade may change."
        },
        "Restaurant Owner": {
            "A": "🍕 Fantastic! Your restaurant is in top condition.",
            "B": "🍕 Good, but improvements needed to reach Grade A.",
            "C": "🍕 Warning: Major improvements needed immediately!",
            "N": "🍕 First inspection pending – prepare thoroughly.",
            "Z": "🍕 Pending inspection – keep standards high.",
            "P": "🍕 Appeal ongoing – ensure compliance evidence is ready."
        },
        "Health Authority": {
            "A": "🏛️ Low priority: No urgent action needed.",
            "B": "🏛️ Medium priority: Monitor this establishment.",
            "C": "🏛️ High priority: Immediate inspection required!",
            "N": "🏛️ Initial inspection required soon.",
            "Z": "🏛️ Pending inspection – monitor closely.",
            "P": "🏛️ Under appeal – review documentation."
        }
    }
    color = colors.get(grade, "#9E9E9E")
    message = messages.get(role, {}).get(grade, "No summary available.")
    return f"<div style='background-color:{color}; padding:15px; border-radius:10px; color:white; font-size:18px; font-weight:bold; text-align:center;'>{message}</div>"

# ------------------------------
# Improved PDF Report Generator
# ------------------------------
def generate_styled_pdf_report(title, details_dict, recommendations=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 20))

    # Details as table
    data = [["Field", "Value"]]
    for key, value in details_dict.items():
        data.append([key, str(value)])

    table = Table(data, hAlign="LEFT", colWidths=[150, 350])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4CAF50")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # Recommendations / Actions
    if recommendations:
        story.append(Paragraph("<b>Recommendations / Actions</b>", styles["Heading2"]))
        story.append(Spacer(1, 10))
        story.append(Paragraph(recommendations, styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ------------------------------
# CUSTOMER ROLE
# ------------------------------
if role == "Customer":
    st.header("👤 Customer Portal")
    st.write("Check the predicted health grade for a restaurant inspection.")
    
    inspection_type = st.selectbox("Inspection Type", label_encoders['INSPECTION TYPE'].classes_)
    critical_flag = st.selectbox("Critical Flag", label_encoders['CRITICAL FLAG'].classes_)
    violation_code = st.selectbox("Violation Code", label_encoders['VIOLATION CODE'].classes_)
    score = st.number_input("Score", min_value=0, max_value=100, value=10)
    inspection_date = st.date_input("Inspection Date", datetime.date.today())

    inspection_year = inspection_date.year
    inspection_month = inspection_date.month
    inspection_day_of_week = inspection_date.weekday()

    input_dict = {
        'INSPECTION TYPE': [inspection_type],
        'CRITICAL FLAG': [critical_flag],
        'VIOLATION CODE': [violation_code],
        'SCORE': [score],
        'inspection_year': [inspection_year],
        'inspection_month': [inspection_month],
        'inspection_day_of_week': [inspection_day_of_week]
    }
    input_df = pd.DataFrame(input_dict)

    for feature in ['INSPECTION TYPE', 'CRITICAL FLAG', 'VIOLATION CODE']:
        le = label_encoders[feature]
        input_df[feature] = le.transform(input_df[feature].astype(str))

    input_scaled = scaler.transform(input_df)

    if st.button("Predict Grade"):
        pred_encoded = model.predict(input_scaled)[0]
        predicted_grade = grade_encoder.inverse_transform([pred_encoded])[0]
        st.markdown(summary_box(predicted_grade, "Customer"), unsafe_allow_html=True)
        st.markdown("### ✅ Predicted Restaurant Grade:")
        st.markdown(grade_badge(predicted_grade), unsafe_allow_html=True)
        st.write(get_grade_message(predicted_grade))

# ------------------------------
# RESTAURANT OWNER ROLE
# ------------------------------
elif role == "Restaurant Owner":
    st.header("🏢 Restaurant Owner Portal")
    st.write("As a restaurant owner, enter your inspection details to predict the grade and see improvement guidance.")

    restaurant_name = st.text_input("Restaurant Name", "Pizza Place in Brooklyn")
    inspection_type = st.selectbox("Inspection Type", label_encoders['INSPECTION TYPE'].classes_)
    critical_flag = st.selectbox("Critical Flag", label_encoders['CRITICAL FLAG'].classes_)
    violation_code = st.selectbox("Violation Code", label_encoders['VIOLATION CODE'].classes_)
    score = st.number_input("Self-audited Score", min_value=0, max_value=100, value=18)
    inspection_date = st.date_input("Scheduled Inspection Date", datetime.date.today())

    inspection_year = inspection_date.year
    inspection_month = inspection_date.month
    inspection_day_of_week = inspection_date.weekday()

    input_dict = {
        'INSPECTION TYPE': [inspection_type],
        'CRITICAL FLAG': [critical_flag],
        'VIOLATION CODE': [violation_code],
        'SCORE': [score],
        'inspection_year': [inspection_year],
        'inspection_month': [inspection_month],
        'inspection_day_of_week': [inspection_day_of_week]
    }
    input_df = pd.DataFrame(input_dict)

    for feature in ['INSPECTION TYPE', 'CRITICAL FLAG', 'VIOLATION CODE']:
        le = label_encoders[feature]
        input_df[feature] = le.transform(input_df[feature].astype(str))

    input_scaled = scaler.transform(input_df)

    if st.button("Predict as Owner"):
        pred_encoded = model.predict(input_scaled)[0]
        predicted_grade = grade_encoder.inverse_transform([pred_encoded])[0]

        st.markdown(summary_box(predicted_grade, "Restaurant Owner"), unsafe_allow_html=True)
        st.markdown(f"### 📊 Predicted Grade for **{restaurant_name}**:")
        st.markdown(grade_badge(predicted_grade), unsafe_allow_html=True)
        rec = owner_recommendations(predicted_grade)
        st.subheader("💡 Recommendations")
        st.write(rec)

        # CSV Download
        result_df = pd.DataFrame({
            "Restaurant": [restaurant_name],
            "Predicted Grade": [predicted_grade],
            "Score": [score],
            "Critical Flag": [critical_flag],
            "Recommendation": [rec]
        })
        csv_data = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report as CSV", csv_data, "owner_report.csv", "text/csv")

        # PDF Download
        pdf_buffer = generate_styled_pdf_report(
            "Restaurant Owner Report",
            {
                "Restaurant": restaurant_name,
                "Predicted Grade": predicted_grade,
                "Score": score,
                "Critical Flag": critical_flag,
                "Scheduled Month": inspection_month
            },
            recommendations=rec
        )
        st.download_button("📥 Download Report as PDF", pdf_buffer, "owner_report.pdf", "application/pdf")

# ------------------------------
# HEALTH AUTHORITY ROLE
# ------------------------------
elif role == "Health Authority":
    st.header("🏛️ Health Authority Dashboard")
    st.write("Enter inspection details to simulate an inspection prediction and help prioritize efforts.")

    location = st.text_input("Location / Borough", "Queens")
    cuisine = st.text_input("Cuisine Type", "Chinese")
    inspection_type = st.selectbox("Inspection Type", label_encoders['INSPECTION TYPE'].classes_)
    critical_flag = st.selectbox("Critical Flag", label_encoders['CRITICAL FLAG'].classes_)
    violation_code = st.selectbox("Violation Code", label_encoders['VIOLATION CODE'].classes_)
    score = st.number_input("Inspection Score", min_value=0, max_value=100, value=32)
    inspection_date = st.date_input("Planned Inspection Date", datetime.date.today())

    inspection_year = inspection_date.year
    inspection_month = inspection_date.month
    inspection_day_of_week = inspection_date.weekday()

    input_dict = {
        'INSPECTION TYPE': [inspection_type],
        'CRITICAL FLAG': [critical_flag],
        'VIOLATION CODE': [violation_code],
        'SCORE': [score],
        'inspection_year': [inspection_year],
        'inspection_month': [inspection_month],
        'inspection_day_of_week': [inspection_day_of_week]
    }
    input_df = pd.DataFrame(input_dict)

    for feature in ['INSPECTION TYPE', 'CRITICAL FLAG', 'VIOLATION CODE']:
        le = label_encoders[feature]
        input_df[feature] = le.transform(input_df[feature].astype(str))

    input_scaled = scaler.transform(input_df)

    if st.button("Predict as Health Authority"):
        pred_encoded = model.predict(input_scaled)[0]
        predicted_grade = grade_encoder.inverse_transform([pred_encoded])[0]

        st.markdown(summary_box(predicted_grade, "Health Authority"), unsafe_allow_html=True)
        st.markdown(f"### 🏷️ Predicted Grade:")
        st.markdown(grade_badge(predicted_grade), unsafe_allow_html=True)
        act = authority_actions(predicted_grade)
        st.subheader("📌 Suggested Action")
        st.write(act)

        # CSV Download
        result_df = pd.DataFrame({
            "Location": [location],
            "Cuisine": [cuisine],
            "Predicted Grade": [predicted_grade],
            "Score": [score],
            "Critical Flag": [critical_flag],
            "Action": [act]
        })
        csv_data = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report as CSV", csv_data, "authority_report.csv", "text/csv")

        # PDF Download
        pdf_buffer = generate_styled_pdf_report(
            "Health Authority Report",
            {
                "Location": location,
                "Cuisine": cuisine,
                "Predicted Grade": predicted_grade,
                "Score": score,
                "Critical Flag": critical_flag,
                "Planned Month": inspection_month
            },
            recommendations=act
        )
        st.download_button("📥 Download Report as PDF", pdf_buffer, "authority_report.pdf", "application/pdf")
