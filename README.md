# NYC Restaurant Health Grade Predictor 🍽️

A machine learning-powered web application that predicts NYC restaurant health grades based on various inspection criteria.

## Features

- **Multi-role Interface**: Designed for Customers, Restaurant Owners, and Health Authorities
- **ML Prediction**: Uses Random Forest model for accurate grade predictions
- **PDF Reports**: Generate detailed prediction reports
- **Interactive UI**: Built with Streamlit for easy use

## Files

- `app.py` - Main Streamlit application
- `random_forest_model.sav` - Trained machine learning model
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deployment

This app is designed to be deployed on Streamlit Cloud. Make sure all required files are in your GitHub repository.

## Model Information

The app uses a Random Forest classifier trained on NYC restaurant inspection data to predict health grades (A, B, C, etc.) based on various inspection criteria.