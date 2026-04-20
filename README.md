# Assignment 5 App
## Title
Customer Retention and Feedback Intelligence Dashboard

## Overview
This Streamlit dashboard integrates multiple business-oriented prediction workflows into one app:
1. Customer churn prediction using tabular customer inputs
2. Text sentiment analysis using customer review text
3. Audio sentiment analysis using uploaded WAV files

The app is designed for non-technical stakeholders and includes simple explainability visuals.

## Files Included
- `streamlit_app.py` — main dashboard app
- `requirements.txt` — Python packages needed to run the app
- `dashboard_artifacts.joblib` — supporting word lists used by the text sentiment section
- `README.md` — instructions for the grader

## Run Instructions
Install the dependencies, then run:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Tab Guide
### Tab 1: Customer Churn
Enter customer account information such as tenure, monthly charges, contract type, and payment method. The app returns a churn prediction, churn probability, and a feature contribution chart.

### Tab 2: Text Sentiment
Paste a customer review or comment. The app predicts positive or negative sentiment and shows a simple explainability chart based on positive and negative word hits.

### Tab 3: Audio Sentiment
Upload a `.wav` audio file. The app estimates whether the speaker sounds calm/positive or upset/negative, then displays extracted audio metrics and a waveform plot.

## Notes
- This project is self-contained and intended to run directly from the submitted ZIP.
- The audio upload expects a WAV file.
- The app focuses on business communication, usability, and explainability.
