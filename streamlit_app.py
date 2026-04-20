import io
import math
import wave
from pathlib import Path

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Customer Retention and Feedback Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)


APP_TITLE = "Customer Retention and Feedback Intelligence Dashboard"
APP_SUBTITLE = (
    "A business-friendly dashboard that predicts customer churn, "
    "analyzes review sentiment, and evaluates audio sentiment."
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@st.cache_resource
def load_artifacts():
    artifacts_path = Path(__file__).parent / "dashboard_artifacts.joblib"
    return joblib.load(artifacts_path)


def sentiment_rule_score(text, positive_words, negative_words):
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    tokens = [tok for tok in cleaned.split() if tok]
    pos_hits = [tok for tok in tokens if tok in positive_words]
    neg_hits = [tok for tok in tokens if tok in negative_words]
    score = len(pos_hits) - len(neg_hits)
    return score, pos_hits, neg_hits


def churn_probability(form_data):
    contract_weights = {
        "Month-to-month": 0.70,
        "One year": -0.15,
        "Two year": -0.55,
    }
    internet_weights = {
        "Fiber optic": 0.35,
        "DSL": 0.05,
        "No": -0.25,
    }
    payment_weights = {
        "Electronic check": 0.25,
        "Mailed check": 0.05,
        "Bank transfer (automatic)": -0.15,
        "Credit card (automatic)": -0.20,
    }

    score = -1.10
    score += 0.55 if form_data["senior_citizen"] == "Yes" else 0.0
    score += 0.45 if form_data["partner"] == "No" else -0.10
    score += 0.35 if form_data["dependents"] == "No" else -0.08
    score += 0.40 if form_data["paperless_billing"] == "Yes" else -0.05
    score += contract_weights[form_data["contract"]]
    score += internet_weights[form_data["internet_service"]]
    score += payment_weights[form_data["payment_method"]]
    score += 0.015 * min(form_data["monthly_charges"], 120)
    score -= 0.030 * min(form_data["tenure"], 72)
    score -= 0.00025 * min(form_data["total_charges"], 10000)

    contributions = {
        "Monthly Charges": 0.015 * min(form_data["monthly_charges"], 120),
        "Tenure": -0.030 * min(form_data["tenure"], 72),
        "Total Charges": -0.00025 * min(form_data["total_charges"], 10000),
        "Contract": contract_weights[form_data["contract"]],
        "Internet Service": internet_weights[form_data["internet_service"]],
        "Payment Method": payment_weights[form_data["payment_method"]],
        "Paperless Billing": 0.40 if form_data["paperless_billing"] == "Yes" else -0.05,
        "Senior Citizen": 0.55 if form_data["senior_citizen"] == "Yes" else 0.0,
        "Partner": 0.45 if form_data["partner"] == "No" else -0.10,
        "Dependents": 0.35 if form_data["dependents"] == "No" else -0.08,
    }

    probability = float(sigmoid(score))
    label = "Churn Risk" if probability >= 0.50 else "Low Churn Risk"
    return label, probability, contributions


def plot_contributions(contributions):
    s = pd.Series(contributions).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(s.index, s.values)
    ax.set_title("Feature Contribution to Churn Prediction")
    ax.set_xlabel("Impact on churn score")
    ax.axvline(0, linewidth=1)
    plt.tight_layout()
    return fig


def plot_sentiment_hits(pos_hits, neg_hits):
    counts = pd.Series(
        {
            "Positive word hits": len(pos_hits),
            "Negative word hits": len(neg_hits),
        }
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values)
    ax.set_title("Text Explainability")
    ax.set_ylabel("Count")
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig


def read_wav_bytes(file_bytes):
    with wave.open(io.BytesIO(file_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    if sampwidth not in dtype_map:
        raise ValueError("Unsupported WAV sample width.")

    signal = np.frombuffer(raw, dtype=dtype_map[sampwidth]).astype(np.float32)

    if n_channels > 1:
        signal = signal.reshape(-1, n_channels).mean(axis=1)

    max_abs = np.max(np.abs(signal)) if len(signal) else 1.0
    if max_abs > 0:
        signal = signal / max_abs

    return signal, framerate


def audio_features_from_wav(file_bytes):
    signal, sr = read_wav_bytes(file_bytes)
    if len(signal) == 0:
        raise ValueError("Uploaded WAV file is empty.")

    rms = float(np.sqrt(np.mean(np.square(signal))))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=signal)[0]))
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_mean = float(np.mean(mfcc))
    duration = len(signal) / sr

    # Simple business-friendly rule:
    # higher energy and sharper variation = more likely upset / negative
    upset_score = (rms * 2.4) + (zcr * 3.0) - (mfcc_mean / 100)
    label = "Negative / Upset Tone" if upset_score >= 0.40 else "Positive / Calm Tone"

    features = {
        "RMS Energy": rms,
        "Zero Crossing Rate": zcr,
        "MFCC Mean": mfcc_mean,
        "Duration (sec)": duration,
        "Upset Score": upset_score,
    }
    return label, features, signal, sr


def plot_waveform(signal, sr):
    times = np.arange(len(signal)) / sr
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(times, signal)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig


def main():
    artifacts = load_artifacts()
    positive_words = artifacts["positive_words"]
    negative_words = artifacts["negative_words"]

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    st.markdown(
        """
        This dashboard is designed for business users. It combines **customer churn prediction**,
        **text sentiment analysis**, and **audio sentiment analysis** into one easy workflow.
        """
    )

    tab1, tab2, tab3 = st.tabs(
        ["Customer Churn", "Text Sentiment", "Audio Sentiment"]
    )

    with tab1:
        st.subheader("Customer Churn Prediction")
        st.write(
            "Use customer account details to estimate churn risk and identify the factors driving the prediction."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 0.0, 150.0, 75.0, 1.0)
            total_charges = st.slider("Total Charges ($)", 0.0, 10000.0, 1200.0, 10.0)
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )
        with col2:
            internet_service = st.selectbox(
                "Internet Service",
                ["Fiber optic", "DSL", "No"],
            )
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        with col3:
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        if st.button("Predict Churn", use_container_width=True):
            form_data = {
                "tenure": tenure,
                "monthly_charges": monthly_charges,
                "total_charges": total_charges,
                "contract": contract,
                "internet_service": internet_service,
                "payment_method": payment_method,
                "paperless_billing": paperless_billing,
                "senior_citizen": senior_citizen,
                "partner": partner,
                "dependents": dependents,
            }

            label, probability, contributions = churn_probability(form_data)

            st.metric("Predicted Outcome", label)
            st.metric("Churn Probability", f"{probability:.1%}")

            if probability >= 0.50:
                st.warning(
                    "This customer appears more likely to leave. A retention offer or service follow-up may be appropriate."
                )
            else:
                st.success(
                    "This customer appears less likely to churn based on the selected profile."
                )

            fig = plot_contributions(contributions)
            st.pyplot(fig)

    with tab2:
        st.subheader("Text Sentiment Analysis")
        st.write(
            "Paste a customer review, survey response, or complaint to estimate sentiment."
        )

        text_input = st.text_area(
            "Customer Review",
            placeholder="Example: The service was fast and the staff was helpful, but billing was confusing.",
            height=180,
        )

        if st.button("Analyze Text", use_container_width=True):
            if not text_input.strip():
                st.error("Please enter some text before running the analysis.")
            else:
                score, pos_hits, neg_hits = sentiment_rule_score(
                    text_input, positive_words, negative_words
                )
                label = "Positive" if score >= 0 else "Negative"

                st.metric("Predicted Sentiment", label)
                st.metric("Rule-Based Score", score)

                if pos_hits:
                    st.write("Positive terms found:", ", ".join(sorted(set(pos_hits))))
                if neg_hits:
                    st.write("Negative terms found:", ", ".join(sorted(set(neg_hits))))
                if not pos_hits and not neg_hits:
                    st.info(
                        "No strong positive or negative words were detected, so the prediction is based on a neutral rule score."
                    )

                fig = plot_sentiment_hits(pos_hits, neg_hits)
                st.pyplot(fig)

    with tab3:
        st.subheader("Audio Sentiment Analysis")
        st.write(
            "Upload a short WAV file to estimate whether the speaker sounds calm/positive or upset/negative."
        )

        audio_file = st.file_uploader("Upload WAV file", type=["wav"])

        if audio_file is not None:
            st.audio(audio_file)

        if st.button("Analyze Audio", use_container_width=True):
            if audio_file is None:
                st.error("Please upload a WAV file before running the analysis.")
            else:
                try:
                    file_bytes = audio_file.getvalue()
                    label, features, signal, sr = audio_features_from_wav(file_bytes)

                    st.metric("Predicted Audio Sentiment", label)
                    st.metric("Upset Score", f"{features['Upset Score']:.3f}")

                    feature_df = pd.DataFrame(
                        {
                            "Feature": list(features.keys()),
                            "Value": [round(v, 4) for v in features.values()],
                        }
                    )
                    st.dataframe(feature_df, use_container_width=True)

                    fig = plot_waveform(signal, sr)
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Could not read the WAV file: {e}")

    st.markdown("---")
    st.markdown(
        """
        **Business narrative:**  
        This app helps managers identify churn risk, summarize written feedback,
        and evaluate the tone of customer audio. By packaging different models into one dashboard,
        the workflow becomes more accessible for non-technical stakeholders.
        """
    )


if __name__ == "__main__":
    main()
