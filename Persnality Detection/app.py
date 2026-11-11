
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Personality Predictor",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# PERSONALITY MAPPING
# ==========================================

personality_mapping = {
    0: "Ambivert",
    1: "Extrovert",
    2: "Introvert"
}

# ==========================================
# LOAD YOUR ACTUAL PICKLE FILES
# ==========================================

@st.cache_resource
def load_files():

    with open("personality_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_files()

# ==========================================
# FEATURE SELECTION
# ==========================================

# These three columns were excluded during
# model training.

excluded_features = [
    "creativity",
    "emotional_stability",
    "stress_handling"
]

# Get original feature names from your scaler
features = [
    col for col in scaler.feature_names_in_
    if col not in excluded_features
]

# Verify feature count
if len(features) != model.n_features_in_:
    st.error(
        f"Feature mismatch! Model expects "
        f"{model.n_features_in_} features, "
        f"but app has {len(features)}."
    )
    st.stop()

# Find the corresponding scaler positions
feature_indices = [
    list(scaler.feature_names_in_).index(col)
    for col in features
]

# ==========================================
# HEADER
# ==========================================

st.title("🧠 Personality Predictor")

st.subheader(
    "Discover Your Personality Through Machine Learning"
)

st.write(
    "Answer the questions below based on your "
    "habits, behaviour and preferences."
)

st.info(
    "Rate yourself from 1 to 10. "
    "1 = Lowest | 10 = Highest"
)

# ==========================================
# INPUT FEATURES
# ==========================================

feature_groups = {

    "👥 Social Behaviour": [
        "social_energy",
        "alone_time_preference",
        "talkativeness",
        "group_comfort",
        "party_liking",
        "listening_skill",
        "friendliness",
        "public_speaking_comfort",
        "online_social_usage"
    ],

    "🧠 Thinking & Behaviour": [
        "deep_reflection",
        "empathy",
        "organization",
        "leadership",
        "risk_taking",
        "curiosity",
        "decision_speed"
    ],

    "🎯 Lifestyle & Interests": [
        "routine_preference",
        "excitement_seeking",
        "planning",
        "spontaneity",
        "adventurousness",
        "reading_habit",
        "sports_interest",
        "travel_desire",
        "gadget_usage",
        "work_style_collaborative"
    ]
}

input_values = {}

for group, columns in feature_groups.items():

    visible_columns = [
        col for col in columns if col in features
    ]

    if not visible_columns:
        continue

    st.markdown("---")
    st.subheader(group)

    cols = st.columns(2)

    for i, feature in enumerate(visible_columns):

        with cols[i % 2]:

            label = feature.replace("_", " ").title()

            input_values[feature] = st.slider(
                label,
                min_value=1,
                max_value=10,
                value=5,
                key=feature
            )

# ==========================================
# PREDICTION
# ==========================================

st.markdown("---")

if st.button(
    "🔮 Predict My Personality",
    type="primary",
    use_container_width=True
):

    try:

        # Create input in exact model feature order
        input_df = pd.DataFrame(
            [[input_values[col] for col in features]],
            columns=features
        )

        # Select corresponding scaler statistics
        means = scaler.mean_[feature_indices]
        scales = scaler.scale_[feature_indices]

        # Scale the 26 features using the saved
        # statistics from your original scaler
        scaled_input = (
            input_df.to_numpy(dtype=float) - means
        ) / scales

        # Predict personality
        prediction = model.predict(scaled_input)[0]

        personality = personality_mapping[int(prediction)]

        # ==================================
        # DISPLAY RESULT
        # ==================================

        st.success("Prediction Completed!")

        st.markdown("## 🎯 Your Predicted Personality")

        st.markdown(
            f"""
            <div style="
                background-color: #163B32;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                margin: 15px 0;
            ">
                <h1 style="
                    color: #50E3A4;
                    font-size: 42px;
                ">
                    {personality}
                </h1>
                <p style="color:white;">
                    Your predicted personality type
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ==================================
        # PROBABILITY RESULTS
        # ==================================

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                scaled_input
            )[0]

            st.markdown("## 📊 Prediction Confidence")

            results = []

            for class_number, probability in zip(
                model.classes_, probabilities
            ):

                name = personality_mapping[int(class_number)]

                results.append({
                    "Personality Type": name,
                    "Probability": probability * 100
                })

            probability_df = pd.DataFrame(results)

            probability_df = probability_df.sort_values(
                by="Probability",
                ascending=False
            )

            for _, row in probability_df.iterrows():

                st.write(
                    f"**{row['Personality Type']} — "
                    f"{row['Probability']:.2f}%**"
                )

                st.progress(
                    float(row["Probability"] / 100)
                )

            st.markdown("### Detailed Results")

            display_df = probability_df.copy()

            display_df["Probability"] = display_df[
                "Probability"
            ].apply(lambda x: f"{x:.2f}%")

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "This is an experimental machine learning "
            "prediction, not a psychological diagnosis."
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")
