# 🧠 Personality Detection Using Machine Learning

A Machine Learning project that predicts personality types based on behavioural patterns, social preferences, and lifestyle characteristics using **Logistic Regression**.

The project combines data analysis, feature engineering, model training, and an interactive **Streamlit web application** to make personality predictions.

## 🚀 Project Overview

This project uses supervised machine learning to classify individuals into three personality categories:

- 🟢 **Ambivert**
- 🔵 **Extrovert**
- 🟣 **Introvert**

The model takes 26 behavioural and lifestyle-related features as input and predicts the most likely personality category.

Users can interact with the trained model through a Streamlit interface by answering questions about their habits, preferences, and behaviour.

## ✨ Features

- Personality classification using Logistic Regression.
- 26 input features for personality prediction.
- Data preprocessing and feature scaling using StandardScaler.
- Interactive Streamlit web interface.
- Prediction results displayed as personality names.
- Probability breakdown for all three personality categories.
- Saved machine learning model using Pickle.

## 🛠️ Technologies & Libraries

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | Machine learning and preprocessing |
| Logistic Regression | Classification algorithm |
| StandardScaler | Feature scaling |
| Streamlit | Interactive web application |
| Pickle | Model serialization |

## 📊 Machine Learning Workflow

The project follows a typical supervised machine learning pipeline:

1. **Data Exploration:** Understanding the dataset and its features.
2. **Data Preprocessing:** Preparing the data for machine learning.
3. **Feature Selection:** Selecting 26 relevant features for prediction.
4. **Data Splitting:** Dividing the dataset into training and testing sets.
5. **Feature Scaling:** Applying StandardScaler to standardize input features.
6. **Model Training:** Training a Logistic Regression classifier.
7. **Model Evaluation:** Evaluating the model using classification metrics.
8. **Model Serialization:** Saving the trained model and scaler using Pickle.
9. **Deployment:** Building an interactive application using Streamlit.

## 🎯 Personality Categories

The model predicts one of the following:

- **Ambivert:** A personality category associated with a combination of introverted and extroverted tendencies.
- **Extrovert:** A personality category associated with greater outward social engagement.
- **Introvert:** A personality category associated with a preference for quieter or more solitary activities.

## 📁 Project Structure

```text
personality-detection-ml/
│
├── app.py
├── personality_model.pkl
├── scaler.pkl
├── feature_names.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## 💻 Installation & Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
```

### 2. Navigate to the Project Directory

```bash
cd personality-detection-ml
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📦 Model Files

The project includes the following serialized files:

- `personality_model.pkl` – Trained Logistic Regression model.
- `scaler.pkl` – Fitted StandardScaler used for preprocessing.
- `feature_names.pkl` – Stores the feature names and their required order.

These files allow the application to make predictions without retraining the model every time.

## 🔮 Future Improvements

- Improve model performance through hyperparameter tuning.
- Experiment with additional classification algorithms.
- Add more detailed personality insights.
- Improve UI/UX and interactive visualizations.
- Deploy the application publicly using Streamlit Community Cloud.

## ⚠️ Disclaimer

This project is an educational machine learning experiment. Its predictions are based on patterns learned from the dataset and should not be considered scientifically validated personality assessments or psychological diagnoses.

## 👨‍💻 Author

Developed as a personal Machine Learning project to explore supervised learning, classification algorithms, data preprocessing, and model deployment.

**If you find this project interesting, consider giving the repository a ⭐!**
