import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load models
rf_model = joblib.load('models/rf_model.pkl')
c45_model = joblib.load('models/c45_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')

# Load data
df = pd.read_csv("dataset.csv")

# Menambahkan target 'UCL_Eligible'
df['UCL_Eligible'] = df['LgRk'].apply(lambda x: 1 if x <= 4 else 0)
features = ['Pts/G', 'xG', 'xGA', 'xGD', 'xGD/90', 'W']
X = df[features]
y = df['UCL_Eligible']

ratios = [0.1, 0.2, 0.3]  # Rasio data (90:10, 80:20, 70:30)

st.title("Evaluasi Model")
st.markdown("Evaluasi model berdasarkan rasio data: **90:10**, **80:20**, dan **70:30**.")

# Loop over different data split ratios
for ratio in ratios:
    st.markdown(f"### Rasio Data **{int((1 - ratio) * 100)}:{int(ratio * 100)}**")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)

    # Define models
    models = [
        ("Random Forest", RandomForestClassifier(random_state=42)),
        ("C4.5", DecisionTreeClassifier(criterion='entropy', random_state=42)),
        ("XGBoost", XGBClassifier(eval_metric='logloss', random_state=42))
    ]

    for name, model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Display results in a card-like style
        with st.expander(f"Evaluasi Model {name} (Rasio {int((1 - ratio) * 100)}:{int(ratio * 100)})"):
            st.write(f"**Model:** {name}")
            st.write(f"- Akurasi: {accuracy_score(y_test, predictions):.2f}")
            st.write(f"- F1 Score: {f1_score(y_test, predictions):.2f}")
            st.write(f"- Presisi: {precision_score(y_test, predictions):.2f}")
            st.write(f"- Recall: {recall_score(y_test, predictions):.2f}")
            st.write(f"- AUC: {roc_auc_score(y_test, y_prob):.2f}")

            # Confusion Matrix
            cm = confusion_matrix(y_test, predictions)
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", linewidths=0.5, ax=ax)
            ax.set_xlabel('Prediksi')
            ax.set_ylabel('Aktual')
            ax.set_title(f"Confusion Matrix - {name}")
            st.pyplot(fig)

            if name == "Random Forest":
                st.markdown("#### Feature Importance (Random Forest)")
                importance = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': importance
                }).sort_values(by='Importance', ascending=False)

                st.dataframe(feature_importance)

                # Bar chart for feature importance
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis", ax=ax)
                ax.set_title("Feature Importance - Random Forest")
                st.pyplot(fig)

            elif name == "C4.5":
                st.markdown("#### Root dan Struktur Pohon Keputusan")
                st.write(f"**Root:** {features[model.tree_.feature[0]]}")
                fig, ax = plt.subplots(figsize=(12, 8))
                plot_tree(model, feature_names=features, class_names=["Tidak Layak", "Layak"], filled=True, ax=ax)
                st.pyplot(fig)
