import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import pickle
import sklearn
from sklearn.preprocessing import LabelEncoder

st.header("Calorie burnt prediction")
col1, col2 = st.columns(2)
gender = option_menu(
    "Gender", 
    ["Male", "Female"], 
    icons=["gender-male", "gender-female"],
    orientation="horizontal"
)

g = -1
if gender=="Male":
    g=0
else:
    g=1
with col2:
    Age=st.number_input("Age")
with col1:
    Height=st.number_input("Height")
with col2:
    Weight=st.number_input("Weight")
with col1:
    Duration=st.number_input("Duration")
with col2:
    Heart_Rate=st.number_input("Heart_Rate")
with col1:
    Body_temp=st.number_input("Body_temp")

def add_bmi_and_intensity(df):
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['Intensity'] = df['Heart_Rate'] / df['Duration']
    return df

def add_age_sex_interactions(df):
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 2})
    df['AgeSex'] = df['Age'].astype(str) + df['Sex'].astype(str)
    df['AgeSex'] = LabelEncoder().fit_transform(df['AgeSex']) + 1
    return df

def add_feature_interactions(df, numeric_cols):
    for i in range(len(numeric_cols)):
        feature_1 = numeric_cols[i]
        for j in range(i + 1, len(numeric_cols)):
            feature_2 = numeric_cols[j]
            df[f'{feature_1}_x_{feature_2}'] = df[feature_1] * df[feature_2]
    return df
def add_duration_category(df):
    df['Duration_cat'] = pd.cut(df['Duration'], bins=10, labels=False, right=False)
    return df

def full_feature_engineering(df):
    numeric_cols = ['Age', 'Weight', 'Height', 'Body_Temp', 'Heart_Rate', 'Duration', 'Sex', 'AgeSex']
    
    df = add_bmi_and_intensity(df)
    df = add_age_sex_interactions(df)
    df = add_feature_interactions(df, numeric_cols)
    df['Sex'] = df['Sex'].astype('category')
    df = add_duration_category(df)
    
    return df

def prepare_features():
    sex_str = "male" if gender == "Male" else "female"
    data = {
        "Sex": sex_str,  # Important: match to expected 'Sex' string
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "Duration": Duration,
        "Heart_Rate": Heart_Rate,
        "Body_Temp": Body_temp,
    }
    df = pd.DataFrame(data, index=[0])
    df = full_feature_engineering(df)
    return df


class MyEnsembleModel:
    def __init__(self, cat_model, xgb_model, lgb_model, weights=(0.30, 0.60, 0.10)):
        self.cat_model = cat_model
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.weights = weights

    def predict(self, X):
        cat_pred = np.expm1(self.cat_model.predict(X))
        xgb_pred = np.expm1(self.xgb_model.predict(X))
        lgb_pred = np.expm1(self.lgb_model.predict(X))

        final_pred = (
            cat_pred * self.weights[0]
            + xgb_pred * self.weights[1]
            + lgb_pred * self.weights[2]
        )

        final_pred = np.clip(final_pred, 1, 314)

        return final_pred

# Load model (assuming you saved the ensemble or best_model)
model = pickle.load(open("model.pkl", "rb"))

# Predict button
if st.button("Predict"):
    df = prepare_features()
    prediction = model.predict(df)
    st.header(f"Predicted Calories Burned: {prediction[0]:.2f} kcal")