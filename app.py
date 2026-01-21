import gradio as gr
import pandas as pd
import numpy as np
import joblib

model = joblib.load("diabetes_model.pkl")


FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def predict_diabetes(
    Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DiabetesPedigreeFunction,
    Age,
):
    
    input_df = pd.DataFrame(
        [[
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age,
        ]],
        columns=FEATURE_COLUMNS,
    )

    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    label = "Diabetic" if prediction == 1 else "Not Diabetic"

    return f"Prediction: {label}\nProbability of Diabetes: {probability:.2f}"


iface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies", value=0),
        gr.Number(label="Glucose", value=120),
        gr.Number(label="Blood Pressure", value=70),
        gr.Number(label="Skin Thickness", value=20),
        gr.Number(label="Insulin", value=80),
        gr.Number(label="BMI", value=25.0),
        gr.Number(label="Diabetes Pedigree Function", value=0.5),
        gr.Number(label="Age", value=30),
    ],
    outputs=gr.Textbox(label="Result"),
    title="🩺 Diabetes Prediction System",
    description="This app predicts whether a patient is diabetic using a pre-trained Machine Learning pipeline (scikit-learn)."
)

if __name__ == "__main__":
    iface.launch()