import gradio as gr
import pandas as pd
import joblib
import requests

# Download model from GitHub raw
MODEL_URL = "https://raw.githubusercontent.com/Hiyansh145/insurance-ML-api/main/insurance_pipeline.pkl"
MODEL_FILE = "insurance_pipeline.pkl"

# Download once
r = requests.get(MODEL_URL)
with open(MODEL_FILE, "wb") as f:
    f.write(r.content)

# Load model
ct, model = joblib.load(MODEL_FILE)


def predict(age, sex, bmi, children, smoker, region):

    data = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }

    df = pd.DataFrame([data])

    X = ct.transform(df)

    pred = model.predict(X)[0]

    return round(float(pred), 2)


app = gr.Interface(

    fn=predict,

    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown(["male", "female"], label="Sex"),
        gr.Number(label="BMI"),
        gr.Number(label="Children"),
        gr.Dropdown(["yes", "no"], label="Smoker"),
        gr.Dropdown(
            ["southeast", "southwest", "northwest", "northeast"],
            label="Region"
        )
    ],

    outputs=gr.Number(label="Predicted Charges"),

    title="Insurance Cost Predictor",
    description="ML-based Medical Insurance Cost Prediction"
)

app.launch()
