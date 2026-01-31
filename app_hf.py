import pandas as pd
import gradio as gr
import joblib

from huggingface_hub import hf_hub_download


# Download model from HuggingFace Hub
MODEL_REPO = "Hiyansh005/insurance-ml-model"
FILENAME = "insurance_pipeline.pkl"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=FILENAME
)

# Load pipeline
ct, model = joblib.load(model_path)


# Prediction function
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


# Gradio Interface
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
