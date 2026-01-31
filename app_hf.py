import joblib
import pandas as pd
import gradio as gr

# Load encoder + model
ct, model = joblib.load("insurance_pipeline.pkl")


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
