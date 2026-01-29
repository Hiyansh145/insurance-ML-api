from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load pipeline (encoder + model)
ct,model = joblib.load("insurance_pipeline.pkl")


@app.route("/")
def home():
    return "Insurance ML API is Running!"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # Convert JSON to DataFrame (VERY IMPORTANT)
    input_df = pd.DataFrame([data])

    X = ct.transform(input_df)

    prediction = model.predict(X)[0]


    return jsonify({
        "predicted_charges": round(float(prediction), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
