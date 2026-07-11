from pathlib import Path
import io
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "weather_outfit_classifier.joblib"
TARGET = "Clothing_Category"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

bundle = None


def load_bundle():
    global bundle
    if bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Run: python train_model.py --csv data/weather_outfit_classification.csv")
        bundle = joblib.load(MODEL_PATH)
    return bundle


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    try:
        data = load_bundle()
        return jsonify({"status": "ready", "classes": list(data["model"].classes_)})
    except Exception as exc:
        return jsonify({"status": "setup_required", "message": str(exc)}), 503


@app.post("/api/predict")
def predict():
    try:
        data = load_bundle()
        values = request.get_json(force=True)
        defaults = {
            "Feels_Like_C": values.get("Temperature_C", 24), "Humidity_Percent": 55,
            "Wind_kmh": 10, "Precipitation_mm": 0, "Rain_Chance_Percent": 10,
            "UV_Index": 4, "Time_of_Day": "Afternoon", "Activity_Level": "Moderate"
        }
        values = {**defaults, **values}
        frame = pd.DataFrame([values], columns=data["input_columns"])
        prediction = data["model"].predict(frame)[0]
        probabilities = data["model"].predict_proba(frame)[0]
        classes = data["model"].classes_
        ranked = sorted(
            [{"label": str(label), "probability": round(float(prob), 4)} for label, prob in zip(classes, probabilities)],
            key=lambda item: item["probability"], reverse=True
        )
        outfit = outfit_suggestion(str(prediction), values.get("Weather_Condition", "Sunny"), values.get("Occasion", "Casual"))
        return jsonify({
            "prediction": str(prediction),
            "outfit": outfit,
            "confidence": ranked[0]["probability"],
            "probabilities": ranked,
            "explanation": explanation(values, str(prediction))
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


def outfit_suggestion(category, condition, occasion):
    suggestions = {
        "Hot Weather": {"icon": "☀️", "title": "Fresh Summer Look", "items": ["Breathable cotton T-shirt", "Light linen trousers or shorts", "Comfortable sandals", "Sunglasses and a cap"]},
        "Warm Weather": {"icon": "🌤️", "title": "Easy Warm-Day Outfit", "items": ["Cotton shirt or airy top", "Light jeans or chinos", "Breathable sneakers", "A thin layer for later"]},
        "Light Layers": {"icon": "🍃", "title": "Smart Layered Look", "items": ["Soft full-sleeve shirt", "Light cardigan or denim jacket", "Jeans or trousers", "Closed-toe shoes"]},
        "Cozy Layers": {"icon": "🧣", "title": "Comfortable Cozy Outfit", "items": ["Warm knit sweater", "Medium-weight jacket", "Thick trousers or jeans", "Scarf and covered shoes"]},
        "Winter Layers": {"icon": "❄️", "title": "Warm Winter Layers", "items": ["Thermal inner layer", "Wool sweater or fleece", "Insulated jacket", "Warm socks and boots"]},
        "Heavy Winter": {"icon": "🧥", "title": "Heavy Cold Protection", "items": ["Thermal base layers", "Heavy sweater", "Puffer coat", "Gloves, scarf, beanie and boots"]},
    }
    result = suggestions.get(category, suggestions["Light Layers"]).copy()
    result["note"] = f"Selected for {condition.lower()} weather and a {occasion.lower()} occasion."
    if condition in {"Rainy", "Stormy"}:
        result["items"] = result["items"] + ["☔ Waterproof jacket or umbrella"]
    return result


def explanation(values, prediction):
    temperature = float(values.get("Temperature_C", 0))
    condition = values.get("Weather_Condition", "the selected weather")
    rain = float(values.get("Rain_Chance_Percent", 0))
    reasons = [f"The model compared this row with patterns learned from the training data.",
               f"At {temperature:g}°C with {condition.lower()} conditions, it selected {prediction} as the strongest class."]
    if rain >= 60:
        reasons.append(f"The rain chance is {rain:g}%, which also influenced the combination of features.")
    reasons.append("The bars show probabilities, not absolute certainty. The highest bar becomes the final prediction.")
    return reasons


@app.post("/api/predict-file")
def predict_file():
    try:
        data = load_bundle()
        uploaded = request.files.get("file")
        if not uploaded or not uploaded.filename.lower().endswith(".csv"):
            raise ValueError("Please upload a CSV file.")
        frame = pd.read_csv(uploaded)
        missing = [column for column in data["input_columns"] if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")
        features = frame[data["input_columns"]]
        frame["Predicted_Clothing_Category"] = data["model"].predict(features)
        frame["Prediction_Confidence"] = data["model"].predict_proba(features).max(axis=1).round(4)
        output = io.BytesIO()
        frame.to_csv(output, index=False)
        output.seek(0)
        return send_file(output, mimetype="text/csv", as_attachment=True, download_name="weather_outfit_predictions.csv")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
