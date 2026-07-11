# WeatherWear AI Lab — Full-Stack Workshop App

This Flask application turns the weather outfit classification model into an interactive classroom dashboard. It supports one-row prediction, probability visualization, student-friendly explanation, and batch CSV prediction.

## Easiest Windows setup

1. Extract the ZIP.
2. Make sure Python 3.11 or newer is installed. During installation, tick **Add Python to PATH**.
3. Double-click `start_windows.bat`.
4. Wait for `Running on http://127.0.0.1:5000`.
5. Open `http://127.0.0.1:5000` in Chrome.

The first launch installs packages and trains the model, so it takes longer. Later launches can use these commands:

```bat
venv\Scripts\activate
python app.py
```

## Windows commands manually

```bat
cd weather_outfit_workshop_app
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python train_model.py --csv data\weather_outfit_classification.csv
python app.py
```

## macOS/Linux commands

```bash
cd weather_outfit_workshop_app
chmod +x start_mac_linux.sh
./start_mac_linux.sh
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python train_model.py --csv data/weather_outfit_classification.csv
python app.py
```

## Use your already-trained Colab model

Copy your downloaded `weather_outfit_classifier.joblib` into the `model` folder. Its expected location is:

```text
model/weather_outfit_classifier.joblib
```

The Colab notebook created with this project saves a compatible bundle containing `model`, `input_columns`, and `target`.

## Project structure

```text
app.py                 Flask backend and API
train_model.py         Training script
templates/index.html   Frontend page
static/style.css       Responsive visual design
static/app.js          API calls and Chart.js graph
data/                  Training CSV
model/                 Saved model location
requirements.txt       Python packages
render.yaml            Render deployment configuration
Procfile               Gunicorn production command
```

## API routes

- `GET /` — dashboard
- `GET /api/health` — model status
- `POST /api/predict` — one weather scenario
- `POST /api/predict-file` — batch CSV predictions

## Deploy on Render

1. Put the project in a GitHub repository.
2. Make sure the trained `.joblib` file is inside the `model` folder and committed.
3. On Render, choose **New > Blueprint** and connect the repository.
4. Render reads `render.yaml`; click deploy.

Production command:

```bash
gunicorn app:app
```

Important: do not use Flask debug mode for a public production deployment. The `Procfile` and `render.yaml` already use Gunicorn.

## Teaching note

The probability bars are class probabilities estimated by the model. The biggest probability becomes the output class. The explanation is a readable summary of the input and prediction; it is not a mathematical feature-importance calculation.
