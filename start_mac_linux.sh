#!/usr/bin/env bash
set -e
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
if [ ! -f model/weather_outfit_classifier.joblib ]; then python train_model.py --csv data/weather_outfit_classification.csv; fi
python app.py
