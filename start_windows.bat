@echo off
python -m venv venv
call venv\Scripts\activate
python -m pip install -r requirements.txt
if not exist model\weather_outfit_classifier.joblib python train_model.py --csv data\weather_outfit_classification.csv
python app.py
