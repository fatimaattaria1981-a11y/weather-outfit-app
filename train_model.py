from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Clothing_Category"


def main(csv_path):
    frame = pd.read_csv(csv_path).drop_duplicates().dropna(subset=[TARGET])
    X, y = frame.drop(columns=[TARGET]), frame[TARGET].astype(str)
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([
        ("preprocessor", preprocess),
        ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(classification_report(y_test, predictions, zero_division=0))
    output = Path(__file__).parent / "model" / "weather_outfit_classifier.joblib"
    output.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "input_columns": list(X.columns), "target": TARGET}, output)
    print(f"Model saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the training CSV")
    main(parser.parse_args().csv)
