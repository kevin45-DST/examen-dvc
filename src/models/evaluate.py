import pandas as pd
import json
import model_utils
from joblib import load

def main():
    
    X_test_scaled = pd.read_csv("data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed_data/y_test.csv")
       
    trained_model = load("models/trained_model.pkl")
    
    metrics = model_utils.evaluate_model(trained_model, X_test_scaled, y_test)
    
    with open("metrics/scores.json", "w+") as f:
        json.dump(metrics, f)
    
if __name__ == "__main__":
    main()