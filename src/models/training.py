import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor

def main():
    
    X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv")
    
    best_params = load("models/best_params.pkl")
    
    rfr = RandomForestRegressor(**best_params, random_state=1664)
    
    rfr.fit(X_train_scaled, y_train)

    dump(rfr, "models/trained_model.pkl")
    
if __name__ == "__main__":
    main()