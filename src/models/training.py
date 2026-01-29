import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor

def main():
    
    X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv")
    
    rfr = RandomForestRegressor(random_state=1664)
    
    best_params = load("models/best_params.pkl")
    
    rfr.fit(X_train_scaled, y_train, best_params)

    dump(rfr, "models/trained_model.pkl")
    
if __name__ == "__main__":
    main()