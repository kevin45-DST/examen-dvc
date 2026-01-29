import pandas as pd
import model_utils
from joblib import dump
from sklearn.ensemble import RandomForestRegressor

def main():
    
    X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv")
    
    rfr = RandomForestRegressor(random_state=1664)
    grid_param = {
    "n_estimators": [200, 400],
    "max_depth": [None, 20],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 4],
    "max_features": ["sqrt", 1.0],
    }
    
    best_params = model_utils.find_best_params(rfr, 
                                                grid_param, 
                                                5, 
                                                "neg_mean_squared_error", 
                                                X_train_scaled, 
                                                y_train)


    dump(best_params, "models/best_params.pkl")
    
if __name__ == "__main__":
    main()