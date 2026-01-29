import pandas as pd

from data import preprocessed_utils

def main():
    
    X_train = pd.read_csv("data/processed_data/X_train.csv")
    X_test = pd.read_csv("data/processed_data/X_test.csv")
    
    X_train_scaled, X_test_scaled = preprocessed_utils.normalize_data(X_train, X_test)

    X_train_scaled.to_csv("data/processed_data/X_train_scaled.csv")
    X_test_scaled.to_csv("data/processed_data/X_test_scaled.csv")   
    
if __name__ == "__main__":
    main()