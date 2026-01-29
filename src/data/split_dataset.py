import pandas as pd
import preprocessed_utils

def main():
    
    df = pd.read_csv("data/raw_data/raw.csv")
    df.drop(columns=["date"], inplace=True)

    X_train, X_test, y_train, y_test = preprocessed_utils.split_data(df, "silica_concentrate")

    X_train.to_csv("data/processed_data/X_train.csv")
    X_test.to_csv("data/processed_data/X_test.csv")
    y_train.to_csv("data/processed_data/y_train.csv")
    y_test.to_csv("data/processed_data/y_test.csv")    
    
if __name__ == "__main__":
    main()