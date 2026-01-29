from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_data(df, target: str, test_size=0.2, random_state=42):
    
    y = df[target]
    
    X = df.drop(columns=[target])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


def normalize_data(X_train, X_test):
    
    std_scaler = StandardScaler()
    
    X_train_scaled = std_scaler.fit_transform(X_train)

    X_test_scaled = std_scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


    

