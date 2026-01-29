from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def find_best_params(model, grid_param, cv, scoring, X_train, y_train):
    
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)

    grid_search.fit(X_train, y_train)
    
    return grid_search.best_param_

def evaluate_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }
    
    return metrics
    
    
    
    