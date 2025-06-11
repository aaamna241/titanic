"""
Train the Titanic model.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_model(tuple_train:tuple) :
    """ 
    Initiate the model and train it on the Titanic dataset.""
    """

    lin = LogisticRegression(max_iter=1000, random_state=42)
    lin.fit(tuple_train[0],tuple_train[1])
    
    return lin


def evaluate_model(lin, tuple_test:tuple):
    return lin.score(tuple_test[0], tuple_test[1])


def optimize_model(lin, tuple_train):
    """
    Optimize the model using GridSearchCV.
    """
    X_train, y_train = tuple_train

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    grid_search = GridSearchCV(lin, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_, "Best score:", grid_search.best_score_)
    
    return grid_search.best_estimator_ 

