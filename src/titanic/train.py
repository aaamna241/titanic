"""
Train the Titanic model.
"""
from sklearn.linear_model import LogisticRegression



def train_model(tuple_train:tuple) :
    """ 
    Initiate the model and train it on the Titanic dataset.""
    """

    lin = LogisticRegression(max_iter=1000, random_state=42)
    lin.fit(tuple_train[0],tuple_train[1])
    
    return lin


def evaluate_model(lin, tuple_test:tuple):
    return lin.score(tuple_test[0], tuple_test[1])


 

def optimize_model():
    pass

