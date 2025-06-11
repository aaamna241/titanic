"""
Save and load models, preprocessors
"""
import pickle, os

def save_model(model, path: str):
    """
    Save a model to the specified path.
    
    Args:
        model: The model to save.
        path (str): The file path where the model will be saved.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f"{path}/best_logistic_model.pkl", "wb") as f:
        pickle.dump(model, f)
   

    

def load_model(path: str):
    """
    Load a model from the specified path.
    
    Args:
        path (str): The file path from which the model will be loaded.
        
    Returns:
        The loaded model.
    """
    with open(f"{path}/best_logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model