from utils.data_loader import load_data
from utils.data_processor import process_data
from utils.predictor import report
from models.models import train_model
import os

def create_model(model_name, vectorizer_name, name_data, retrain):
    
    if os.path.exists("saved_models/" + model_name + "_" + vectorizer_name + ".pkl") and not retrain:
        return
    
    if os.path.exists("saved_models/bert.pkl") and not retrain:
        return

    data = load_data(name_data)
    
    X_train, X_test, y_train, y_test = process_data(model_name, vectorizer_name,data)
    
    model = train_model(model_name, vectorizer_name, X_train, y_train)
    
    report(y_test, X_test, model,model_name)


