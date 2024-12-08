from utils.data_loader import load_data
from utils.data_processor import process_data
from utils.predictor import report
from models.models import train_model
import os

def create_model(model_name, preprocesor_name, name_data, retrain):
    
    if not retrain:
        if os.path.exists("saved_models/" + model_name + "_" + preprocesor_name + ".pkl") or (os.path.exists("saved_models/bert_bert.pkl") and model_name == "bert"):
            return

    data = load_data(name_data)
        
    X_train, X_test, y_train, y_test = process_data(model_name, preprocesor_name,data)

    train_model(model_name, preprocesor_name, X_train, y_train)

    report(y_test, X_test,model_name,preprocesor_name)


