from utils.data_loader import load_data
from utils.data_processor import process_data
from utils.predictor import report
from models.models import train_model

def create_model(model_name, vectorizer_name, name_data):
    
    data = load_data(name_data)
    
    X_train, X_test, y_train, y_test = process_data(vectorizer_name,data)
    
    model = train_model(model_name,X_train, y_train)
    
    report(y_test, X_test, model)


