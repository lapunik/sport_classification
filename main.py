from train import create_model
from test import use_model

# TODO: List of models and vectorizers
model = "svm"
vectorizer = "vec"
data = "sportoclanky"
retrain = True

title = "Česká reprezentace vyhrála zápas v kopané"
perex = "Včera večer česká reprezentace porazila soupeře v dramatickém zápase golem z penaly."
    
create_model(model, vectorizer, data, retrain)
use_model(model, vectorizer, title, perex)