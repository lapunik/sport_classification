from train import create_model
from test import use_model

model = "native_bayes"
vectorizer = "vec"
data = "sportoclanky.csv"

title = "Česká reprezentace vyhrála zápas v kopané"
perex = "Včera večer česká reprezentace porazila soupeře v dramatickém zápase golem z penaly."
    
create_model(model, vectorizer, data)
use_model(model, vectorizer, title, perex)