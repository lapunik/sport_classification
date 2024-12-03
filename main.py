from train import create_model
from test import use_model

available_models = ["transformer","native_bayes", "logistic_regression", "svm"]
available_vectorizers = ["count", "tfidf", "hashing"]
available_data = ["sportoclanky", "first100", "first10"]

model = "transformer"
vectorizer = "tfidf"
data = "firts100"
retrain = False

title = "Vstupenky na ME basketbalistek jsou v prodeji, soupeřky Češek ale zatím nejsou známy"
perex = "Začal prodej lístků na mistrovství Evropy basketbalistek v příštím roce, během něhož se odehraje jedna ze základních skupin v Brně. Na svém webu o tom informoval CZ.Basketball. K dispozici jsou zatím celodenní vstupenky do části hlediště, další budou uvolněny po 8. březnu."
    

# create_model(model, vectorizer, data, retrain)
use_model(model, vectorizer, title, perex)