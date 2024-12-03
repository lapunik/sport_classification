from train import create_model
from test import use_model

available_models = ["bert",
                    "native_bayes", # tfidf: 0.85, count: 0.97, hashing: 0.70
                    "logistic_regression", # tfidf: 0.96, count: 0.98, hashing: 0.96
                    "svm"]

available_vectorizers = ["count", 
                         "tfidf", 
                         "hashing",
                         "sentence_transformer"]

available_data = ["sportoclanky",
                  "first100",
                  "first1k",
                  "first10"]

model = "logistic_regression"
vectorizer = "sentence_transformer"
data = "sportoclanky"
retrain = True

title = "Zemřel wimbledonský vítěz. Fraser byl v tenisu pojmem"
perex = "Ve věku 91 let zemřel bývalý australský tenista Neale Fraser. O úmrtí trojnásobného grandslamového vítěze ve dvouhře, dlouholetého kapitána daviscupové reprezentace a člena mezinárodní tenisové Síně slávy informoval národní svaz."
    

create_model(model, vectorizer, data, retrain)
# use_model(model, vectorizer, title, perex)