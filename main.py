from train import create_model
from test import use_model

available_models = ["bert", 
                    "native_bayes", # tfidf: 0.85, count: 0.97, hashing: 0.70, 
                    "logistic_regression", # tfidf: 0.96, count: 0.98, hashing: 0.96
                    "decision_tree", # tfidf: 0., count: 0., hashing: 0.
                    "mlp", # tfidf: 0., count: 0., hashing: 0.
                    "random_forest", # tfidf: 0., count: 0., hashing: 0.
                    "svm"] # tfidf: 0.98, count: 0.97, hashing: 0.98

available_preprocesors = ["count", 
                         "tfidf", 
                         "hashing",]

available_data = ["sportoclanky",
                  "first100",
                  "first1k",
                  "first10k",
                  "first10"]

model = "bert"
preprocesor = "count"
data = "sportoclanky"
retrain = True

title = "Dakar 2025: Vše o 46. ročníku nejslavnější rallye v Saudské Arábii"
perex = "Začátek sportovního roku bude jako vždy patřit nejslavnější světové rallye. Dakar se pojede i v roce 2025 už pošesté za sebou v Saudské Arábii, a to v termínu od 3. do 17. ledna. Kompletní programový i výsledkový servis slavného motoristického závodu vám nabízí Sport.cz."
    

create_model(model, preprocesor, data, retrain)
use_model(model, preprocesor, title, perex)
