from train import create_model
from test import use_model

available_models = ["bert", 
                    "native_bayes", 
                    "logistic_regression", 
                    "decision_tree", 
                    "mlp", 
                    "random_forest",
                    "kneighbors", 
                    "svm"] 

# only if model is not bert
available_preprocessors = ["count", 
                         "tfidf", 
                         "hashing",]

available_data = ["sportoclanky"] #.csv


model = "bert"
preprocessor = "count"
data = "sportoclanky"
retrain = False

title = "Pohár konstruktérů vyhrává McLaren, pro vítězství si v Abú Zabí dojel Norris"
perex = "McLaren si zásluhou Landa Norrise dojel v Abú Zabí pro vítězství v letošním Poháru konstruktérů formule 1. Norris vyrážel z pole position a první místo udržel po dobu trvání celého závodu. Naopak jeho týmový kolega Oscar Piastri se po kolizi v prvním kole propadl a dojel desátý. Druhý dojel Carlos Sainz a třetí Charles Leclerc, ale Ferrari to na zisk týmového titulu nestačilo."

if __name__ == '__main__':  
    create_model(model, preprocessor, data, retrain)
    use_model(model, preprocessor, title, perex)