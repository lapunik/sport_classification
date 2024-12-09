from train import create_model
from test import use_model

# acc/f1
available_models = ["bert", # 0.9879/0.9878("Seznam/dist-mpnet-paracrawl-cs-en") 
                    "native_bayes", # tfidf: 0.98/0.98, count: 0.98/0.98, hashing: 0.98/0.98 
                    "logistic_regression", # tfidf: 0.97/0.97  , count: 0.98/0.98, hashing: 0.97/0.96
                    "decision_tree", # tfidf:0.93/0.93, count: 0.94/0.93, hashing: 0.93/0.93
                    "mlp", # tfidf: 0.98/0.98, count: 0.98/0.98, hashing: 0./0.
                    "random_forest", # tfidf: 0.96/0.95, count: 0.96/0.95, hashing: 0.95/0.95
                    "kneighbors", # tfidf: 0.97/0.97, count: 0.77/0.76, hashing: 0.96/0.96
                    "svm"]  # tfidf: 0.98/0.98, count: 0.98/0.98, hashing: 0./0.

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
retrain = False

title = "Pohár konstruktérů vyhrává McLaren, pro vítězství si v Abú Zabí dojel Norris"
perex = "McLaren si zásluhou Landa Norrise dojel v Abú Zabí pro vítězství v letošním Poháru konstruktérů formule 1. Norris vyrážel z pole position a první místo udržel po dobu trvání celého závodu. Naopak jeho týmový kolega Oscar Piastri se po kolizi v prvním kole propadl a dojel desátý. Druhý dojel Carlos Sainz a třetí Charles Leclerc, ale Ferrari to na zisk týmového titulu nestačilo."

if __name__ == '__main__':  
    # create_model(model, preprocesor, data, retrain)
    use_model(model, preprocesor, title, perex)