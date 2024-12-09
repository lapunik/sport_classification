from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from utils.transformer_data_format import TextDataset, training_args
from transformers import AutoModelForSequenceClassification, Trainer
import torch
import numpy as np
from joblib import dump, load
from os import remove

def train_model(name, preprocesor_name, X_train, y_train, bert_model =  "Seznam/dist-mpnet-paracrawl-cs-en"): 
    if name == "bert":

        dataset = TextDataset(X_train["train"], y_train["train"])
        dataset_eval = TextDataset(X_train["eval"], y_train["eval"])
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(bert_model,
                                                               num_labels=len(set(np.concatenate((y_train["train"], y_train["eval"]))))).to(device)
        # checkpoint = "./results/checkpoint-26694" 
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset_eval
        )
        # trainer.train(resume_from_checkpoint=checkpoint)
        trainer.train()
        preprocesor_name = "bert"
    elif name == "svm":
        # TODO: More types of kernel
        model = SVC(kernel="sigmoid",verbose=True)
    elif name == "logistic_regression":
        model = LogisticRegression(solver="sag",verbose=True)
    elif name == "native_bayes":
        model = MultinomialNB(alpha=0.0006) # count: 0.006, tfidf: 0.011, hashing: 0.0006 (Note: n-grams decrease performance)
    elif name == "decision_tree":
        model = DecisionTreeClassifier(criterion="gini")
    elif name == "kneighbors":
        model = KNeighborsClassifier()
    elif name == "mlp":
        model = MLPClassifier(max_iter=20,hidden_layer_sizes=(100,50),activation="tanh",early_stopping=True,solver="adam", learning_rate="adaptive" ,verbose=True)
    elif name == "random_forest":
        model = RandomForestClassifier(verbose=True)
    else:
        raise ValueError("Model: " + name + " not included")
        
    if name != "bert":
        model.fit(X_train, y_train)

    path = "saved_models/"+ preprocesor_name +".pkl"
    data = load(path)
    preprocesor = data["preprocesor"]
    labels = data["labels"]
    remove(path)
    dump({"model": model, "preprocesor": preprocesor, "labels":labels}, "saved_models/" + name  + "_" +  preprocesor_name + ".pkl")
