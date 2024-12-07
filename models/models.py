from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from utils.transformer_data_format import TextDataset, training_args
from transformers import BertForSequenceClassification, Trainer, TFAutoModel, AutoModel
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from joblib import dump, load
from os import remove, environ

def check_data(dataset):
    # Kontrola encodings
    for key, val in dataset.encodings.items():
        tensor = torch.tensor(val)
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"Encodings for {key} contain NaN or Inf values")
            return False

    # Kontrola labels
    labels_tensor = torch.tensor(dataset.numeric_labels)
    if torch.isnan(labels_tensor).any() or torch.isinf(labels_tensor).any():
        print("Labels contain NaN or Inf values")
        return False

    return True

def train_model(name, preprocesor_name, X_train, y_train):

    if name == "bert":

        dataset = TextDataset(X_train["train"], y_train["train"])
        dataset_eval = TextDataset(X_train["eval"], y_train["eval"])
        
        device = torch.device("cuda")
        model = BertForSequenceClassification.from_pretrained("Seznam/dist-mpnet-czeng-cs-en",
                                                               num_labels=len(set(np.concatenate((y_train["train"], y_train["eval"]))))).to(device)
        # checkpoint = "./results/checkpoint-600" 
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
        model = LogisticRegression(max_iter=1000, verbose=True)
    elif name == "native_bayes":
        model = MultinomialNB()
    elif name == "decision_tree":
        model = DecisionTreeClassifier()
    elif name == "mlp":
        model = MLPClassifier(verbose=True)
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

    return model