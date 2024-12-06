from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from utils.transformer_data_format import TextDataset, training_args
from transformers import BertForSequenceClassification, Trainer
from joblib import dump, load
from os import remove

def train_model(name, preprocesor_name, X_train, y_train):
    
    if name == "bert":
        dataset = TextDataset(X_train, y_train)
        model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=len(set(y_train)))
        # checkpoint = "./results/checkpoint-600" 
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            # eval_dataset=dataset,
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