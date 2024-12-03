from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from utils.transformer_data_format import TextDataset, training_args
from transformers import BertForSequenceClassification, Trainer
from joblib import dump, load
from os import remove


def train_model(name, vectorizer_name, X_train, y_train):
    
    if name == "transformer":
        dataset = TextDataset(X_train, y_train)
        model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=len(set(y_train)))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        tok_path = "saved_models/transformer.pkl"
        data = load(tok_path)
        tokenizer = data["tokenizer"]
        labels = data["labels"]
        remove(tok_path)
        dump({"model": model, "tokenizer": tokenizer, "labels":labels}, "saved_models/transformer.pkl")

    else:
        # TODO: More models and make comparism
        if name == "svm":
            # TODO: More types of kernel
            model = SVC(kernel="sigmoid")
        elif name == "logistic_regression":
            model = LogisticRegression(max_iter=1000)
        elif name == "native_bayes":
            model = MultinomialNB()
        else:
            raise ValueError("Model: " + name + " not included")
        
        model.fit(X_train, y_train)

        vec_path = "saved_models/"+ vectorizer_name +".pkl"
        vectorizer = load(vec_path)
        remove(vec_path)
        dump({"model": model, "vectorizer": vectorizer}, "saved_models/" + name  + "_" +  vectorizer_name + ".pkl")

    return model