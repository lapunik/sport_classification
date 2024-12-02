from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from joblib import dump, load
from os import remove


def train_model(name, vectorizer_name, X_train, y_train):
    
    # TODO: More models and make comparism
    # TODO: Transformer model!!!!
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