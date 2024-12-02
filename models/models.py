from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from joblib import dump


def train_model(name, X_train, y_train):
    
    if name == "svc":
        model = SVC(kernel="linear", probability=True)  
    elif name == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    elif name == "native_bayes":    
        model = MultinomialNB()
    model.fit(X_train, y_train)
    dump(model, "saved_models/classifier_model_" + name + ".pkl")
    return model