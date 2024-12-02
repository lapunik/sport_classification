from sklearn.metrics import classification_report
from joblib import load


def report(y_test, X_test, model):
    print(classification_report(y_test, model.predict(X_test),zero_division=1))


def predict(model_name, vectorizer_name, title, perex):
    model = load("saved_models/classifier_model_" + model_name + ".pkl")
    vectorizer = load("saved_models/vectorizer_ " + vectorizer_name + " .pkl")
    text = title + " "  + perex
    text_tfidf = vectorizer.transform([text])
    return model.predict(text_tfidf)[0]