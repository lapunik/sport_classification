from sklearn.metrics import classification_report
from joblib import load


#  TODO: Visualization of report (Confusion matrix, ROC, AUC?)
def report(y_test, X_test, model):
    print(classification_report(y_test, model.predict(X_test),zero_division=1))


def predict(model_name, vectorizer_name, title, perex):
    data = load("saved_models/" + model_name + "_" + vectorizer_name + ".pkl")
    model = data['model']
    vectorizer = data['vectorizer']
    text = title + " "  + perex
    text_tfidf = vectorizer.transform([text])
    return model.predict(text_tfidf)[0]