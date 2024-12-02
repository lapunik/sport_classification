from utils.data_loader import load_data
from utils.data_processor import process_data
import models.models as m
from sklearn.metrics import classification_report

data = load_data("first100.csv")

X_train, X_test, y_train, y_test = process_data(data)

model = m.native_bayes(X_train, y_train)



y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred,zero_division=1))