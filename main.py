from utils.data_loader import load_data
from utils.data_processor import process_data

data = load_data("first10.csv")

X_train, X_test, y_train, y_test = process_data(data)

