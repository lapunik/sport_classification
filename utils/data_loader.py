import pandas as pd
import os

def load_data(name):
   
    csv_path = "data/" + name + ".csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError("file not found.")

    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        raise ValueError("empty CSV file")
    except pd.errors.ParserError:
            raise ValueError("wrong parse CSV file")