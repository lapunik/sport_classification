import pandas as pd
import os

def load_data(name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # /../../
    csv_path = os.path.join(base_dir, "data", name) # /data/"name.csv"
    return pd.read_csv(csv_path)