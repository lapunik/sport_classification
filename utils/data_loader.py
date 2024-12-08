import pandas as pd
import os

def load_data(name, encoding='utf-8', sep=',', dtype=None):
    csv_path = os.path.join("data", f"{name}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError("file not found.")

    try:
        df = pd.read_csv(csv_path, encoding=encoding, sep=sep, dtype=dtype)
        
        if not df.empty:
                    return df
        raise ValueError(f"CSV file {name} is empty after loading")
    
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file {name} is empty")
    except pd.errors.ParserError:
            raise ValueError(f"Unable to parse CSV file {name}")