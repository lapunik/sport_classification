from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import utils.csl as csl
from joblib import dump

 
def process_data(model_name, preprocesor_name, data):
    data["text"] = data["rss_title"] + " "  + data["rss_perex"]
   
    label_encoder = LabelEncoder()
    y_en = label_encoder.fit_transform(data["category"])
    labels = label_encoder.classes_.tolist()  
    X_train, X_test, y_train, y_test = train_test_split(data["text"], y_en, test_size=0.2, random_state=42)

    if model_name == "bert":
        preprocesor = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
        X_train_vec = preprocesor(X_train.tolist(), truncation=True, padding=True, max_length=512)
        X_test_vec = preprocesor(X_test.tolist(), truncation=True, padding=True, max_length=512)
        preprocesor_name = "bert"
    # TODO: other types of stoplist
    elif preprocesor_name == "tfidf": 
        preprocesor = TfidfVectorizer(stop_words = csl.czech_stop_words, ngram_range=(1, 2)) # TODO: Trying ngrams
    elif preprocesor_name == "count":
        preprocesor = CountVectorizer(stop_words = csl.czech_stop_words)
    elif preprocesor_name == 'hashing':
        preprocesor = HashingVectorizer(stop_words=csl.czech_stop_words,alternate_sign=False)
    else:
        raise ValueError("Model: " + preprocesor_name + " not included")
    if model_name != "bert":
        X_train_vec = preprocesor.fit_transform(X_train)
        X_test_vec = preprocesor.transform(X_test)

    dump({"preprocesor":preprocesor, "labels":labels}, "saved_models/" + preprocesor_name + ".pkl")    
    return X_train_vec, X_test_vec, y_train, y_test
