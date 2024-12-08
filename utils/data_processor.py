from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from transformers import AutoTokenizer
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import utils.csl as csl
from joblib import dump

 
def process_data(model_name, preprocesor_name, data, min_class_count=2, test_size=0.2, random_state=42,bret_model = "Seznam/dist-mpnet-paracrawl-cs-en"): # "Seznam/dist-mpnet-czeng-cs-en" "Seznam/simcse-small-e-czech"
    class_counts = Counter(data["category"])
    classes_to_keep = [cl for cl, count in class_counts.items() if count >= min_class_count]
    mask = data["category"].isin(classes_to_keep)
    data = data[mask].copy()

    data.loc[:,"text"] = data["rss_title"] + " "  + data["rss_perex"]
    label_encoder = LabelEncoder()
    y_en = label_encoder.fit_transform(data["category"])
    labels = label_encoder.classes_.tolist()  

    X_train, X_test, y_train, y_test = train_test_split(data["text"], y_en, test_size=test_size, random_state=random_state,stratify=y_en)


    if model_name == "bert":
        preprocesor = AutoTokenizer.from_pretrained(bret_model)
        max_length = min(512,preprocesor.model_max_length, max(len(preprocesor.encode(text)) for text in data["text"]))
        
        X_train_s, X_eval, y_train_s, y_eval = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state,stratify=y_train)
        X_train_vec = {
            "train": preprocesor(X_train_s.tolist(), truncation=True, padding=True, max_length=max_length),
            "eval": preprocesor(X_eval.tolist(), truncation=True, padding=True, max_length=max_length)}
        y_train = {
            "train": y_train_s,
            "eval": y_eval}
        X_test_vec = preprocesor(X_test.tolist(), truncation=True, padding=True, max_length=max_length)
        preprocesor_name = "bert"
    # TODO: other types of stoplist
    elif preprocesor_name == "tfidf": 
        preprocesor = TfidfVectorizer(stop_words = csl.czech_stop_words) # n-grams decrease efficiency
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
