from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import utils.csl as csl
from joblib import dump

 
def process_data(model_name, vectorizer_name, data):
    # TODO: does it make sense to distinguish between a perex and a title?
    data["text"] = data["rss_title"] + " "  + data["rss_perex"]
    # TODO: try randomstate = none!
    X_train, X_test, y_train, y_test = train_test_split(data["text"], data["category"], test_size=0.2, random_state=42)
    if model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        X_train_vec = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=512)
        X_test_vec = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=512)
        labels = list(set(y_train))  # Získání názvů tříd
        dump({"tokenizer": tokenizer, "labels":labels}, "saved_models/transformer.pkl")
    elif vectorizer_name == "sentence_transformer":
        vectorizer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        X_train_vec = vectorizer.encode(X_train.tolist(), show_progress_bar=True)
        X_test_vec = vectorizer.encode(X_test.tolist(), show_progress_bar=True)
        dump(vectorizer, "saved_models/" + vectorizer_name + ".pkl")

    else:
        # TODO: other types of vectorizer
        # TODO: other types of stoplist
        # TODO: set max_features = for example 5000 (number of word limit by usage)
        # TODO: try some ngram_range (ngram_range=(1, 3) from one word to three words gram)
        if vectorizer_name == "tfidf": 
            vectorizer = TfidfVectorizer(stop_words = csl.czech_stop_words)
        elif vectorizer_name == "count":
            vectorizer = CountVectorizer(stop_words = csl.czech_stop_words)
        elif vectorizer_name == 'hashing':
            vectorizer = HashingVectorizer(stop_words=csl.czech_stop_words,alternate_sign=False)
        else:
            raise ValueError("Model: " + vectorizer_name + " not included")

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        dump(vectorizer, "saved_models/" + vectorizer_name + ".pkl")
    
    return X_train_vec, X_test_vec, y_train, y_test

# Notes:
# 
# 