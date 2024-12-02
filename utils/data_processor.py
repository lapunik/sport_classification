from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import utils.csl as csl
 
def process_data(data):
    data["text"] = data["rss_title"] + ' ' + data["rss_perex"]
    X_train, X_test, y_train, y_test = train_test_split(data["text"], data["category"], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words = csl.czech_stop_words, max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test
