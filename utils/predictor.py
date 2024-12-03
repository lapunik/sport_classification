from sklearn.metrics import classification_report
from utils.transformer_data_format import TextDataset, training_args
from joblib import load
import torch


#  TODO: Visualization of report (Confusion matrix, ROC, AUC?)
def report(y_test, X_test, model,model_name):
    if model_name == "transformer":
        test_dataset = TextDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
        data = load("saved_models/transformer.pkl")
        model = data["model"]
        label = data["labels"]


        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = {key: val.to(model.device) for key, val in batch.items() if key != "labels"}
                labels = batch["labels"].to(model.device)
                outputs = model(**inputs)
                logits = outputs.logits
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds_labels = [label[pred] for pred in all_preds]
        all_labels_labels = [label[l] for l in all_labels]

        print(classification_report(all_labels_labels, all_preds_labels, zero_division=1))
    else:
        print(classification_report(y_test, model.predict(X_test),zero_division=1))


def predict(model_name, vectorizer_name, title, perex):
    text = title + " "  + perex
    
    if model_name == "bert":
        data = load("saved_models/bert.pkl")
        model = data["model"]
        tokenizer = data["tokenizer"]
        labels = data["labels"]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        device = model.device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()

        return labels[predicted_class_id]
    else:
        data = load("saved_models/" + model_name + "_" + vectorizer_name + ".pkl")
        model = data["model"]
        vectorizer = data["vectorizer"]
        text_vec = vectorizer.transform([text])
        return model.predict(text_vec)[0]