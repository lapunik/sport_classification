from sklearn.metrics import classification_report
from utils.transformer_data_format import TextDataset, training_args
from joblib import load
import torch


#  TODO: Visualization of report (Confusion matrix, ROC, AUC?)
def report(y_test, X_test, model,model_name,preprocesor_name):
    if model_name == "bert":
        data = load("saved_models/"+ model_name + "_" + model_name + ".pkl")
        model = data["model"]
        label = data["labels"]

        all_preds = []
        all_labels = []
        inputs = {}
        test_dataset = TextDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
        with torch.inference_mode():
            for batch in test_loader:
                for key, val in batch.items():
                    if key != "labels":
                        inputs.update({key: val.to(model.device)})
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
        data = load("saved_models/" + model_name  + "_" +  preprocesor_name + ".pkl")
        labels = data["labels"]
        model = data["model"]
        print(classification_report(y_test, model.predict(X_test),zero_division=1,target_names=labels))


def predict(model_name, preprocesor_name, title, perex):
    text = title + " "  + perex
    
    if model_name == "bert":
        data = load("saved_models/"+ model_name + "_" + model_name + ".pkl")
        model = data["model"]
        preprocesor = data["preprocesor"]
        labels = data["labels"]

        inputs = preprocesor(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        device = model.device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()

        return labels[predicted_class_id]
    else:
        data = load("saved_models/" + model_name + "_" + preprocesor_name + ".pkl")
        model = data["model"]
        preprocesor = data["preprocesor"]
        labels = data["labels"]
        text_vec = preprocesor.transform([text])
        return labels[model.predict(text_vec)[0]]