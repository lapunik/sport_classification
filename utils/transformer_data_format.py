import torch as t
from torch.utils.data import Dataset
from transformers import TrainingArguments

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.label_map = {label: idx for idx, label in enumerate(set(self.labels))}
        self.numeric_labels = [self.label_map[label] for label in self.labels]

    def __getitem__(self, idx):
        item = {key: t.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = t.tensor(self.numeric_labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,  
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8,
    warmup_steps=100, 
    weight_decay=0.01,
    learning_rate=5e-5,  
    eval_strategy="epoch",  
    save_strategy="epoch",
    load_best_model_at_end=True,  
    metric_for_best_model="loss",
    logging_dir='./logs',
    logging_steps=10,
)