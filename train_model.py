import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

# ---- Load dataset ----
file_path = "chatbot_dataset.csv"  # make sure this name matches your file
df = pd.read_csv(file_path)
df.columns = [c.strip().lower() for c in df.columns]
print("Detected columns:", df.columns)

# ---- Normalize dataset columns ----
# rename depending on what exists
if 'question' in df.columns and 'answer' in df.columns:
    df.rename(columns={'question': 'pattern', 'answer': 'intent'}, inplace=True)
    print("✅ Detected Q/A format, mapping question→pattern, answer→intent")
elif 'patterns' in df.columns and ('tag' in df.columns or 'tags' in df.columns):
    df.rename(columns={'patterns': 'pattern', 'tag': 'intent', 'tags': 'intent'}, inplace=True)
    print("✅ Detected Kaggle intent format (patterns/tags)")
elif 'pattern' in df.columns and 'intent' in df.columns:
    print("✅ Using standard custom dataset format")
else:
    raise ValueError("❌ Could not detect expected column names. Please check your CSV structure.")

df.dropna(subset=['pattern', 'intent'], inplace=True)
print(f"Loaded {len(df)} samples.")

# ---- Tokenizer setup ----
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_texts, val_texts, train_labels, val_labels = train_test_split(df['pattern'], df['intent'], test_size=0.2)
label2id = {label: i for i, label in enumerate(df['intent'].unique())}
id2label = {i: label for label, i in label2id.items()}

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = [label2id[label] for label in labels]
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = ChatDataset(train_encodings, train_labels)
val_dataset = ChatDataset(val_encodings, val_labels)

# ---- Model and training ----
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label2id)
)

args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    save_total_limit=1,
    logging_dir='./logs'
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
model.save_pretrained("chatbot_model")
tokenizer.save_pretrained("chatbot_model")
print("✅ Model training complete and saved to chatbot_model/")
