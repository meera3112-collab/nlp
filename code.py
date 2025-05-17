import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import DistilBertModel, DistilBertTokenizerFast, get_linear_schedule_with_warmup

# 1) Load & filter just 2 000 samples
df = pd.read_csv(
    '/content/training.1600000.processed.noemoticon.csv',
    header=None,
    names=['target','ids','date','query','user','text'],
    sep=',',
    encoding='latin-1',
    usecols=[0,5]  # only need target and text
)
df = df[df.target.isin([0,4])].sample(2000, random_state=42)
df['label'] = df.target.map({0:0,4:1})

# 2) Train/Val split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df.text.tolist(), df.label.tolist(), test_size=0.2, random_state=42
)

# 3) Dataset + DataLoader
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

class TinyTweetDataset(Dataset):
    def _init_(self, texts, labels):
        self.enc = tokenizer(texts,
                             padding='max_length',
                             truncation=True,
                             max_length=64,
                             return_tensors='pt')
        self.labels = torch.tensor(labels)
    def _len_(self): return len(self.labels)
    def _getitem_(self, idx):
        return {
            'input_ids':      self.enc.input_ids[idx],
            'attention_mask': self.enc.attention_mask[idx],
            'label':          self.labels[idx]
        }

train_loader = DataLoader(TinyTweetDataset(train_texts, train_labels), batch_size=32, shuffle=True)
val_loader   = DataLoader(TinyTweetDataset(val_texts,   val_labels),   batch_size=32)

# 4) Model, optimizer, scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
head = nn.Linear(bert.config.hidden_size, 2).to(device)

opt = AdamW(list(bert.parameters()) + list(head.parameters()), lr=3e-5)
total_steps = len(train_loader) * 1  # 1 epoch
sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = nn.CrossEntropyLoss()

# 5) Single‐epoch train & eval
bert.train(); head.train()
for batch in train_loader:
    ids  = batch['input_ids'].to(device)
    mask = batch['attention_mask'].to(device)
    lbls = batch['label'].to(device)
    opt.zero_grad()
    out = bert(ids, attention_mask=mask).last_hidden_state[:,0]
    logits = head(out)
    loss = loss_fn(logits, lbls)
    loss.backward()
    opt.step()
    sched.step()

bert.eval(); head.eval()
all_preds, all_lbls = [], []
with torch.no_grad():
    for batch in val_loader:
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        lbls = batch['label'].to(device)
        out = bert(ids, attention_mask=mask).last_hidden_state[:,0]
        preds = head(out).argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_lbls.extend(lbls.cpu().tolist())

# 6) Metrics
acc = accuracy_score(all_lbls, all_preds)
f1  = f1_score(all_lbls, all_preds, average='weighted')
print(f"Val Accuracy: {acc:.4f}, Val F1: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(all_lbls, all_preds, target_names=['negative','positive']))
