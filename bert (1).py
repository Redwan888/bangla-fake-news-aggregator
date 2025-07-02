# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
import os

# Load datasets
authentic_df = pd.read_csv('/kaggle/input/bangla-fake-news/Authentic-48K.csv')
fake_df = pd.read_csv('/kaggle/input/bangla-fake-news/Fake-1K.csv')

# Sample data to balance
authentic_df = authentic_df[:1000]
fake_df = fake_df[:1000]

# Combine datasets
df = pd.concat([authentic_df, fake_df], ignore_index=True)
df = df.drop(columns=['articleID', 'domain', 'date', 'category'])

# Preprocess text (lowercase and clean text)
def preprocess_bangla_text(text):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = ''.join(e for e in text if e.isalnum() or e.isspace())  # Remove punctuation
    return text

df['content'] = df['content'].apply(preprocess_bangla_text)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)

# Use BERT tokenizer for Bangla or multilingual model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenize the texts
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=256, return_tensors='pt')
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=256, return_tensors='pt')

# Convert to Torch Datasets
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(y_train.values))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(y_test.values))

# Create DataLoader for batching
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Check if GPU is available and move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
model.to(device)  # Move model to GPU if available

# Prepare optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Training Loop
model.train()
for epoch in range(3):  # Training for 3 epochs
    total_loss = 0
    for batch_idx, batch in enumerate(train_dataloader):
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()

        # Move data to the GPU if available
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print loss every 100 batches
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch + 1}/3], Batch [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item()}")

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} complete. Average loss: {avg_loss}")

# Evaluation
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        y_pred.extend(predictions.cpu().numpy())  # Move to CPU for evaluation
        y_true.extend(labels.cpu().numpy())  # Move to CPU for evaluation

# Print classification report
print("BERT Performance:")
print(classification_report(y_true, y_pred))

# Save the model and tokenizer
output_dir = './bert_fake_news_model'
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")
