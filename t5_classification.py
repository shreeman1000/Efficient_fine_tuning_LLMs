from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AdamW
import torch
import evaluate
import os
from sklearn.metrics import accuracy_score, f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
model_name = "google-t5/t5-small"
# num_labels = 4 for ag news
model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map = 'cuda', num_labels = 2)

lora_config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["q", "k", "v", "o"], 
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_CLASS_LM",
)

model = get_peft_model(model, lora_config)

from datasets import load_dataset, DatasetDict
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# dataset2 = load_dataset('ag_news')

def preprocess_function(examples):
    inputs = ["sentiment: " + doc for doc in examples["sentence"]]  # Add task prefix
    model_inputs = tokenizer(inputs, max_length=270, truncation=True, padding="max_length")

    model_inputs["labels"] = examples["label"]
    return model_inputs

def preprocess_function_ag(examples):
    inputs = ["classify: " + doc for doc in examples["text"]]  # Add task prefix
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    model_inputs["labels"] = examples["label"]  # Labels are already numerical (0-3) in AG News
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['sentence', 'label', 'idx']).with_format("torch")
from torch.utils.data import DataLoader
train_dataset = tokenized_datasets["train"]
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

eval_dataset = tokenized_datasets["validation"]
eval_dataloader = DataLoader(eval_dataset, batch_size=16)
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        inputs = {key: value.to('cuda') for key, value in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader)}")

    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in eval_dataloader:
        inputs = {key: value.to('cuda') for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        preds = logits.argmax(dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(inputs['labels'].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"Epoch {epoch+1} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")