from transformers import AdamW
import torch
import os
from sklearn.metrics import accuracy_score, f1_score
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model

# num_labels = 4 for ag news also test = validation for agnews
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-base", num_labels=2, device_map='cuda')
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # Target modules to apply LoRA
    lora_dropout=0.1,  
    bias="none", 
    task_type="SEQ_2_CLASS_LM",
)

model = get_peft_model(model, lora_config)
from datasets import load_dataset, DatasetDict
dataset = load_dataset("glue", "sst2")
# dataset2 = load_dataset('ag_news')

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
def preprocess_function(examples):
    inputs = ["sentiment: " + doc for doc in examples["sentence"]] 
    model_inputs = tokenizer(inputs, max_length=270, truncation=True, padding="max_length")

    model_inputs["labels"] = examples["label"]
    return model_inputs

# for ag-news
def preprocess_function_ag(examples):
    inputs = ["classify: " + doc for doc in examples["text"]]
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
    model.train()
    total_loss = 0
    for i in train_dataloader:
        optimizer.zero_grad()
        i = {k : v.cuda() for k,v in i.items()}
        o = model(**i)
        l = o.loss
        l.backward()
        optimizer.step()
        total_loss += l.item()

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



