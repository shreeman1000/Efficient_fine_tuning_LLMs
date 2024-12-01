from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AdamW
import torch
import evaluate
import os
from sklearn.metrics import accuracy_score, f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

model_name = "google/pegasus-large"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

lora_config = LoraConfig(
    r=4, 
    lora_alpha=32, 
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

model = get_peft_model(model, lora_config)
from datasets import load_dataset, DatasetDict
dataset = load_dataset('ag_news')
tokenizer = AutoTokenizer.from_pretrained(model_name)
def preprocess_function(examples):
    inputs = ["classify: " + text for text in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=270, truncation=True, padding="max_length")
    targets = [
        "World" if i == 0 else
        "Sports" if i == 1 else
        "Business" if i == 2 else
        "Sci/Tech" for i in examples['label']
    ]
    labels = tokenizer(targets, max_length=4, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True).with_format("torch")

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./pegasus_lora_results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()

def evaluate_model(dataset, model, tokenizer, device="cuda"):
    model.eval()
    predictions = []
    true_labels = []

    def predict(text):
        input_text = "classify: " + text
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_length=4)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    for i in range(0, len(dataset["test"]), 100):
        text = dataset["test"][i]["text"]
        true_label = dataset["test"][i]["label"]
        true_labels.append(true_label)
        
        predicted_label = predict(text)
        predicted_numeric = {
            "World": 0,
            "Sports": 1,
            "Business": 2,
            "Sci/Tech": 3
        }.get(predicted_label, 0)
        predictions.append(predicted_numeric)
        

    accuracy = sum([p == t for p, t in zip(predictions, true_labels)]) / len(true_labels)
    print(accuracy, true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='micro')

    return accuracy, f1

evaluate_model(dataset, model, tokenizer)



