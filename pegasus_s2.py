from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AdamW
import torch
import evaluate
import os
from sklearn.metrics import accuracy_score, f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = ["classify: " + text for text in examples["sentence"]]
    model_inputs = tokenizer(inputs, max_length=270, truncation=True, padding="max_length")
    targets = ["positive" if i == 1 else "negative" for i in examples['label']]
    labels = tokenizer(targets, max_length=3, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['sentence', 'label', 'idx']).with_format("torch")

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
    eval_dataset=tokenized_datasets["validation"],
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

    for i in range(0, len(dataset["validation"]), 1):
        text = dataset["validation"][i]["sentence"]
        true_label = dataset["validation"][i]["label"]
        true_labels.append(true_label)
        
        predicted_label = predict(text)
        if predicted_label != 'negative' and predicted_label != 'positive':
            print(predicted_label, i)
        predicted_numeric = 0 if predicted_label == "negative" else 1
        predictions.append(predicted_numeric)
        

    accuracy = sum([p == t for p, t in zip(predictions, true_labels)]) / len(true_labels)
    # print(accuracy, true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return accuracy, f1

evaluate_model(dataset, model, tokenizer)