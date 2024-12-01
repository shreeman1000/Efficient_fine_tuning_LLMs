from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
import torch
import evaluate
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small", device_map = 'cuda')

lora_config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["q", "k", "v", "o"], 
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

model = get_peft_model(model, lora_config)

from datasets import load_dataset, DatasetDict
dataset = load_dataset("cnn_dailymail", "3.0.0")
small_dataset = DatasetDict({
    "train": dataset["train"].select(range(150000)),  
    "validation": dataset["validation"].select(range(10000)), 
    "test": dataset["test"].select(range(2000))          
})
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    labels = tokenizer(examples["highlights"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = small_dataset.map(preprocess_function, batched=True)
training_args = TrainingArguments(
    output_dir="./t5_lora_finetuned", 
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16, 
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch", 
    logging_dir="./logs",
    logging_steps=100,
    fp16=True,
    save_total_limit=2, 
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
)

trainer.train()

trainer.save_model("./t5_lora_finetuned_model")

input_text = "summarize: " + dataset["test"][80]["article"]
input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.cuda()
generated_ids = model.generate(input_ids = input_ids, max_length=128, num_beams=4, early_stopping=True)
summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(summary)

from tqdm import tqdm

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

def decode(test_dataset, model, tokenizer):
    generated_preds = []
    references = []
    for example in test_dataset:
        input_text = "summarize: " + example["article"]
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(model.device)
        
        with torch.no_grad():
            output = model.generate(input_ids = inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
        generated_summary = tokenizer.decode(output[0], skip_special_tokens=True)
        reference_summary = example["highlights"]
        
        generated_preds.append(generated_summary) 
        references.append(reference_summary) 
    
    return generated_preds, references

generated_preds, references = decode(small_dataset['test'], model, tokenizer)
bleu_score = bleu_metric.compute(predictions=generated_preds, references=references)
rouge_score = rouge_metric.compute(predictions=generated_preds, references=references)
print("bleu_score =", bleu_score, "rouge_score =", rouge_score)