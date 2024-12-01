
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
import torch
import evaluate
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base", device_map = 'cuda')

lora_config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], 
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

model = get_peft_model(model, lora_config)

from datasets import load_dataset, DatasetDict
dataset = load_dataset("cnn_dailymail", "3.0.0")
small_dataset = DatasetDict({
    "train": dataset["train"].select(range(100000)),     
    "validation": dataset["validation"].select(range(10000)), 
    "test": dataset["test"].select(range(2000))         
})

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["highlights"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

tokenized_datasets = small_dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./bart_lora_finetuned",  # Directory to save model
    eval_strategy="epoch",  # Evaluate after every epoch
    learning_rate=5e-5,  # Learning rate
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    num_train_epochs=3,  # Number of training epochs
    weight_decay=0.01,  # Weight decay
    save_strategy="epoch",  # Save model after each epoch
    logging_dir="./logs",  # Directory for logs
    logging_steps=100,  # Log every 100 steps
    fp16=True,  # Use mixed precision
    save_total_limit=2,  # Save only the last 2 checkpoints
    load_best_model_at_end=True,  # Load best model after training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
)

trainer.train()

trainer.save_model("./bart_lora_finetuned_model")

input_text = "summarize: " + dataset["test"][1]["article"]
input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.cuda()
generated_ids = model.generate(input_ids = input_ids, max_length=128, num_beams=4, early_stopping=True)
summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(summary)

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
        
        generated_preds.append(generated_summary)  # BLEU expects a list of predictions
        references.append(reference_summary)  # BLEU expects a list of references
    
    return generated_preds, references

generated_preds, references = decode(small_dataset['test'], model, tokenizer)

bleu_score = bleu_metric.compute(predictions=generated_preds, references=references)
rouge_score = rouge_metric.compute(predictions=generated_preds, references=references)

print(bleu_score, rouge_score)




