import torch
from transformers import AutoTokenizer, T5ForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset

dataset = load_dataset("glue", "mrpc")

tokenizer = AutoTokenizer.from_pretrained("ihgn/Paraphrase-Detection-T5")
model = T5ForSequenceClassification.from_pretrained("ihgn/Paraphrase-Detection-T5", num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch", 
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')