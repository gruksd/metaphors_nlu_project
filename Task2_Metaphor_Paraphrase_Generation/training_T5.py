import pandas as pd
import numpy as np # type: ignore
import torch
import evaluate
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq



# Load dataset
data=pd.read_csv('/trainingdata_paraphrases_cleaned.csv',sep=',')

data = Dataset.from_pandas(data)


print(data)

data = data.train_test_split(test_size=0.03)

num_labels = len(set(data["train"][:]["paraphrase_word"]))
print(num_labels)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small", num_labels=num_labels).to(device)


#Tokenize and map input

def tok_input(example):
  sentence = example["sentence"]
  input = tokenizer(sentence, return_tensors="pt")
  for key in input.keys():
    input[key] = torch.squeeze(input[key])
  return input

def tok_label(example):
  #print(example)
  label = example["paraphrase_word"]
  label = tokenizer(label, return_tensors="pt")
  for key in label.keys():
    label[key] = torch.squeeze(label[key])
  return label

prefix = "paraphrase highlighted word: "
max_input_length = 512
max_target_length = 24

def preprocess_data(example):
  #tokenise input
  input = prefix + example["sentence"]
  model_inputs = tokenizer(input, max_length=max_input_length, truncation=True)

  # Setup the tokenizer for targets
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(example["paraphrase_word"], max_length=max_target_length, 
                       truncation=True)
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

data["train"] = data["train"].map(preprocess_data)
data["test"] = data["test"].map(preprocess_data)
#data["train"] = data["train"].map(tok_label)
#data["test"] = data["test"].map(tok_label)

print(data["test"])
print(data["test"][0])

# Prepare training

data_collator = DataCollatorForSeq2Seq(tokenizer)


training_args = TrainingArguments(output_dir="training", 
                                  num_train_epochs = 5.0, 
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=32,
                                  do_train = True,
                                  do_eval = True,
                                  save_strategy = "epoch", 
                                  eval_strategy="steps",
                                  eval_steps=100,)


trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = data["train"],
    eval_dataset=data["test"],
    data_collator = data_collator
)

trainer.train()