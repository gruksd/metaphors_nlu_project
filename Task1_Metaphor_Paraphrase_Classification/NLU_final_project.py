import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

with open('metaphor_paraphrase_corpus.txt', 'r') as file:
    data = file.read().strip()

blocks = data.split('\n\n')

test_data = []

result_data = []

for block in blocks:
    lines = [line.strip() for line in block.split("\n") if line.strip()]
    if not lines:
        continue  

    premise = lines[0]

    for line in lines[1:]:
        if '#' in line:
            hypothesis, label = line.rsplit('#', 1)
            label = int(label)
            if label == 4 or label == 3:
                new_label = 0
            if label == 2:
                new_label = 1
            if label == 1:
                new_label = 2
            result_data.append([premise, hypothesis.strip(), new_label])
            test_data.append([premise, hypothesis.strip()])
        else:
            continue



tokenizer = AutoTokenizer.from_pretrained("ixaxaar/flan-t5-base_multi-nli")
model = AutoModelForSequenceClassification.from_pretrained("ixaxaar/flan-t5-base_multi-nli")


label_map = {0: 0, 1: 1, 2: 2} #{0: "entailment", 1: "neutral", 2: "contradiction"}

def classify_nli(premise, hypothesis):
    input_text = f"{premise} [SEP] {hypothesis}"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = torch.argmax(logits, dim=-1).item()
    return label_map[predicted_label]

with open("output_file.txt", 'w', encoding='utf-8') as file:
        for premise, hypothesis in test_data:
            result = classify_nli(premise, hypothesis)
            file.write(f"{premise}#{hypothesis}#{result}\n")



