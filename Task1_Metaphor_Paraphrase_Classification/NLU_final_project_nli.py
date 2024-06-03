import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_data_f():
    with open('metaphor_paraphrase_corpus.txt', 'r') as file:
        data = file.read().strip()

    blocks = data.split('\n\n')

    test_data = []

    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue  

        premise = lines[0]

        for line in lines[1:]:
            if '#' in line:
                hypothesis, label = line.rsplit('#', 1)
                label = int(label)
                test_data.append([premise, hypothesis.strip()])
            else:
                continue
    return test_data

def classify_nli(tokenizer,model,premise, hypothesis):
    input_text = f"[CLS] {premise} [SEP] {hypothesis} [SEP]"
    
    label_map = {0: 0, 1: 1, 2: 2} #{0: "entailment", 1: "neutral", 2: "contradiction"}
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = torch.argmax(logits, dim=-1).item()
    return label_map[predicted_label]

def nli_t5_test(test_data):
    tokenizer = AutoTokenizer.from_pretrained("ixaxaar/flan-t5-base_multi-nli")
    model = AutoModelForSequenceClassification.from_pretrained("ixaxaar/flan-t5-base_multi-nli")

    with open("output_file_nli_t5.txt", 'w', encoding='utf-8') as file:
            for premise, hypothesis in test_data:
                result = classify_nli(tokenizer, model,premise, hypothesis)
                file.write(f"{premise}#{hypothesis}#{result}\n")

def nli_large_roberta(test_data):
    tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
    model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")

    with open("output_file_nli_large_roberta.txt", 'w', encoding='utf-8') as file:
            for premise, hypothesis in test_data:
                result = classify_nli(tokenizer, model, premise, hypothesis)
                file.write(f"{premise}#{hypothesis}#{result}\n")


def nli_large_bart(test_data):
    tokenizer = AutoTokenizer.from_pretrained("ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli")
    model = AutoModelForSequenceClassification.from_pretrained("ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli")

    with open("output_file_nli_large_bart.txt", 'w', encoding='utf-8') as file:
            for premise, hypothesis in test_data:
                result = classify_nli(tokenizer, model, premise, hypothesis)
                file.write(f"{premise}#{hypothesis}#{result}\n")

def nli_debiased_bert(test_data):
    tokenizer = AutoTokenizer.from_pretrained("tomhosking/bert-base-uncased-debiased-nli")
    model = AutoModelForSequenceClassification.from_pretrained("tomhosking/bert-base-uncased-debiased-nli")

    with open("output_file_nli_debiased_bert.txt", 'w', encoding='utf-8') as file:
            for premise, hypothesis in test_data:
                result = classify_nli(tokenizer, model, premise, hypothesis)
                file.write(f"{premise}#{hypothesis}#{result}\n")


test_data = test_data_f()
#nli_t5_test(test_data)
#nli_large_roberta(test_data)
nli_large_bart(test_data)
nli_debiased_bert(test_data)
