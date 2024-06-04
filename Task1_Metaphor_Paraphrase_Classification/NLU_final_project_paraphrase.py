
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



def process_pair(tokenizer, model, sentence1, sentence2):
    softmax = torch.nn.Softmax(dim=1)
    inputs = tokenizer(sentence1, sentence2, return_tensors='pt')
    logits = model(**inputs).logits
    probs = softmax(logits)
    label = torch.argmax(probs, dim=1).item()  
    return label


def para_bert(test_data):
    tokenizer = AutoTokenizer.from_pretrained("Prompsit/paraphrase-bert-en")
    model = AutoModelForSequenceClassification.from_pretrained("Prompsit/paraphrase-bert-en")
    with open("output_file_para_bert.txt", 'w') as f:
        for premise, hypothesis in test_data:
            label = process_pair(tokenizer, model, premise, hypothesis)
            f.write(f"{premise}#{hypothesis}#{label}\n")

def para_t5(test_data):
    tokenizer = AutoTokenizer.from_pretrained("ihgn/Paraphrase-Detection-T5")
    model = AutoModelForSequenceClassification.from_pretrained("ihgn/Paraphrase-Detection-T5")
    with open("output_file_para_t5.txt", 'w') as f:
        for premise, hypothesis in test_data:
            label = process_pair(tokenizer, model, premise, hypothesis)
            f.write(f"{premise}#{hypothesis}#{label}\n")

test_data = test_data_f()
#para_bert(test_data)
para_t5(test_data)