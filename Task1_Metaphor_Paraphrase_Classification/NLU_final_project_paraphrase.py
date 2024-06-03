from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

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


model_name = "Prompsit/paraphrase-bert-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

paraphrase_detector = pipeline("text-classification", model=model, tokenizer=tokenizer)

def is_paraphrase(sentence1, sentence2):
    result = paraphrase_detector(f"[CLS] {sentence1} [SEP] {sentence2} [SEP]")
    label = result[0]['label']
    score = result[0]['score']
    print(result, label, score)

test_sentence1 = "The quick brown fox jumps over the lazy dog."
test_sentence2 = "A fast, dark-colored fox leaps over a sleepy dog."

is_paraphrase(test_sentence1, test_sentence2)
