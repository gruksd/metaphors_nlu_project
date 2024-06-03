from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Попробуем загрузить другую модель и токенайзер для NLI
alternative_model_name = "textattack/gpt2-nli"

try:
    model = AutoModelForSequenceClassification.from_pretrained(alternative_model_name)
    tokenizer = AutoTokenizer.from_pretrained(alternative_model_name)

    # Пример тестовых данных
    test_data = [
        {"premise": "A man inspects the uniform of a figure in some East Asian country.", "hypothesis": "The man is sleeping."},
        {"premise": "An older and younger man smiling.", "hypothesis": "Two men are smiling and laughing at the cats playing on the floor."},
        {"premise": "A soccer game with multiple males playing.", "hypothesis": "Some men are playing a sport."},
        {"premise": "A smiling costumed woman is holding an umbrella.", "hypothesis": "A happy woman in a fairy costume holds an umbrella."}
    ]

    # Токенизация данных
    def tokenize_data(premise, hypothesis):
        return tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)

    # Получение предсказаний модели
    def get_prediction(premise, hypothesis):
        inputs = tokenize_data(premise, hypothesis)
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        return probabilities

    # Проверка модели на тестовых данных
    for data in test_data:
        premise = data["premise"]
        hypothesis = data["hypothesis"]
        probabilities = get_prediction(premise, hypothesis)
        print(f"Premise: {premise}")
        print(f"Hypothesis: {hypothesis}")
        print(f"Probabilities: {probabilities}\n")

except EnvironmentError as e:
    print(f"Error loading model or tokenizer: {e}")
