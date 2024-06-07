with open('metaphor_paraphrase_corpus.txt', 'r', encoding='utf-8') as file:
    data = file.read().strip()

blocks = data.split('\n\n')


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
                new_label = 1 #paraphrase
            if label == 2 or label == 1:
                 new_label = 0 #nonparaphrase
            result_data.append([premise, hypothesis.strip(), new_label])

with open("test_suit_paraphrase_1_0.txt", 'w', encoding='utf-8') as file:
        for premise, hypothesis, label in result_data:
            file.write(f"{premise}#{hypothesis}#{label}\n")
