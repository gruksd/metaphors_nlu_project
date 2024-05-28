def read_labels(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        labels = [int(line.strip().split()[-1]) for line in file]
    return labels

true_labels_file = 'metaphor_paraphrase_corpus.txt'
predicted_labels_file = 'output_file.txt'

print(true_labels_file)
print(predicted_labels_file)
true_labels = read_labels(true_labels_file)
predicted_labels = read_labels(predicted_labels_file)

matches = sum(1 for true, predicted in zip(true_labels, predicted_labels) if true == predicted)

accuracy = (matches / len(true_labels)) * 100

print(f"Accuracy: {accuracy:.2f}%")