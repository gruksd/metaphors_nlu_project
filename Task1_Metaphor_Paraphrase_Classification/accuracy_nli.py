def read_labels(file_path):
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('#')
            if parts:
                labels.append(parts[-1])
    return labels

true_labels_file = 'test_suit.txt'
predicted_labels_file = 'output_file.txt'

true_labels = read_labels(true_labels_file)
predicted_labels = read_labels(predicted_labels_file)

matches = sum(1 for true, predicted in zip(true_labels, predicted_labels) if true == predicted)

accuracy = (matches / len(true_labels)) * 100

print(f"Accuracy: {accuracy:.2f}%")