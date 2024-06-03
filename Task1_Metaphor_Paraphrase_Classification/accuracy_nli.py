def read_labels(file_path):
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('#')
            if parts:
                labels.append(parts[-1])
    return labels

def accuracy_f(true_labels_file, predicted_labels_file):
    true_labels = read_labels(true_labels_file)
    predicted_labels = read_labels(predicted_labels_file)
    print(predicted_labels_file)
    if len(true_labels) == len(predicted_labels):
        matches = sum(1 for true, predicted in zip(true_labels, predicted_labels) if true == predicted)
        accuracy = (matches / len(true_labels)) * 100
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("We are missing some labels:", len(predicted_labels), "/", len(true_labels))

true_labels_file = 'test_suit.txt'
output_files = ['output_file_nli_t5.txt', 
                "output_file_nli_large_roberta.txt", "output_file_nli_multi_bert.txt", 
                "output_file_nli_large_bart.txt"]
for file in output_files:
    accuracy_f(true_labels_file, file)