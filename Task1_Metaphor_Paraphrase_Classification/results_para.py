import os

def load_data(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('#')
            if len(parts) == 3:
                key = '#'.join(parts[:2])
                label = parts[2]
                data[key] = label
    return data

true_labels_file = 'test_suit_paraphrase_0_1.txt'
predicted_files = ["output_file_para_roberta.txt"]

true_labels = load_data(true_labels_file)
predictions = [load_data(file) for file in predicted_files]

incorrect_predictions = {}

for key, true_label in true_labels.items():
    predicted_labels = [pred[key] for pred in predictions]
    if all(pred_label != true_label for pred_label in predicted_labels):
        incorrect_predictions[key] = (true_label, predicted_labels)

output_file = 'incorrect_predictions_nli.txt'

with open(output_file, 'w') as f:
    for key, (true_label, pred_labels) in incorrect_predictions.items():
        f.write(f"{key}#{true_label}#{'#'.join(pred_labels)}\n")

print(f"Incorrect predictions written to {output_file}")
