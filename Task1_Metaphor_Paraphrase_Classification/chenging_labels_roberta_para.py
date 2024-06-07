def invert_labels(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            parts = line.strip().split('#')
            if len(parts) == 3:
                sentence1, sentence2, label = parts
                new_label = '1' if label == '0' else '0'
                outfile.write(f"{sentence1}#{sentence2}#{new_label}\n")
           


input_file = 'output_file_para_roberta.txt'  
output_file = 'output_file_para_roberta_changed_labels.txt' 

invert_labels(input_file, output_file)

