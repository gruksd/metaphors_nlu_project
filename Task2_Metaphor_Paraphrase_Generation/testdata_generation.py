import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer

testdata=pd.read_csv('/for_generation.csv',sep=',')

testdata = testdata.drop(['idx', 'novelty', 'sid', 'i0'], axis=1)

testdata['human_ans'] = testdata.human_ans.apply(lambda x: x[0:].split(' '))
print(testdata)


dataset_test = Dataset.from_pandas(testdata)

print(dataset_test)

def replaceHighlight(example):
  example["s0"] = example["s0"].replace("<b>", "*")
  example["s0"] = example["s0"].replace("</b>", "*")
  return example

dataset_test = dataset_test.map(replaceHighlight)

print(dataset_test[0])

# Load T5 tokenizer directly
T5tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

#Load T5 models

model_T5_finetuned_E3 = T5ForConditionalGeneration.from_pretrained("/training/T5_E3")
model_T5_finetuned_E5 = T5ForConditionalGeneration.from_pretrained("/training/T5_E5")
model_T5 = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

#Test T5 models
input_sentence = "Paraphrase highlighted word: She said *firmly*."
input = T5tokenizer(input_sentence, return_tensors="pt")
output_E3 = model_T5_finetuned_E3.generate(**input, max_new_tokens=3)
output_E5 = model_T5_finetuned_E5.generate(**input, max_new_tokens=3)
output = model_T5.generate(**input, max_new_tokens=3)
print(T5tokenizer.decode(output_E3[0]))
print(T5tokenizer.decode(output_E5[0]))
print(T5tokenizer.decode(output[0]))


#Load Bart tokenizer
BartTokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')

#Load Bart models
model_Bart_finetuned_E1 = BartForConditionalGeneration.from_pretrained("/training/Bart_1E")
model_Bart_finetuned_E3 = BartForConditionalGeneration.from_pretrained("/training/Bart_3E")
model_Bart = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')

#Test Bart models
input_sentence = "Paraphrase the highlighted word: She said *firmly*."
input = BartTokenizer(input_sentence, return_tensors="pt")
output_E1 = model_Bart_finetuned_E1.generate(**input, max_new_tokens=10)
output_E3 = model_Bart_finetuned_E3.generate(**input, max_new_tokens=10)
output = model_Bart.generate(**input, max_new_tokens=10)
print(BartTokenizer.decode(output_E1[0]))
print(BartTokenizer.decode(output_E3[0]))
print(BartTokenizer.decode(output[0]))

def GenerateTestT5(example):
  input_sentence = "Paraphrase the highlighted word: " + example["s0"]
  input_T5 = T5tokenizer(input_sentence, return_tensors="pt")
  output_T5 = model_T5.generate(**input_T5)
  output_T5_E3 = model_T5_finetuned_E3.generate(**input_T5)
  output_T5_E5 = model_T5_finetuned_E5.generate(**input_T5)
  example["Paraphrase T5"] = T5tokenizer.decode(output_T5[0])
  example["Paraphrase T5 E3"] = T5tokenizer.decode(output_T5_E3[0])
  example["Paraphrase T5 E5"] = T5tokenizer.decode(output_T5_E5[0])
  return example

dataset_test = dataset_test.map(GenerateTestT5)

def GenerateTestBart(example):
  input_sentence = "Paraphrase the highlighted word: " + example["s0"]
  input_Bart = BartTokenizer(input_sentence, return_tensors="pt")
  output_Bart = model_Bart.generate(**input_Bart)
  example["Paraphrase Bart"] = BartTokenizer.decode(output_Bart[0])
  output_Bart_E1 = model_Bart_finetuned_E1.generate(**input_Bart)
  example["Paraphrase Bart E1"] = BartTokenizer.decode(output_Bart_E1[0])
  output_Bart_E3 = model_Bart_finetuned_E3.generate(**input_Bart)
  example["Paraphrase Bart E3"] = BartTokenizer.decode(output_Bart_E3[0])
  return example

dataset_test = dataset_test.map(GenerateTestBart)

print(dataset_test[0])

def CleanParaphrase(example):
  example["Paraphrase T5"] = example["Paraphrase T5"].replace("<pad> ", "")
  example["Paraphrase T5 E3"] = example["Paraphrase T5 E3"].replace("<pad> ", "")
  example["Paraphrase T5 E5"] = example["Paraphrase T5 E5"].replace("<pad> ", "")
  example["Paraphrase Bart"] = example["Paraphrase Bart"].replace("<pad> ", "")
  example["Paraphrase Bart E1"] = example["Paraphrase Bart E1"].replace("<pad> ", "")
  example["Paraphrase Bart E3"] = example["Paraphrase Bart E3"].replace("<pad> ", "")
  example["Paraphrase T5"] = example["Paraphrase T5"].replace("<pad>", "")
  example["Paraphrase T5 E3"] = example["Paraphrase T5 E3"].replace("<pad>", "")
  example["Paraphrase T5 E5"] = example["Paraphrase T5 E5"].replace("<pad>", "")
  example["Paraphrase Bart"] = example["Paraphrase Bart"].replace("<pad>", "")
  example["Paraphrase Bart E1"] = example["Paraphrase Bart E1"].replace("<pad>", "")
  example["Paraphrase Bart E3"] = example["Paraphrase Bart E3"].replace("<pad>", "")
  example["Paraphrase T5"] = example["Paraphrase T5"].replace("</s>", "")
  example["Paraphrase T5 E3"] = example["Paraphrase T5 E3"].replace("</s>", "")
  example["Paraphrase T5 E5"] = example["Paraphrase T5 E5"].replace("</s>", "")
  example["Paraphrase Bart"] = example["Paraphrase Bart"].replace("</s>", "")
  example["Paraphrase Bart E1"] = example["Paraphrase Bart E1"].replace("</s>", "")
  example["Paraphrase Bart E3"] = example["Paraphrase Bart E3"].replace("</s>", "")
  example["Paraphrase Bart"] = example["Paraphrase Bart"].replace("<s>", "")
  example["Paraphrase Bart E1"] = example["Paraphrase Bart E1"].replace("<s>", "")
  example["Paraphrase Bart E3"] = example["Paraphrase Bart E3"].replace("<s>", "")
  return example

dataset_test = dataset_test.map(CleanParaphrase)

print(dataset_test[0])

dataset_test.to_csv("testdata_generated.csv")
