import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer
import evaluate
from evaluate import load

testdata=pd.read_excel('Generated_testdata.xlsx')

testdata = Dataset.from_pandas(testdata)

a = testdata[6]["human_ans"]
print(a)
print(type(a))

def ListCreation(example):
  cleaned = []
  example["human_ans"] = example["human_ans"].split(" ")
  for item in example["human_ans"]:
    #item = item.replace("[^a-zA-Z0-9 \n\.]", '')
    item = item.translate ({ord(c): "" for c in "['] "})
    cleaned.append(item)
    #item = item.replace('[', '')
    #item = item.replace(']', '')
  example["human_ans"] = cleaned
  return example

testdata = testdata.map(ListCreation)
b = testdata[6]["human_ans"]
print(b)

print(testdata[0:5])

def TestMatch(example):
  for item in example['human_ans']:
    if example['Paraphrase T5 E3'].lower() in item.lower() and len(example['Paraphrase T5 E3']) == len(item):
      example['Match T5 E3'] = 1
    else:
      example['Match T5 E3'] = 0
    if example['Paraphrase T5 E5'].lower() in item.lower() and len(example['Paraphrase T5 E5']) == len(item):
      example['Match T5 E5'] = 1
    else:
      example['Match T5 E5'] = 0
    if example['Paraphrase Bart E1'].lower() in item.lower() and len(example['Paraphrase Bart E1']) == len(item):
      example['Match Bart E1'] = 1
    else:
      example['Match Bart E1'] = 0
    if example['Paraphrase Bart E3'].lower() in item.lower() and len(example['Paraphrase Bart E3']) == len(item):
      example['Match Bart E3'] = 1
    else:
      example['Match Bart E3'] = 0
  return example

testdata = testdata.map(TestMatch)

print(testdata)

Testdata_exactMatchT5E3 = testdata.filter(lambda example: example["Match T5 E3"]==1)
Testdata_exactMatchT5E5 = testdata.filter(lambda example: example["Match T5 E5"]==1)
Testdata_exactMatchBartE1 = testdata.filter(lambda example: example["Match Bart E1"]==1)
Testdata_exactMatchBartE3 = testdata.filter(lambda example: example["Match Bart E3"]==1)

print(Testdata_exactMatchT5E3)
print(Testdata_exactMatchT5E5)
print(Testdata_exactMatchBartE1)
print(Testdata_exactMatchBartE3)


print(Testdata_exactMatchT5E3[0:5])
print(Testdata_exactMatchT5E5[0:5])
print(Testdata_exactMatchBartE1[0:5])
print(Testdata_exactMatchBartE3[0:5])


#calculate percentage of exact matches
AccuracyT5E3 = len(Testdata_exactMatchT5E3)/len(testdata)
print("Accuracy of the T5 model fine-tuned for 3 epochs on the paraphrase task: " + str(AccuracyT5E3))
AccuracyT5E5 = len(Testdata_exactMatchT5E5)/len(testdata)
print("Accuracy of the T5 model fine-tuned for 5 epochs on the paraphrase task: " + str(AccuracyT5E5))
AccuracyBartE1 = len(Testdata_exactMatchBartE1)/len(testdata)
print("Accuracy of the Bart model fine-tuned for 1 epochs on the paraphrase task: " + str(AccuracyBartE1))
AccuracyBartE3 = len(Testdata_exactMatchBartE3)/len(testdata)
print("Accuracy of the Bart model fine-tuned for 3 epochs on the paraphrase task: " + str(AccuracyBartE3))


#print results into output file
sourceFile = open('Results/accuracy.txt', 'w')
print("Accuracy of the T5 model fine-tuned for 3 epochs on the paraphrase task: " + str(AccuracyT5E3), file = sourceFile)
print("Accuracy of the T5 model fine-tuned for 5 epochs on the paraphrase task: " + str(AccuracyT5E5), file = sourceFile)
print("Accuracy of the Bart model fine-tuned for 1 epoch on the paraphrase task: " + str(AccuracyBartE1), file = sourceFile)
print("Accuracy of the Bart model fine-tuned for 3 epochs on the paraphrase task: " + str(AccuracyBartE3), file = sourceFile)
sourceFile.close()

#CALCULATE CROSS ENTROPY LOSS

# Load T5 tokenizer directly
T5tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

#Load T5 models

model_T5_finetuned_E3 = T5ForConditionalGeneration.from_pretrained("C:/Users/angr05/Desktop/University/LT/NLU/training/Model3Epochs")
model_T5_finetuned_E5 = T5ForConditionalGeneration.from_pretrained("C:/Users/angr05/Desktop/University/LT/NLU/training/Model5Epochs")

#Load Bart tokenizer
Barttokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')

#Load Bart models
model_Bart_finetuned_E1 = BartForConditionalGeneration.from_pretrained("C:/Users/angr05/Desktop/University/LT/NLU/training/Bart_1Epoch")
model_Bart_finetuned_E3 = BartForConditionalGeneration.from_pretrained("C:/Users/angr05/Desktop/University/LT/NLU/training/Bart_3Epoch")

def CrossEntropyT5 (example):
  crossentropy_list_T5_E3 = []
  crossentropy_list_T5_E5 = []
  for item in example["human_ans"]:
    label = T5tokenizer.encode(item, return_tensors="pt")
    input_T5_E3 = T5tokenizer.encode(example["Paraphrase T5 E3"], return_tensors="pt")
    input_T5_E5 = T5tokenizer.encode(example["Paraphrase T5 E5"], return_tensors="pt")
    outputs_T5_E3 = model_T5_finetuned_E3(input_T5_E3, labels=label)
    outputs_T5_E5 = model_T5_finetuned_E5(input_T5_E5, labels=label)
    example["Loss T5 E3"] = outputs_T5_E3["loss"]
    example["Loss T5 E5"] = outputs_T5_E5["loss"]
  return example

def CrossEntropyBart (example):
  crossentropy_list_Bart_E1 = []
  crossentropy_list_Bart_E3 = []
  for item in example["human_ans"]:
    label = Barttokenizer.encode(item, return_tensors="pt")
    input_Bart_E1 = Barttokenizer.encode(example["Paraphrase Bart E1"], return_tensors="pt")
    input_Bart_E3 = Barttokenizer.encode(example["Paraphrase Bart E3"], return_tensors="pt")
    outputs_Bart_E1 = model_Bart_finetuned_E1(input_Bart_E1, labels=label)
    outputs_Bart_E3 = model_Bart_finetuned_E3(input_Bart_E3, labels=label)
    example["Loss Bart E1"] = outputs_Bart_E1["loss"]
    example["Loss Bart E3"] = outputs_Bart_E3["loss"]
  return example

testdata = testdata.map(CrossEntropyT5)
testdata = testdata.map(CrossEntropyBart)

print(testdata[0:5])

bertscore = load("bertscore")
chrf =  evaluate.load("chrf")

def Add_Evaluation_BERT (example):
  bert_scores_list_T5_E3 = []
  bert_scores_list_T5_E5 = []
  bert_scores_list_Bart_E1 = []
  bert_scores_list_Bart_E3 = []
  for item in example["human_ans"]:
    predictions_T5_E3 = [example["Paraphrase T5 E3"]]
    predictions_T5_E5 = [example["Paraphrase T5 E5"]]
    predictions_Bart_E1 = [example["Paraphrase Bart E1"]]
    predictions_Bart_E3 = [example["Paraphrase Bart E3"]]
    references = [item]
    bertscores_T5_E3 = bertscore.compute(predictions=predictions_T5_E3, references=references, lang="en")
    bertscores_T5_E5 = bertscore.compute(predictions=predictions_T5_E5, references=references, lang="en")
    bertscores_Bart_E1 = bertscore.compute(predictions=predictions_Bart_E1, references=references, lang="en")
    bertscores_Bart_E3 = bertscore.compute(predictions=predictions_Bart_E3, references=references, lang="en")
    bert_scores_list_T5_E3.append(bertscores_T5_E3)
    bert_scores_list_T5_E5.append(bertscores_T5_E5)
    bert_scores_list_Bart_E1.append(bertscores_Bart_E1)
    bert_scores_list_Bart_E3.append(bertscores_Bart_E3)
  example["Bert Scores T5 E3"] = bert_scores_list_T5_E3
  example["Bert Scores T5 E5"] = bert_scores_list_T5_E5
  example["Bert Scores Bart E1"] = bert_scores_list_Bart_E1
  example["Bert Scores Bart E3"] = bert_scores_list_Bart_E3
  return example


def Add_Evaluation_ChrF (example):
  chrf_scores_list_T5_E3 = []
  chrf_scores_list_T5_E5 = []
  chrf_scores_list_Bart_E1 = []
  chrf_scores_list_Bart_E3 = []
  for item in example["human_ans"]:
    predictions_T5_E3 = [example["Paraphrase T5 E3"]]
    predictions_T5_E5 = [example["Paraphrase T5 E5"]]
    predictions_Bart_E1 = [example["Paraphrase Bart E1"]]
    predictions_Bart_E3 = [example["Paraphrase Bart E3"]]
    references = [item]
    chrfscores_T5_E3 = chrf.compute(predictions=predictions_T5_E3, references=references)
    chrfscores_T5_E5 = chrf.compute(predictions=predictions_T5_E5, references=references)
    chrfscores_Bart_E1 = chrf.compute(predictions=predictions_Bart_E1, references=references)
    chrfscores_Bart_E3 = chrf.compute(predictions=predictions_Bart_E3, references=references)
    chrf_scores_list_T5_E3.append(chrfscores_T5_E3['score'])
    chrf_scores_list_T5_E5.append(chrfscores_T5_E5['score'])
    chrf_scores_list_Bart_E1.append(chrfscores_Bart_E1['score'])
    chrf_scores_list_Bart_E3.append(chrfscores_Bart_E3['score'])
  example["ChrF Scores T5 E3"] = chrf_scores_list_T5_E3
  example["ChrF Scores T5 E5"] = chrf_scores_list_T5_E5
  example["ChrF Scores Bart E1"] = chrf_scores_list_Bart_E1
  example["ChrF Scores Bart E3"] = chrf_scores_list_Bart_E3
  return example

testdata = testdata.map(Add_Evaluation_BERT)
testdata = testdata.map(Add_Evaluation_ChrF)
print(testdata[0])

testdata.to_csv("testdata_generated_withMetrics.csv")