import pandas as pd
import numpy as np
import re
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import functools, operator


#load data
Testdata_1 = pd.read_excel('/Annotation/Generated_testdata_Annika.xlsx')
Testdata_2 = pd.read_excel('/Annotation/Generated_testdata_Sofia.xlsx')
Testdata_3 = pd.read_excel('/Annotation/Generated_testdata_Mitja.ods')
Testdata_autoMetrics = pd.read_csv('testdata_generated_withMetrics.csv')
Testdata_All = Testdata_autoMetrics[["s0", "human_ans", "Paraphrase T5 E3", "Paraphrase T5 E5", "Paraphrase Bart E1", "Paraphrase Bart E3", "Loss T5 E3","Loss T5 E5","Loss Bart E1","Loss Bart E3","Bert Scores T5 E3","Bert Scores T5 E5","Bert Scores Bart E1","Bert Scores Bart E3","ChrF Scores T5 E3","ChrF Scores T5 E5","ChrF Scores Bart E1","ChrF Scores Bart E3"]]
Annotations = pd.DataFrame({ 'AnnotationT5E31':Testdata_1['Annotation T5 E3'], 'AnnotationT5E51':Testdata_1['Annotation T5 E5'], 'AnnotationBartE11':Testdata_1['Annotation Bart E1'], 'AnnotationBartE31':Testdata_1['Annotation Bart E3'], 'AnnotationT5E32':Testdata_2['Annotation T5 E3'], 'AnnotationT5E52':Testdata_2['Annotation T5 E5'], 'AnnotationBartE12':Testdata_2['Annotation Bart E1'], 'AnnotationBartE32':Testdata_2['Annotation Bart E3'], 'AnnotationT5E33':Testdata_3['Annotation T5 E3'], 'AnnotationT5E53':Testdata_3['Annotation T5 E5'], 'AnnotationBartE13':Testdata_3['Annotation Bart E1'], 'AnnotationBartE33':Testdata_3['Annotation Bart E3']})


# Create lists out of the manual annotations per model and store them in a dataframe
AnnotationsT5E3_list =[] 
AnnotationsT5E5_list =[] 
AnnotationsBartE1_list =[] 
AnnotationsBartE3_list =[] 
  
 
for index, rows in Annotations.iterrows(): 
    if index > 229:
        pass
    else:
        list_T5E3 =[rows.AnnotationT5E31, rows.AnnotationT5E32, rows.AnnotationT5E33]
        list_T5E5 =[rows.AnnotationT5E51, rows.AnnotationT5E52, rows.AnnotationT5E53]
        list_BartE1 =[rows.AnnotationBartE11, rows.AnnotationBartE12, rows.AnnotationBartE13]
        list_BartE3 =[rows.AnnotationBartE31, rows.AnnotationBartE32, rows.AnnotationBartE33]
        AnnotationsT5E3_list.append(list_T5E3) 
        AnnotationsT5E5_list.append(list_T5E5) 
        AnnotationsBartE1_list.append(list_BartE1) 
        AnnotationsBartE3_list.append(list_BartE3) 
  

Testdata_AllAnnotated = Testdata_All.head(n=230)

Testdata_AllAnnotated['Annotation T5 E3'] = AnnotationsT5E3_list
Testdata_AllAnnotated['Annotation T5 E5'] = AnnotationsT5E5_list
Testdata_AllAnnotated['Annotation Bart E1'] = AnnotationsBartE1_list
Testdata_AllAnnotated['Annotation Bart E3'] = AnnotationsBartE3_list

print(Testdata_AllAnnotated)


#CALCULATE AVERAGES FROM AUTOMATED METRICES

#LOSS

lossT5E3 = Testdata_All["Loss T5 E3"].mean()
lossT5E5 = Testdata_All["Loss T5 E5"].mean()
lossBartE1 = Testdata_All["Loss Bart E1"].mean()
lossBartE3 = Testdata_All["Loss Bart E3"].mean()

#print results into output file
sourceFile = open('Results/loss.txt', 'w')
print("Mean loss of the T5 model fine-tuned for 3 epochs on the paraphrase task: " + str(lossT5E3), file = sourceFile)
print("Mean loss of the T5 model fine-tuned for 5 epochs on the paraphrase task: " + str(lossT5E5), file = sourceFile)
print("Mean loss of the Bart model fine-tuned for 1 epoch on the paraphrase task: " + str(lossBartE1), file = sourceFile)
print("Mean loss of the Bart model fine-tuned for 3 epochs on the paraphrase task: " + str(lossBartE3), file = sourceFile)
sourceFile.close()


#BERT SCORES

#extract highest BERT scores

expression_precision = r"'precision': array\(\[\d.\d*\]\)"
expression_recall = r"'recall': array\(\[\d.\d*\]\)"

#column = rows["Bert Scores T5 E3"]
def extractBERTScores(column):
    highest_precision = []
    highest_recall = []
    for index, rows in Testdata_All.iterrows():
    #extraction of precision (unfortunately, through the export and the import the dictionary were converted into strings)
        precision_uncleaned = re.findall(expression_precision, rows[column])
        list_precision = []
        for item in precision_uncleaned:
            precision = re.findall(r"\d.\d*", item)
            list_precision.append(precision)
        list_precision2 = []
        for item in list_precision:
            item = ''.join(item)
            item = float(item)
            list_precision2.append(item)
        highest_precision.append(max(list_precision2))
        #extraction of recall
        recall_uncleaned = re.findall(expression_recall, rows[column])
        list_recall = []
        for item in recall_uncleaned:
            recall = re.findall(r"\d.\d*", item)
            list_recall.append(recall)
        list_recall2 = []
        for item in list_recall:
            item = ''.join(item)
            item = float(item)
            list_recall2.append(item)
        highest_recall.append(max(list_recall2))
    return highest_precision, highest_recall

highest_precision_T5E3, highest_recall_T5E3 = extractBERTScores("Bert Scores T5 E3")
highest_precision_T5E5, highest_recall_T5E5 = extractBERTScores("Bert Scores T5 E5")
highest_precision_BartE1, highest_recall_BartE1 = extractBERTScores("Bert Scores Bart E1")
highest_precision_BartE3, highest_recall_BartE3 = extractBERTScores("Bert Scores Bart E3")
#print(highest_precision_BartE3)

Highest_BERTscores = {"s0":Testdata_autoMetrics["s0"], "human_ans": Testdata_autoMetrics["human_ans"], "Paraphrase T5 E3": Testdata_autoMetrics["Paraphrase T5 E3"], "Paraphrase T5 E5": Testdata_autoMetrics["Paraphrase T5 E5"], "Paraphrase Bart E1": Testdata_autoMetrics["Paraphrase Bart E1"], "Paraphrase Bart E3": Testdata_autoMetrics["Paraphrase Bart E3"], 'Highest Precision T5 E3':highest_precision_T5E3,'Highest Recall T5 E3':highest_recall_T5E3,'Highest Precision T5 E5':highest_precision_T5E5,'Highest Recall T5 E5':highest_recall_T5E5, 'Highest Precision Bart E1':highest_precision_BartE1, 'Highest Recall Bart E1':highest_recall_BartE1, 'Highest Precision Bart E3':highest_precision_BartE3, 'Highest Recall Bart E3':highest_recall_BartE3}

Highest_BERTscores = pd.DataFrame(Highest_BERTscores)

#print(Highest_BERTscores.to_string(index=False))

mean_precisionT5E3 = Highest_BERTscores['Highest Precision T5 E3'].mean()
mean_recallT5E3 = Highest_BERTscores['Highest Recall T5 E3'].mean()
mean_precisionT5E5 = Highest_BERTscores['Highest Precision T5 E5'].mean()
mean_recallT5E5 = Highest_BERTscores['Highest Recall T5 E5'].mean()
mean_precisionBartE1 = Highest_BERTscores['Highest Precision Bart E1'].mean()
mean_recallBartE1 = Highest_BERTscores['Highest Recall Bart E1'].mean()
mean_precisionBartE3 = Highest_BERTscores['Highest Precision Bart E3'].mean()
mean_recallBartE3 = Highest_BERTscores['Highest Recall Bart E3'].mean()


#print BERT score means into output file
sourceFile = open('Results/precision_recall_BERTscores.txt', 'w')
print("Mean precision of the T5 model fine-tuned for 3 epochs on the paraphrase task: " + str(mean_precisionT5E3), file = sourceFile)
print("Mean recall of the T5 model fine-tuned for 3 epochs on the paraphrase task: " + str(mean_recallT5E3), file = sourceFile)
print("Mean precision of the T5 model fine-tuned for 5 epochs on the paraphrase task: " + str(mean_precisionT5E5), file = sourceFile)
print("Mean recall of the T5 model fine-tuned for 5 epochs on the paraphrase task: " + str(mean_recallT5E5), file = sourceFile)
print("Mean precision of the Bart model fine-tuned for 1 epoch on the paraphrase task: " + str(mean_precisionBartE1), file = sourceFile)
print("Mean recall of the Bart model fine-tuned for 1 epoch on the paraphrase task: " + str(mean_recallBartE1), file = sourceFile)
print("Mean precision of the Bart model fine-tuned for 3 epochs on the paraphrase task: " + str(mean_precisionBartE3), file = sourceFile)
print("Mean recall of the Bart model fine-tuned for 3 epochs on the paraphrase task: " + str(mean_recallBartE3), file = sourceFile)
sourceFile.close()


#ChrF SCORES


def extractChrf(column):
    highest_ChrF = []
    for index, row in Testdata_All.iterrows():
        row[column] = row[column].replace("[", '').replace("]", '')
        row[column] = [splits for splits in row[column].split(" ") if splits]
        for item in row[column]:
            item = float(item)
        highest_ChrF.append(float(max(row[column])))
    return highest_ChrF

ChrF_T5E3 = extractChrf("ChrF Scores T5 E3")
ChrF_T5E5 = extractChrf("ChrF Scores T5 E5")
ChrF_BartE1 = extractChrf("ChrF Scores Bart E1")
ChrF_BartE3 = extractChrf("ChrF Scores Bart E3")


Highest_ChrFscores = {"s0":Testdata_autoMetrics["s0"], "human_ans": Testdata_autoMetrics["human_ans"], "Paraphrase T5 E3": Testdata_autoMetrics["Paraphrase T5 E3"], "Paraphrase T5 E5": Testdata_autoMetrics["Paraphrase T5 E5"], "Paraphrase Bart E1": Testdata_autoMetrics["Paraphrase Bart E1"], "Paraphrase Bart E3": Testdata_autoMetrics["Paraphrase Bart E3"], 'Highest ChrF T5 E3':ChrF_T5E3, 'Highest ChrF T5 E5':ChrF_T5E5, 'Highest ChrF Bart E1':ChrF_BartE1, 'Highest ChrF Bart E3':ChrF_BartE3}

Highest_ChrFscores = pd.DataFrame(Highest_ChrFscores)

#print(Highest_ChrFscores.to_string(index=False))
#print(type(Highest_ChrFscores["Highest ChrF T5 E3"][0]))

mean_ChrFT5E3 = Highest_ChrFscores['Highest ChrF T5 E3'].mean()
mean_ChrFT5E5 = Highest_ChrFscores['Highest ChrF T5 E5'].mean()
mean_ChrFBartE1 = Highest_ChrFscores['Highest ChrF Bart E1'].mean()
mean_ChrFBartE3 = Highest_ChrFscores['Highest ChrF Bart E3'].mean()

#print ChrF score means into output file
sourceFile = open('Results/ChrFscores.txt', 'w')
print("Mean ChrF of the T5 model fine-tuned for 3 epochs on the paraphrase task: " + str(mean_ChrFT5E3), file = sourceFile)
print("Mean ChrF of the T5 model fine-tuned for 5 epochs on the paraphrase task: " + str(mean_ChrFT5E5), file = sourceFile)
print("Mean ChrF of the Bart model fine-tuned for 1 epoch on the paraphrase task: " + str(mean_ChrFBartE1), file = sourceFile)
print("Mean ChrF of the Bart model fine-tuned for 3 epochs on the paraphrase task: " + str(mean_ChrFBartE3), file = sourceFile)
sourceFile.close()


#SPEARMAN CORRELATION BETWEEN AUTOMATED METRICES

lossT5E3_list = Testdata_All["Loss T5 E3"].tolist()
lossT5E5_list = Testdata_All["Loss T5 E5"].tolist()
lossBartE1_list = Testdata_All["Loss Bart E1"].tolist()
lossBartE3_list = Testdata_All["Loss Bart E3"].tolist()


chrf_precisionT5E3 = stats.spearmanr(ChrF_T5E3, highest_precision_T5E3)
chrf_recallT5E3 = stats.spearmanr(ChrF_T5E3, highest_recall_T5E3)
chrf_lossT5E3 = stats.spearmanr(ChrF_T5E3, lossT5E3_list)
recall_precisionT5E3 = stats.spearmanr(highest_recall_T5E3, highest_precision_T5E3)
loss_precisionT5E3 = stats.spearmanr(lossT5E3_list, highest_precision_T5E3)
loss_recallT5E3 = stats.spearmanr(lossT5E3_list, highest_recall_T5E3)

chrf_precisionT5E5= stats.spearmanr(ChrF_T5E5, highest_precision_T5E5)
chrf_recallT5E5 = stats.spearmanr(ChrF_T5E5, highest_recall_T5E5)
chrf_lossT5E5 = stats.spearmanr(ChrF_T5E5, lossT5E5_list)
recall_precisionT5E5 = stats.spearmanr(highest_recall_T5E5, highest_precision_T5E5)
loss_precisionT5E5 = stats.spearmanr(lossT5E5_list, highest_precision_T5E5)
loss_recallT5E5 = stats.spearmanr(lossT5E5_list, highest_recall_T5E5)

chrf_precisionBartE1 = stats.spearmanr(ChrF_BartE1, highest_precision_BartE1)
chrf_recallBartE1 = stats.spearmanr(ChrF_BartE1, highest_recall_BartE1)
chrf_lossBartE1 = stats.spearmanr(ChrF_BartE1, lossBartE1_list)
recall_precisionBartE1 = stats.spearmanr(highest_recall_BartE1, highest_precision_BartE1)
loss_precisionBartE1 = stats.spearmanr(lossBartE1_list, highest_precision_BartE1)
loss_recallBartE1 = stats.spearmanr(lossBartE1_list, highest_recall_BartE1)

chrf_precisionBartE3 = stats.spearmanr(ChrF_BartE3, highest_precision_BartE3)
chrf_recallBartE3 = stats.spearmanr(ChrF_BartE3, highest_recall_BartE3)
chrf_lossBartE3 = stats.spearmanr(ChrF_BartE3, lossBartE3_list)
recall_precisionBartE3 = stats.spearmanr(highest_recall_BartE3, highest_precision_BartE3)
loss_precisionBartE3 = stats.spearmanr(lossBartE3_list, highest_precision_BartE3)
loss_recallBartE3 = stats.spearmanr(lossBartE3_list, highest_recall_BartE3)

#print Spearman correlation values into output file
sourceFile = open('Results/Spearman.txt', 'w')
print("Between Loss and Recall (BERT scores):", file = sourceFile)
print("T5 E3: " + str(loss_recallT5E3), file = sourceFile)
print("T5 E5: " + str(loss_recallT5E5), file = sourceFile)
print("Bart E1: " + str(loss_recallBartE1), file = sourceFile)
print("Bart E3: " + str(loss_recallBartE3), file = sourceFile)
print("Between Loss and Precision (BERT scores):", file = sourceFile)
print("T5 E3: " + str(loss_precisionT5E3), file = sourceFile)
print("T5 E5: " + str(loss_precisionT5E5), file = sourceFile)
print("Bart E1: " + str(loss_precisionBartE1), file = sourceFile)
print("Bart E3: " + str(loss_precisionBartE3), file = sourceFile)
print("Between Loss and ChrF scores:", file = sourceFile)
print("T5 E3: " + str(chrf_lossT5E3), file = sourceFile)
print("T5 E5: " + str(chrf_lossT5E5), file = sourceFile)
print("Bart E1: " + str(chrf_lossBartE1), file = sourceFile)
print("Bart E3: " + str(chrf_lossBartE3), file = sourceFile)
print("Between Precision and Recall of BERT scores:", file = sourceFile)
print("T5 E3: " + str(recall_precisionT5E3), file = sourceFile)
print("T5 E5: " + str(recall_precisionT5E5), file = sourceFile)
print("Bart E1: " + str(recall_precisionBartE1), file = sourceFile)
print("Bart E3: " + str(recall_precisionBartE3), file = sourceFile)
print("Between Recall and ChrF scores:", file = sourceFile)
print("T5 E3: " + str(chrf_recallT5E3), file = sourceFile)
print("T5 E5: " + str(chrf_recallT5E5), file = sourceFile)
print("Bart E1: " + str(chrf_recallBartE1), file = sourceFile)
print("Bart E3: " + str(chrf_recallBartE3), file = sourceFile)
print("Between Precision and ChrF scores:", file = sourceFile)
print("T5 E3: " + str(chrf_precisionT5E3), file = sourceFile)
print("T5 E5: " + str(chrf_precisionT5E5), file = sourceFile)
print("Bart E1: " + str(chrf_precisionBartE1), file = sourceFile)
print("Bart E3: " + str(chrf_precisionBartE3), file = sourceFile)
sourceFile.close()


#Create visualisations of correlation between metrices

#commented out because of long runtime of the script
#DataVisualisationT5E3 = pd.DataFrame(
#    {'Loss': lossT5E3_list,
#     'chrf': ChrF_T5E3,
#     'precision': highest_precision_T5E3,
#     'recall': highest_recall_T5E3
#    })

#correlation_plotT5E3 = sns.pairplot(DataVisualisationT5E3)
#correlation_plotT5E3.figure.savefig("correlation_plotT5E3.png")

#DataVisualisationT5E5 = pd.DataFrame(
#    {'Loss': lossT5E5_list,
#     'chrf': ChrF_T5E5,
#     'precision': highest_precision_T5E5,
#     'recall': highest_recall_T5E5
#    })

#correlation_plotT5E5 = sns.pairplot(DataVisualisationT5E5)
#correlation_plotT5E5.figure.savefig("correlation_plotT5E5.png")

#DataVisualisationBartE3 = pd.DataFrame(
#    {'Loss': lossBartE3_list,
#     'chrf': ChrF_BartE3,
#     'precision': highest_precision_BartE3,
#     'recall': highest_recall_BartE3
#    })

#correlation_plotBartE3 = sns.pairplot(DataVisualisationBartE3)
#correlation_plotBartE3.figure.savefig("correlation_plotBartE3.png")

#DataVisualisationBartE1 = pd.DataFrame(
#    {'Loss': lossBartE1_list,
#     'chrf': ChrF_BartE1,
#     'precision': highest_precision_BartE1,
#     'recall': highest_recall_BartE1
#    })

#correlation_plotBartE1 = sns.pairplot(DataVisualisationBartE1)
#correlation_plotBartE1.figure.savefig("correlation_plotBartE1.png")



#ANALYSIS OF MANUAL ANNOTATION

#Inter-Annotation Agreement: Fleiss Kappa

Annotations_T5E3 = Annotations.head(n=230)[["AnnotationT5E31","AnnotationT5E32", "AnnotationT5E33"]]
Annotations_T5E5 = Annotations.head(n=230)[["AnnotationT5E51","AnnotationT5E52", "AnnotationT5E53"]]
Annotations_BartE1 = Annotations.head(n=230)[["AnnotationBartE11","AnnotationBartE12", "AnnotationBartE13"]]
Annotations_BartE3 = Annotations.head(n=230)[["AnnotationBartE31","AnnotationBartE32", "AnnotationBartE33"]]


AnnotationT5E31 = Counter(Annotations_T5E3["AnnotationT5E31"])
AnnotationT5E32 = Counter(Annotations_T5E3["AnnotationT5E32"])
AnnotationT5E33 = Counter(Annotations_T5E3["AnnotationT5E33"])
AnnotationT5E51 = Counter(Annotations_T5E5["AnnotationT5E51"])
AnnotationT5E52 = Counter(Annotations_T5E5["AnnotationT5E52"])
AnnotationT5E53 = Counter(Annotations_T5E5["AnnotationT5E53"])
AnnotationBartE11 = Counter(Annotations_BartE1["AnnotationBartE11"])
AnnotationBartE12 = Counter(Annotations_BartE1["AnnotationBartE12"])
AnnotationBartE13 = Counter(Annotations_BartE1["AnnotationBartE13"])
AnnotationBartE31 = Counter(Annotations_BartE3["AnnotationBartE31"])
AnnotationBartE32 = Counter(Annotations_BartE3["AnnotationBartE32"])
AnnotationBartE33 = Counter(Annotations_BartE3["AnnotationBartE33"])
print(AnnotationT5E31)

def CoundSumLabels(label):
    sum = AnnotationT5E31[label] + AnnotationT5E32[label] + AnnotationT5E33[label] + AnnotationT5E51[label] + AnnotationT5E52[label] + AnnotationT5E53[label] + AnnotationBartE11[label] + AnnotationBartE12[label] + AnnotationBartE13[label] + AnnotationBartE31[label] + AnnotationBartE32[label] + AnnotationBartE33[label]
    return sum

sum_0a = CoundSumLabels("0a")
print("Number of 0a labels: " + str(sum_0a))
sum_0b = CoundSumLabels("0b")
print("Number of 0b labels: " + str(sum_0b))
sum_0c = CoundSumLabels("0c")
print("Number of 0c labels: " + str(sum_0c))
sum_1a = CoundSumLabels("1a")
print("Number of 1a labels: " + str(sum_1a))
sum_1b = CoundSumLabels("1b")
print("Number of 1b labels: " + str(sum_1b))
sum_2a = CoundSumLabels("2a")
print("Number of 2a labels: " + str(sum_2a))
sum_3a = CoundSumLabels("3a")
print("Number of 3a labels: " + str(sum_3a))
sum_3b = CoundSumLabels("3b")
print("Number of 3b labels: " + str(sum_3b))

sum_ALL = sum_0a + sum_0b + sum_0c + sum_1a + sum_1b + sum_2a + sum_3a + sum_3b
print("Number of all labels: " + str(sum_ALL))

Pe = (sum_0a/sum_ALL)**2 + (sum_0b/sum_ALL)**2 + (sum_0c/sum_ALL)**2 + (sum_1a/sum_ALL)**2 + (sum_1b/sum_ALL)**2 + (sum_2a/sum_ALL)**2 + (sum_3a/sum_ALL)**2 + (sum_3b/sum_ALL)**2
#print(Pe)

part1_Po = 1/(230*3*(3-1))
part2_Po = 0
for index, row in Annotations_T5E3.iterrows():
    labelsinrow = Counter(row)
    part2_Po += labelsinrow["0a"]**2 + labelsinrow["0b"]**2 + labelsinrow["0c"]**2 + labelsinrow["1a"]**2 + labelsinrow["1b"]**2 + labelsinrow["2a"]**2 + labelsinrow["3a"]**2 + labelsinrow["3b"]**2
#print(part2_Po)
part3_Po = 230*3

Po = part1_Po*(part2_Po - part3_Po)
#print(Po)

Fleiss_Kappa = (Po-Pe)/(1-Pe)
print(Fleiss_Kappa)


# Vocabulary size of models vs. annotators
Vocabulary_T5E3 = Counter(Testdata_All["Paraphrase T5 E3"])
print(len(Vocabulary_T5E3))

Vocabulary_T5E5 = Counter(Testdata_All["Paraphrase T5 E5"])
print(len(Vocabulary_T5E5))

Vocabulary_BartE1 = Counter(Testdata_All["Paraphrase Bart E1"])
print(len(Vocabulary_BartE1))

Vocabulary_BartE3 = Counter(Testdata_All["Paraphrase Bart E3"])
print(len(Vocabulary_BartE3))

list_counts1 = [Vocabulary_T5E3, Vocabulary_T5E5, Vocabulary_BartE1, Vocabulary_BartE3]
Vocabulary_Models = dict(functools.reduce(operator.add,
         map(Counter, list_counts1)))

print("Size of vocabulary of all 4 models together: " + str(len(Vocabulary_Models)))

list_counts = []
for index, row in Testdata_All.iterrows():
    row["human_ans"] = row["human_ans"].replace("[", '').replace("]", '')
    row["human_ans"] = [splits for splits in row["human_ans"].split(" ") if splits]
    annotations_per_row = Counter(row["human_ans"])
    list_counts.append(annotations_per_row)

Vocabulary_Humans = dict(functools.reduce(operator.add,
         map(Counter, list_counts)))

print("Size of vocabulary of all human annotators together: " + str(len(Vocabulary_Humans)))


#Extract words from the  

list_focusword = []
for index, row in Testdata_AllAnnotated.iterrows():
  focusword = re.findall(r"\*\w*.\w*\*", row["s0"])
  list_focusword.append(focusword[0].replace("*", ''))
Testdata_AllAnnotated['focusword'] = list_focusword
print(Testdata_AllAnnotated['focusword'])


list_1a_paraphrasesT5E3 = []
list_1b_paraphrasesT5E3 = []
list_0c_paraphrasesT5E3 = []
list_2a_paraphrasesT5E3 = []
list_3a_b_paraphrasesT5E3 = []

for index, row in Testdata_AllAnnotated.iterrows():
    previous_items = []
    for item in row['Annotation T5 E3']:
        if item in previous_items:
            pass
        elif item == "0c":
            list_0c_paraphrasesT5E3.append([row["focusword"], row['Paraphrase T5 E3']])
        elif item == "1a":
            list_1a_paraphrasesT5E3.append([row["focusword"], row['Paraphrase T5 E3']])
        elif item == "1b":
            list_1b_paraphrasesT5E3.append([row["focusword"], row['Paraphrase T5 E3']])
        elif item == "2a":
            list_2a_paraphrasesT5E3.append([row["focusword"], row['Paraphrase T5 E3']])
        elif item == "3a" or item == "3b":
            list_3a_b_paraphrasesT5E3.append([row["focusword"], row['Paraphrase T5 E3']])
        else:
            pass
        previous_items.append(item)

print(list_1a_paraphrasesT5E3)
print(len(list_1a_paraphrasesT5E3))

print(list_3a_b_paraphrasesT5E3)
print(len(list_3a_b_paraphrasesT5E3))

print(AnnotationT5E31)
print(AnnotationT5E32)
print(AnnotationT5E33)


#Paraphrases with high scores

lowestloss_T5E3 = Testdata_All.sort_values(by="Loss T5 E3", ascending = False).head(10)
lowestloss_T5E5 = Testdata_All.sort_values(by="Loss T5 E5", ascending = False).head(10)
lowestloss_BartE1 = Testdata_All.sort_values(by="Loss Bart E1", ascending = False).head(10)
lowestloss_BartE3 = Testdata_All.sort_values(by="Loss Bart E3", ascending = False).head(10)

lowestloss = pd.concat([lowestloss_T5E3, lowestloss_T5E5, lowestloss_BartE1, lowestloss_BartE3], ignore_index=True)

lowestloss.to_csv('Results/lowestloss.csv', index=False) 

lowestloss_T5E3 = Testdata_All.sort_values(by="Loss T5 E3", ascending = True).head(10)
lowestloss_T5E5 = Testdata_All.sort_values(by="Loss T5 E5", ascending = True).head(10)
lowestloss_BartE1 = Testdata_All.sort_values(by="Loss Bart E1", ascending = True).head(10)
lowestloss_BartE3 = Testdata_All.sort_values(by="Loss Bart E3", ascending = True).head(10)

lowestloss = pd.concat([lowestloss_T5E3, lowestloss_T5E5, lowestloss_BartE1, lowestloss_BartE3], ignore_index=True)

lowestloss.to_csv('Results/lowestloss.csv', index=False) 


highest_BERTprecision_T5E3 = Highest_BERTscores.sort_values(by='Highest Precision T5 E3', ascending=False).head(10)
highest_BERTprecision_T5E5 = Highest_BERTscores.sort_values(by='Highest Precision T5 E5', ascending=False).head(10)
highest_BERTprecision_BartE1 = Highest_BERTscores.sort_values(by='Highest Precision Bart E1', ascending=False).head(10)
highest_BERTprecision_BartE3 = Highest_BERTscores.sort_values(by='Highest Precision Bart E3', ascending=False).head(10)

highest_BERTprecision = pd.concat([highest_BERTprecision_T5E3, highest_BERTprecision_T5E5, highest_BERTprecision_BartE1, highest_BERTprecision_BartE3], ignore_index=True)

highest_BERTprecision.to_csv('Results/highest_BERTprecision.csv', index=False)

highest_BERTrecall_T5E3 = Highest_BERTscores.sort_values(by='Highest Recall T5 E3', ascending=False).head(10)
highest_BERTrecall_T5E5 = Highest_BERTscores.sort_values(by='Highest Recall T5 E5', ascending=False).head(10)
highest_BERTrecall_BartE1 = Highest_BERTscores.sort_values(by='Highest Recall Bart E1', ascending=False).head(10)
highest_BERTrecall_BartE3 = Highest_BERTscores.sort_values(by='Highest Recall Bart E3', ascending=False).head(10)

highest_BERTrecall = pd.concat([highest_BERTrecall_T5E3, highest_BERTrecall_T5E5, highest_BERTrecall_BartE1, highest_BERTrecall_BartE3], ignore_index=True)

highest_BERTrecall.to_csv('Results/highest_BERTrecall.csv', index=False)

Highest_ChrFscores_T5E3 = Highest_ChrFscores.sort_values(by='Highest ChrF T5 E3', ascending=False).head(10)
Highest_ChrFscores_T5E5 = Highest_ChrFscores.sort_values(by='Highest ChrF T5 E5', ascending=False).head(10)
Highest_ChrFscores_BartE1 = Highest_ChrFscores.sort_values(by='Highest ChrF Bart E1', ascending=False).head(10)
Highest_ChrFscores_BartE3 = Highest_ChrFscores.sort_values(by='Highest ChrF Bart E3', ascending=False).head(10)

highest_ChrF = pd.concat([Highest_ChrFscores_T5E3, Highest_ChrFscores_T5E5, Highest_ChrFscores_BartE1, Highest_ChrFscores_BartE3], ignore_index=True)

highest_ChrF.to_csv('Results/highest_ChrF.csv', index=False)

Testdata_AllAnnotated.to_csv('Testdata_AllAnnotated.csv', index=False)


