import evaluate
import nltk
import numpy as np
import pandas as pd
import sys

def compute_metrics(gloss_type: str, metrics: dict, preds, refs):
    # rougeLSum expects newline after each sentence
    preds_rouge = ["\n".join(nltk.sent_tokenize(pred))
            for pred in preds]
    refs_rouge = ["\n".join(nltk.sent_tokenize(ref))
            for ref in refs]

    results_sacrebleu = metrics["sacrebleu"].compute(
            predictions=preds,
            references=refs,
            )
    results_rouge = metrics["rouge"].compute(
            predictions=preds_rouge,
            references=refs_rouge,
            tokenizer=lambda x: x.split()
            )
    results_bertscore = metrics["bertscore"].compute(
            predictions=preds,
            references=refs,
            model_type="distilbert-base-uncased",
            )

    bleu = results_sacrebleu["score"]
    rouge_l = results_rouge["rougeL"] * 100
    bert_f1 = results_bertscore["f1"][0] * 100

    if gloss_type == "metaphorical":
        return {
            "bleu_met": bleu,
            "rougeL_met": rouge_l,
            "bertF1_met": bert_f1,
            }
    elif gloss_type == "literal":
        return {
            "bleu_lit": bleu,
            "rougeL_lit": rouge_l,
            "bertF1_lit": bert_f1,
            }
    else:
        raise ValueError

def get_scores(data, metrics):
    preds = data["pred"].to_list()
    refs_met = data["metaphorical_gloss"].to_list()
    refs_lit = data["literal_gloss"].to_list()

    scores_total = {
            "bleu_met": [], "rougeL_met": [], "bertF1_met": [],
            "bleu_lit": [], "rougeL_lit": [], "bertF1_lit": [],
            }

    for i in range(len(preds)):
        scores_met = compute_metrics("metaphorical", metrics, [preds[i]],
                [refs_met[i]])
        scores_lit = compute_metrics("literal", metrics, [preds[i]],
                [refs_lit[i]])
        for key, val in scores_met.items():
            scores_total[key].append(val)
        for key, val in scores_lit.items():
            scores_total[key].append(val)

    return pd.DataFrame(scores_total)

def main():
    datafile = sys.argv[1]
    output_name = datafile.split("/")[-1].split(".")[0]

    data = pd.read_json(datafile)
    data = data.drop(data[data.metaphorical_gloss == data.literal_gloss].index)
    data = data.reset_index(drop=True)

    metrics = {
        "sacrebleu": evaluate.load("sacrebleu"),
        "rouge": evaluate.load("rouge"),
        "bertscore": evaluate.load("bertscore"),
        }
    scores = get_scores(data, metrics)

    data = data.join(scores)

    data.to_json(f"./preds/scored/{output_name}_scored.json", indent=4,
        orient="records")

if __name__ == "__main__":
    main()
