import argparse
import evaluate
import nltk
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def get_dataloader(batch_size, data, max_length, tokenizer):
    prompts = (data["example"] + " What is the definition of "
                + data["word"] + "?").to_list()

    model_inputs = tokenizer(
            text=prompts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            )
    target_ids = tokenizer(
            data["word"].to_list(),
            add_special_tokens=False,
            ).input_ids
    target_ids = torch.tensor([el[-1] for el in target_ids])

    dataset = torch.utils.data.TensorDataset(
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            target_ids,
            )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    return dataloader

@torch.no_grad
def get_preds(
        batch_size, data, device, max_length, max_new_tokens, model,
        tokenizer):
    dataloader = get_dataloader(batch_size, data, max_length, tokenizer)
    
    model.to(device)

    preds = []
    for inp, att, tgt in tqdm(dataloader):
        input_ids = inp.to(device)
        attention_mask = att.to(device)
        targetwords = tgt.to(device) 
        bad = [[el] for el in targetwords.tolist()]
        outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bad_words_ids=bad,
                max_new_tokens=max_new_tokens,
                )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds.extend(decoded)

    return preds

def compute_metrics(metrics, preds, refs):
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

    scores = {
            "bleu": results_sacrebleu["score"],
            "rougeL": results_rouge["rougeL"] * 100,
            "bertF1": np.mean([res * 100 for res in results_bertscore["f1"]]),
            }
    return scores

def get_scores_metaphor(data, metrics):
    preds = data["pred"].to_list()
    refs_met = data["metaphorical_gloss"].to_list()
    refs_lit = data["literal_gloss"].to_list()

    scores_met_total = {"bleu": [], "rougeL": [], "bertF1": []}
    scores_lit_total = {"bleu": [], "rougeL": [], "bertF1": []}

    for i in range(len(preds)):
        scores_met = compute_metrics(metrics, [preds[i]], [refs_met[i]])
        scores_lit = compute_metrics(metrics, [preds[i]], [refs_lit[i]])
        for key, val in scores_met.items():
            scores_met_total[key].append(val)
        for key, val in scores_lit.items():
            scores_lit_total[key].append(val)

    scores_total = {
            "bleu_met": scores_met_total["bleu"],
            "bleu_lit": scores_lit_total["bleu"],
            "rougeL_met": scores_met_total["rougeL"],
            "rougeL_lit": scores_lit_total["rougeL"],
            "bertF1_met": scores_met_total["bertF1"],
            "bertF1_lit": scores_lit_total["bertF1"],
            }
    return pd.DataFrame(scores_total)

def get_scores_naacl(data, metrics):
    preds = data["pred"].to_list()
    refs = data["gloss"].to_list()

    scores = compute_metrics(metrics, preds, refs)
    return pd.DataFrame(scores, index=[0])

def get_scores(data, dataset_name, metrics):
    if dataset_name == "metaphor_paraphrase":
        return get_scores_metaphor(data, metrics)
    else:
        return get_scores_naacl(data, metrics)

def get_tokenizer(model_name):
    if "flan-t5" in model_name:
        return AutoTokenizer.from_pretrained("google/flan-t5-base")
    elif "bart" in model_name:
        return AutoTokenizer.from_pretrained("facebook/bart-base")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dataset_name = args.data_path.split(".")[0].split("/")[1]
    model_name = args.model_name_or_path.split("/")[1]
    preds_path = f"preds/{dataset_name}/{model_name}_{dataset_name}.json"
    scores_path = f"scores/{dataset_name}/{model_name}_{dataset_name}.json"

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    tokenizer = get_tokenizer(model_name)
    device = (torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu"))
  
    data = pd.read_json(args.data_path)
    gen_args = {
            "batch_size": 16,
            "data": data,
            "device": device,
            "max_length": 256,
            "max_new_tokens": 32,
            "model": model,
            "tokenizer": tokenizer,
            }
    data["pred"] = get_preds(**gen_args)
    metrics = {
            "sacrebleu": evaluate.load("sacrebleu"),
            "rouge": evaluate.load("rouge"),
            "bertscore": evaluate.load("bertscore"),
            }
    scores = get_scores(data, dataset_name, metrics)

    data.to_json(preds_path, indent=4, orient="records")
    scores.to_json(scores_path, indent=4, orient="records")

if __name__ == "__main__":
    main()
