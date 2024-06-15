import argparse
import evaluate
import nltk
import numpy as np
import os
import pandas as pd
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
        DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer)

def preprocess_function(
        examples, max_source_length, max_target_length, tokenizer):
    inputs = (examples["example"] + " What is the definition of "
             + examples["word"] + "?")
    
    model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
            )
    labels = tokenizer(
            text_target=examples["gloss"],
            max_length=max_target_length,
            truncation=True,
            )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def get_datasets(
        do_debug, max_source_length, max_target_length, tokenizer, train_file,
        val_file):
    train_dataset = pd.read_json(train_file)
    val_dataset = pd.read_json(val_file)

    if do_debug:
        train_dataset = train_dataset.iloc[:100]
        val_dataset = val_dataset.iloc[:100]

    preprocess_args = (max_source_length, max_target_length, tokenizer)

    train_tokenized = train_dataset.apply(
            preprocess_function,
            axis=1,
            args=preprocess_args,
            )
    val_tokenized = val_dataset.apply(
            preprocess_function,
            axis=1,
            args=preprocess_args,
            )

    return train_tokenized, val_tokenized

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
 
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
                   
    return preds, labels

def compute_metrics_factory(tokenizer, metric):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels)

        result = metric.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                tokenizer=lambda x: x.split()
                )
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [
                np.count_nonzero(pred != tokenizer.pad_token_id)
                for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
    return compute_metrics

def get_trainer(
        compute_metrics, data_collator, model, output_dir, tokenizer,
        train_dataset, val_dataset):
    training_args = Seq2SeqTrainingArguments(
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            logging_strategy="epoch",
            metric_for_best_model="eval_rouge1",
            num_train_epochs=15,
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_eval_batch_size=32,
            per_device_train_batch_size=32,
            predict_with_generate=True,
            report_to="none",
            save_strategy="epoch",
            save_total_limit=5,
            )

    trainer = Seq2SeqTrainer(
            args=training_args,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            eval_dataset=val_dataset,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            )

    return trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_name",
            choices=["facebook/bart-base", "google/flan-t5-base"])
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    metric = evaluate.load("rouge")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    compute_metrics = compute_metrics_factory(tokenizer, metric)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    output_dir = "/".join(["models", args.model_name.split("/")[1]])

    data_args = {
            "do_debug": args.debug,
            "max_source_length": 256,
            "max_target_length": 128,
            "tokenizer": tokenizer,
            "train_file": "data/train.json",
            "val_file": "data/val.json",
            }
    train_dataset, val_dataset = get_datasets(**data_args)

    run_args = {
            "compute_metrics": compute_metrics,
            "data_collator": data_collator,
            "model": model,
            "output_dir": output_dir,
            "tokenizer": tokenizer,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            }
    trainer = get_trainer(**run_args)
    trainer.train()
    trainer.save_state()

if __name__ == "__main__":
    main()
