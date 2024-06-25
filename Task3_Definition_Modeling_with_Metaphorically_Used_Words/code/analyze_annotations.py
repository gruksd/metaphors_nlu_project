import argparse
import numpy as np
import pandas as pd

def determine_selection(row):
    if not row["inverted"]:
        a = "metaphorical_gloss"
        b = "literal_gloss"
    else:
        a = "literal_gloss"
        b = "metaphorical_gloss"
    if row["more_similar"] == "a":
        return a
    elif row["more_similar"] == "b":
        return b
    else:
        return np.nan

def process(df, annkey):
    new_df = df.join(annkey["inverted"])
    new_df = new_df.drop(new_df[new_df.gloss2a == new_df.gloss2b].index)
    new_df = new_df.reset_index(drop=True)
    new_df["selection"] = new_df.apply(determine_selection, axis=1)
    return new_df

def join_annotations(annset1, annset2, annset3):
    annotations = []
    for i in range(len(annset1)):
        annotations.append(
                [
                    annset1.iloc[i]["selection"],
                    annset2.iloc[i]["selection"],
                    annset3.iloc[i]["selection"],
                    ]
                )
    return pd.Series(annotations)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann1")
    parser.add_argument("--ann2")
    parser.add_argument("--ann3")
    parser.add_argument("--model")
    args = parser.parse_args()

    data_path = f"./annotation/completed/{args.model}_annset"
    key_path = f"./annotation/keys/{args.model}_annkey.csv"
    preds_path = (f"./preds/scored/{args.model}"
            + "_metaphor_paraphrase_scored.json")

    anndata1 = pd.read_excel(f"{data_path}_{args.ann1}.ods")
    anndata2 = pd.read_excel(f"{data_path}_{args.ann2}.ods")
    anndata3 = pd.read_excel(f"{data_path}_{args.ann3}.ods")
    annkey = pd.read_csv(key_path)
    preds = pd.read_json(preds_path)
    
    annset1 = process(anndata1, annkey)
    annset2 = process(anndata2, annkey)
    annset3 = process(anndata3, annkey)
    
    summary = pd.DataFrame(
            {
                "annotator1": annset1["selection"].value_counts(),
                "annotator2": annset2["selection"].value_counts(),
                "annotator3": annset3["selection"].value_counts(),
                }
            )
    summary.to_json(f"./annotation/{args.model}_summary.json", indent=4)

    annotations = join_annotations(annset1, annset2, annset3)
    full_set = pd.DataFrame(
            {
                "word": preds["word"],
                "example": preds["example"],
                "metaphorical_gloss": preds["metaphorical_gloss"],
                "literal_gloss": preds["literal_gloss"],
                "pred": preds["pred"],
                "more_similar_to": annotations,
                }
            )
    full_set.to_json(f"./annotation/{args.model}_full.json", indent=4,
            orient="records")

if __name__ == "__main__":
    main()
