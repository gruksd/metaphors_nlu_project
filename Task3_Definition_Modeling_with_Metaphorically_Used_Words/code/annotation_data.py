import numpy as np
import pandas as pd
import sys
import random

def shuffle_cols(row):
    cols1 = ["metaphorical_gloss", "literal_gloss"]
    cols2 = ["metaphorical_gloss", "literal_gloss"]

    random.shuffle(cols1)
    inverted = cols1 != cols2

    contents = {
        "word": row["word"],
        "gloss1": row["pred"],
        "gloss2a": row[cols1[0]],
        "gloss2b": row[cols1[1]],
        "inverted": inverted,
        }
    return pd.Series(contents)

def main():
    data_path = sys.argv[1]
    model_name = data_path.split("_")[0].split("/")[1]

    data = pd.read_json(data_path)
    annset = data.apply(shuffle_cols, axis=1)
    annset["more_similar"] = np.nan

    annset[["word", "gloss1", "more_similar", "gloss2a", "gloss2b"]].to_csv(
            f"annotation/data/{model_name}_annset.tsv", sep="\t")
    annset["inverted"].to_csv(
            f"annotation/keys/{model_name}_annkey.csv")

if __name__ == "__main__":
    main()
