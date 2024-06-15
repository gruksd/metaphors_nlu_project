import pandas as pd
import re

def codwoe_dataset(datafile):
    dataset = pd.read_csv(datafile)
    return dataset[["word", "example", "gloss"]]

def oxford_dataset(datafile):
    dataset = pd.read_json(datafile)
    dataset = dataset.map(lambda x: " ".join(x))
    data = {
            "word": dataset[0],
            "example": dataset[2],
            "gloss": dataset[1]
            }
    return pd.DataFrame(data)

def wn_word_regex(el):
    word_match = re.search(r"(?<=%w.)\w+", el)
    return word_match.group(0)

def wn_example_regex(row):
    word = wn_word_regex(row[0])
    return re.sub("<TRG>", word, row[1])

def wordnet_dataset(glosses_file, examples_file):
    gloss_df = pd.read_csv(glosses_file, sep="\t", header=None)
    example_df = pd.read_csv(examples_file, sep="\t", header=None)

    words = gloss_df[0].map(wn_word_regex)
    glosses = gloss_df[3]
    examples = example_df.apply(wn_example_regex, axis=1)

    data = {
            "word": words,
            "example": examples,
            "gloss": glosses,
            }
    return pd.DataFrame(data)

def main():
    codwoe_train = codwoe_dataset("data/codwoe/en.complete.csv")
    oxford_train = oxford_dataset("data/oxford/train.json")
    wordnet_train = wordnet_dataset(
            "data/wordnet/train.txt", "data/wordnet/train.eg")

    train_dataset = pd.concat(
            [codwoe_train, oxford_train, wordnet_train],
            ignore_index=True)

    oxford_val = oxford_dataset("data/oxford/valid.json")
    wordnet_val = wordnet_dataset(
            "data/wordnet/valid.txt", "data/wordnet/valid.eg")

    val_dataset = pd.concat(
            [oxford_val, wordnet_val],
            ignore_index=True)

    oxford_test = oxford_dataset("data/oxford/test.json")
    wordnet_test = wordnet_dataset(
            "data/wordnet/test.txt", "data/wordnet/test.eg")

    train_dataset.to_json("data/train.json", indent=4, orient="records")
    val_dataset.to_json("data/val.json", indent=4, orient="records")
    oxford_test.to_json("data/oxford_test.json", indent=4, orient="records")
    wordnet_test.to_json("data/wordnet_test.json", indent=4, orient="records")

if __name__ == "__main__":
    main()
