import json
import pandas as pd
import re

def get_definitions(contextual, more_basic, paraphrases):
    # If there are no sense annotations for either of the meanings,
    # return nothing.
    if (contextual["sense"] == "0"
            or set(mng["sense"] for mng in more_basic) == {"0"}):
        return
    # If neither definition is empty and the definitions are not the same,
    # return them.
    elif ("" not in (contextual["definition"], more_basic[0]["definition"])
            and contextual["definition"] != more_basic[0]["definition"]):
        return (contextual["definition"], more_basic[0]["definition"])
    # If there is more than one basic meaning annotation,
    # see if the sense annotation for the contextual meaning is included
    # in them. If so, use that for the contextual meaning and another
    # one for the basic meaning.
    elif len(more_basic) > 1:
        possible_sense = contextual["sense"].strip("-")
        for mng in more_basic:
            if possible_sense == mng["sense"]:
                contextual_def = mng["definition"]
                more_basic.remove(mng)
        return (contextual_def, more_basic[0]["definition"])
    # Otherwise find a definition for the contextual meaning from
    # a suitable paraphrase.
    else:
        possible_sense = contextual["sense"].strip("-")
        interpreted = "interpreted_meaning_target_word"
        parap_mngs = [p[interpreted] for p in paraphrases]
        for pm in parap_mngs:
            if pm["sense"] == possible_sense:
                return (pm["definition"], more_basic[0]["definition"])

def preprocess(examples):
    processed_examples = []

    for ex in examples:
        source_text = re.sub(r"</?em>", "", ex["source_text"])
        target_word = ex["target_word"]["lemma"]
        contextual = ex["contextual_meaning"]
        more_basic = ex["more_basic_meaning"]
        paraphrases = ex["paraphrases"]

        definitions = get_definitions(
                contextual, more_basic, paraphrases)
        if definitions == None:
            continue
        else: contextual_def, more_basic_def = definitions

        proc_ex = {
                "word": target_word,
                "example": source_text,
                "metaphorical_gloss": contextual_def,
                "literal_gloss": more_basic_def,
                }
        processed_examples.append(proc_ex)
    return pd.DataFrame(processed_examples)

def main():
    data_path = "data/metaphor-paraphrase/dataset.json"
    with open(data_path) as datafile:
        dataset = json.load(datafile)
    dataset = list(dataset.values())[0]

    processed_dataset = preprocess(dataset)
    processed_dataset.to_json("data/metaphor_paraphrase.json",
            indent=4, orient="records")

if __name__ == "__main__":
    main()
