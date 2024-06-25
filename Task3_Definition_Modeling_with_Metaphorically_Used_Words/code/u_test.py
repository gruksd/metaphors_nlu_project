import pandas as pd
from scipy.stats import mannwhitneyu
import sys

def main():
    data = pd.read_json(sys.argv[1])

    bleu = mannwhitneyu(data["bleu_met"], data["bleu_lit"])
    rouge_l = mannwhitneyu(data["rougeL_met"], data["rougeL_lit"])
    bert_f1 = mannwhitneyu(data["bertF1_met"], data["bertF1_lit"])

    print(f"{bleu=}")
    print(f"{rouge_l=}")
    print(f"{bert_f1=}")

if __name__ == "__main__":
    main()
