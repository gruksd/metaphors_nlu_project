import pandas as pd
from statsmodels.stats import inter_rater
import sys

def main():
    data = pd.read_json(sys.argv[1])
    data = data.dropna()
    data = data["more_similar_to"].to_list()
    data = inter_rater.aggregate_raters(data)
    kappa = inter_rater.fleiss_kappa(data[0])
    print(f"{kappa=}")

if __name__ == "__main__":
    main()
