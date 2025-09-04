import argparse
import os 
import numpy as np
import pandas as pd
from scipy.stats import beta
from utils.constants import SCORE_RANGE


def scale_scores(
        scores: np.array,
        min_score: int,
        max_score: int,
):
    return (scores + 0.5 - min_score) / (max_score - min_score + 1)

if __name__ == '__main__':

    dataset_name = 'ASAP'
    beta_statistics = {
        dataset_name: {},
    }
    for prompt_id in [1, 2, 3, 4, 5, 6, 7, 8]:
        train_path = f"./assets/{dataset_name}/cross-prompt-individual/prompt_{prompt_id}/train.tsv"
        dev_path = f"./assets/{dataset_name}/cross-prompt-individual/prompt_{prompt_id}/dev.tsv"
        train_df, dev_df = pd.read_csv(train_path, sep='\t'), pd.read_csv(dev_path, sep='\t')
        df = pd.concat([train_df, dev_df], axis=0)
        min_score = SCORE_RANGE[dataset_name][prompt_id][0]
        max_score = SCORE_RANGE[dataset_name][prompt_id][1]
        scaled_scores = scale_scores(df['score'], min_score, max_score)
        a, b, loc, scale = beta.fit(scaled_scores, floc=0, fscale=1)
        beta_statistics[dataset_name][prompt_id] = (float(a), float(b))

    print(f"beta statistics of {dataset_name}:")
    print(f"{beta_statistics}") 