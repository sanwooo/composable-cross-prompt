import os
import pandas as pd

RESERVED_TOKEN_MAPPING = {
    "Llama-3.1-8B-Instruct": "<|finetune_right_pad_id|>",
    'Phi-4-mini-instruct' : "<|endoftext17|>",
}

RESPONSE_TEMPLATE_MAPPING = {
    "Llama-3.1-8B-Instruct": [128006, 78191, 128007, 271],
    'Phi-4-mini-instruct': [200019,],
}

SCORE_RANGE = {
    "ASAP": {
        1: (2, 12),
        2: (1, 6),
        3: (0, 3),
        4: (0, 3),
        5: (0, 4),
        6: (0, 4),
        7: (0, 30),
        8: (0, 60),
    },
    "PERSUADE2.0": {
        1: (1, 6),
        2: (1, 6),
        3: (1, 6),
        4: (1, 6),
        5: (1, 6),
        6: (1, 6),
        7: (1, 6),
        8: (1, 6),
        9: (1, 6),
        10: (1, 6),
        11: (1, 6),
        12: (1, 6),
        13: (1, 6),
        14: (1, 6),
        15: (1, 6),
    },
}

PROMPT_ID_LIST = {
    "ASAP": [1, 2, 3, 4, 5, 6, 7, 8],
    "PERSUADE2.0": [1, 3, 5, 7, 9, 11, 13, 15],
}


# this is pre-computed by src/precompute_source_statistics.py
# note that the statistics of target prompt is NOT leveraged during PIM optimization (please see src/bayesian_merge_pim.py).
BETA_STATISTICS = {
    'ASAP': {
        1: (5.955649406128633, 3.3820219431399874),
        2: (6.2996433178912135, 6.705282664316442),
        3: (2.9464327959512215, 2.048691968864633),
        4: (1.7888831722189131, 1.9073186997284732),
        5: (3.1430055315373955, 2.238585097834922),
        6: (3.320065123096264, 1.8803219164891098),
        7: (5.617391123640216, 4.916856648768847),
        8: (14.596006283502254, 9.161140792574725),
    }
}