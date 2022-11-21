import pandas as pd
import numpy as np


def print_cmp_score(predictions, cmp_file):
    cmp = pd.read_csv(cmp_file, header=None).data
    scores = []
    for cmp_str in cmp:
        if '>' in cmp_str:
            left, right = cmp_str.split('>')
            left = int(left)
            right = int(right)
            scores.append(int(predictions[left] > predictions[right]))
    scores = np.array(scores)

    print('(MonoM (Monotonicity matching score)')
    print(" Median: {}".format(np.median(scores)))
    print(" 90th percentile: {}".format(np.percentile(scores, 90)))
    print(" 95th percentile: {}".format(np.percentile(scores, 95)))
    print(" 99th percentile: {}".format(np.percentile(scores, 99)))
    print(" Max: {}".format(np.max(scores)))
    print(" Mean: {}".format(np.mean(scores)))
