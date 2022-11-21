import pandas as pd
import numpy as np


def print_monom(predictions, cmp_file):
    cmp = np.array(pd.read_csv(cmp_file, header=None)[0])
    scores = []
    for cmp_str in cmp:
        if '>' in cmp_str:
            left, right = cmp_str.split('>')
            left = int(left)
            right = int(right)
            if 0 <= left < len(predictions) and 0 <= right < len(predictions):
                scores.append(int(predictions[left] >= predictions[right]))
    scores = np.array(scores)
    print("Median: {}".format(np.median(scores)))
    print("90th percentile: {}".format(np.percentile(scores, 90)))
    print("95th percentile: {}".format(np.percentile(scores, 95)))
    print("99th percentile: {}".format(np.percentile(scores, 99)))
    print("Max: {}".format(np.max(scores)))
    print("Mean: {}".format(np.mean(scores)))
