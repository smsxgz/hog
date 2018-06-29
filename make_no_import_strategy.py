import json
import pickle
import numpy as np
from hog_contest import final_strategy as contest_strategy


def make_strategy_file(file):
    with open(file, 'rb') as f:
        Q = pickle.load(f)

    strategy = {}
    for i in range(100):
        for j in range(100):

            if (i, j) in Q:
                strategy[(i, j)] = np.argmax(Q[(i, j)])
            else:
                strategy[(i, j)] = contest_strategy(i, j)

    with open('no_import_strategy.py', 'w') as f:
        json.dump(strategy, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')

    args = parser.parse_args()

    make_strategy_file(args.model)
