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
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model')
    #
    # args = parser.parse_args()

    # make_strategy_file(args.model)

    with open('save/strategy-mcts_v21.pkl', 'rb') as f:
        Q = pickle.load(f)

    strategy = {}
    for i in range(100):
        for j in range(100):

            if (i, j) in Q:
                strategy[(i, j)] = np.argmax(Q[(i, j)])
            else:
                strategy[(i, j)] = contest_strategy(i, j)

    with open('no_import_strategy.py', 'w') as f:
        f.write('strategy = {\n')
        for i in range(100):
            res = [f'({i}, {j}): {strategy[(i, j)]}, ' for j in range(100)]
            f.write('    ' + ''.join(res) + '\n')
        f.write('}\n')

        s = 'def final_strategy(s1, s2):\n' + \
            '    return strategy[(s1, s2)]\n'

        f.write(s)
