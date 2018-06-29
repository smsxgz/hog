import pickle
import numpy as np

from env import DeterministicOpponentEnv

from hog import final_strategy
from hog_contest import final_strategy as contest_strategy


def simple_strategy(*args):
    return 4


def test(strategy, base_strategy=simple_strategy, rounds=10000):
    win = 0
    env = DeterministicOpponentEnv(base_strategy)
    for _ in range(rounds):
        s = env.reset()
        while True:
            a = strategy(*s)
            s, r, d, _ = env.step(a)
            if d:
                if r > 0:
                    win += 1
                break
    return win / rounds


def get_strategy(file):
    with open(file, 'rb') as f:
        Q = pickle.load(f)

    def strategy(*state):
        if state in Q:
            return np.argmax(Q[state])
        else:
            return np.random.randint(11)

    return strategy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--rounds', type=int, default=10000)

    args = parser.parse_args()

    tmp1 = args.model.split('.')
    tmp2 = tmp1[0].split('_')
    name = tmp2[0] + '_v{}.' + tmp1[1]
    i = int(tmp2[1][1:])

    strategy = get_strategy(args.model)

    print("Vs simple strategy: {}".format(test(strategy, rounds=args.rounds)))
    print("Vs hog strategy: {}".format(
        test(strategy, base_strategy=final_strategy, rounds=args.rounds)))
    print("Vs hog_contest strategy: {}".format(
        test(strategy, base_strategy=contest_strategy, rounds=args.rounds)))

    for j in range(i):
        print("Vs {}-th strategy: {}".format(j,
                                             test(
                                                 strategy,
                                                 base_strategy=get_strategy(
                                                     name.format(j)),
                                                 rounds=args.rounds)))
