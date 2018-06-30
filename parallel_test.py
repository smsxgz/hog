import ray
import pickle
import numpy as np
from collections import Counter

from env import DeterministicOpponentEnv

from hog import final_strategy
from hog_contest import final_strategy as contest_strategy


def simple_strategy(*args):
    return 4


@ray.remote
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


def parallel_test(strategy,
                  base_strategy=simple_strategy,
                  rounds=10000,
                  cores=4):
    results = ray.get([
        test.remote(strategy, base_strategy, rounds // cores)
        for _ in range(cores)
    ])
    return sum(results) / len(results)


def get_strategy(file):
    with open(file, 'rb') as f:
        Q = pickle.load(f)

    def strategy(*state):
        if state in Q:
            return np.argmax(Q[state])
        else:
            return np.random.randint(11)

    return strategy


def get_vote_strategy(file):

    tmp1 = file.split('.')
    tmp2 = tmp1[0].split('_')
    name = tmp2[0] + '_v{}.' + tmp1[1]
    i = int(tmp2[1][1:])
    file_list = [name.format(j) for j in range(i - 9, i + 1)]

    Q_list = []
    for file in file_list:
        with open(file, 'rb') as f:
            Q_list.append(pickle.load(f))

    A = {}
    for i in range(100):
        for j in range(100):
            A[(i, j)] = Counter(
                [np.argmax(Q[(i, j)]) for Q in Q_list]).most_common(1)[0][0]

    def strategy(*state):
        return A[state]

    return strategy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--rounds', type=int, default=10000)
    parser.add_argument('--cores', type=int, default=4)
    parser.add_argument('--vote', action='store_true', default=False)

    args = parser.parse_args()

    tmp1 = args.model.split('.')
    tmp2 = tmp1[0].split('_')
    name = tmp2[0] + '_v{}.' + tmp1[1]

    if args.vote:
        print('Vote strategy:')
        strategy = get_vote_strategy(args.model)
    else:
        strategy = get_strategy(args.model)

    ray.init(num_cpus=args.cores)
    print("Vs simple strategy: {:.4f}".format(
        parallel_test(strategy, rounds=args.rounds, cores=args.cores)))
    print("Vs hog strategy: {:.4f}".format(
        parallel_test(
            strategy,
            base_strategy=final_strategy,
            rounds=args.rounds,
            cores=args.cores)))
    print("Vs hog_contest strategy: {:.4f}".format(
        parallel_test(
            strategy,
            base_strategy=contest_strategy,
            rounds=args.rounds,
            cores=args.cores)))

    print("Vs mcts_base strategy: {:.4f}".format(
        parallel_test(
            strategy,
            base_strategy=get_strategy('save/strategy-mcts_base.pkl'),
            rounds=args.rounds,
            cores=args.cores)))

    try:
        i = int(tmp2[1][1:])
        for j in range(i - 1, i - 6, -1):
            print("Vs {}-th strategy: {:.4f}".format(
                j,
                parallel_test(
                    strategy,
                    base_strategy=get_strategy(name.format(j)),
                    rounds=args.rounds,
                    cores=args.cores)))

    except Exception:
        pass
