import ray
import time
from env import Env
from collections import defaultdict

ACTIONSPACE = list(range(11))

from mcts import Node
from mcts import UCT
from mcts import get_a_trace


@ray.remote
def get_trace(values, time_limit=0.1):
    env = Env()
    traces = []
    start = time.time()

    while True:
        trace, score = get_a_trace(values, env)

        traces.append([score, trace])
        if time.time() - start > time_limit:
            break

    return traces


def UCT_parallel(values, iters=500000, cores=4, time_limit=0.1):
    for i in range(iters):
        results = [get_trace.remote(values) for _ in range(cores)]
        results = [ray.get(task) for task in results]

        for traces in results:
            for score, trace in traces:
                for s, p, a in trace:
                    if p == 0:
                        values[s].update(a, score)
                    if p == 1:
                        values[tuple(reversed(s))].update(a, 1 - score)

    return values


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', type=int, default=4)
    parser.add_argument('--time_limit', type=float, default=0.1)
    parser.add_argument('--iters', type=int, default=5000)

    args = parser.parse_args()

    ray.init(num_cpus=args.cores)

    values = defaultdict(Node)
    print('Init values...')
    values = UCT(values, Env(), 500000)
    print('Parallel update values...')
    values = UCT_parallel(
        values, args.iters, cores=args.cores, time_limit=args.time_limit)

    v = {}
    for s in values:
        d = values[s].tried
        v[s] = d

    import pickle
    with open('save/mcts.pkl', 'wb') as f:
        pickle.dump(v, f)

    v = {}
    for s in values:
        d = values[s].tried
        v[s] = max(ACTIONSPACE, key=lambda a: d[a]['wins'] / d[a]['visits'])
