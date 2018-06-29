import ray
import time
import tqdm
import pickle
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
        results = ray.get(
            [get_trace.remote(values, time_limit) for _ in range(cores)])

        for traces in results:
            for score, trace in traces:
                for s, p, a in trace:
                    if p == 0:
                        values[s].update(a, score)
                    if p == 1:
                        values[tuple(reversed(s))].update(a, 1 - score)

    return values


@ray.remote
def get_trace_v2(values):
    env = Env()
    return get_a_trace(values, env)


def UCT_parallel_v2(values, iters=500000, cores=4):
    for i in tqdm.tqdm(range(iters)):
        traces = ray.get([get_trace_v2.remote(values) for _ in range(cores)])

        for trace, score in traces:
            print(score, trace)
            for s, p, a in trace:
                if p == 0:
                    values[s].update(a, score)
                if p == 1:
                    values[tuple(reversed(s))].update(a, 1 - score)

    return values


def save_values(values, filename):
    v = {}
    for s in values:
        d = values[s].tried
        v[s] = d
    with open(filename, 'wb') as f:
        pickle.dump(v, f)


def load_values(filename):
    with open(filename, 'rb') as f:
        v = pickle.load(f)

    values = defaultdict(Node)
    for s in v:
        values[s].restore(v[s])

    return values


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', type=int, default=4)
    parser.add_argument('--time_limit', type=float, default=0.1)
    parser.add_argument('--init_iters', type=int, default=100000)
    parser.add_argument('--iters', type=int, default=5000)

    args = parser.parse_args()

    ray.init()

    values = defaultdict(Node)
    print('Initialize values...')
    values = UCT(values, Env(), args.init_iters)
    save_values(values, 'save/mcts_v0.pkl')

    print('Parallely update values...')
    values = UCT_parallel(
        values, args.iters, cores=args.cores, time_limit=args.time_limit)
    # values = UCT_parallel_v2(values, args.iters, cores=args.cores)

    save_values(values, 'save/mcts_v1.pkl')

    v = {}
    for s in values:
        d = values[s].tried
        v[s] = max(ACTIONSPACE, key=lambda a: d[a]['wins'] / d[a]['visits'])
