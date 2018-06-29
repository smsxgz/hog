import ray
import time
import copy
from env import Env
from collections import defaultdict

ACTIONSPACE = list(range(11))

from mcts import Node
from mcts import get_a_trace
from mcts import save_values, load_values


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
        if i % 100 == 0:
            print('{}th-iter / {}iters'.format(i, iters))

        tmp_values = copy.deepcopy(values)
        results = [
            get_trace.remote(tmp_values, time_limit) for _ in range(cores)
        ]

        for task in results:
            traces = ray.get(task)
            for score, trace in traces:
                for s, p, a in trace:
                    if p == 0:
                        values[s].update(a, score)
                    if p == 1:
                        values[tuple(reversed(s))].update(a, 1 - score)

    return values


@ray.remote
def get_trace_v2(values, mini_iter=10):
    env = Env()
    traces = [get_a_trace(values, env) for _ in range(mini_iter)]
    return traces


def UCT_parallel_v2(values, iters=500000, cores=4, mini_iter=10):
    for i in range(iters):
        if i % 100 == 0:
            print('{}th-iter / {}iters'.format(i, iters))

        tmp_values = copy.deepcopy(values)
        results = [get_trace_v2.remote(tmp_values) for _ in range(cores)]

        for task in results:
            traces = ray.get(task)
            for trace, score in traces:
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
    parser.add_argument('--mini_iter', type=int, default=10)
    # parser.add_argument('--init_iters', type=int, default=100000)
    parser.add_argument('--iters', type=int, default=5000)

    args = parser.parse_args()

    ray.init()

    print('Restore values...')
    values = load_values('save/mcts_v0.pkl')

    print('Parallely update values...')
    # values = UCT_parallel(
    #     values, args.iters, cores=args.cores, time_limit=args.time_limit)
    values = UCT_parallel_v2(
        values, args.iters, cores=args.cores, mini_iter=args.mini_iter)

    save_values(values, 'para-mcts_v1.pkl')
