import ray
import time
from env import Env
import numpy as np
from collections import defaultdict

ACTIONSPACE = list(range(11))


class Node:
    def __init__(self):
        self.untried = ACTIONSPACE.copy()
        self.tried = {}
        self.visits = 0

    def update(self, action, result):
        self.visits += 1
        if action not in self.tried:
            self.tried[action] = {'visits': 0, 'wins': 0}
            self.untried.remove(action)
        self.tried[action]['visits'] += 1
        self.tried[action]['wins'] += result

    def get_weights(self, action):
        d = self.tried[action]
        return d['wins'] / d['visits'] + np.sqrt(
            2 * np.log(self.visits) / d['visits'])

    def UCTselect(self):
        assert self.untried == []
        return max(ACTIONSPACE, key=lambda a: self.get_weights(a))


def get_state_key(state, player):
    if player == 0:
        return state
    elif player == 1:
        return tuple(reversed(state))


def wrapper_step(env, action):
    s, p, w = env.step(action)
    done = False
    score = None
    if w is not None:
        done = True
        score = 1 - w
    s_key = get_state_key(s, p)
    return s, p, done, score, s_key


@ray.remote
def get_trace(values, time_limit=10):
    env = Env()
    traces = []
    start = time.time()

    while True:
        s, p = env.reset()
        s_key = get_state_key(s, p)

        trace = []
        done = False
        while values[s_key].untried == [] and not done:
            a = values[s_key].UCTselect()
            trace.append((s, p, a))
            s, p, done, score, s_key = wrapper_step(env, a)

        if values[s_key].untried != [] and not done:
            a = np.random.choice(values[s_key].untried)
            trace.append((s, p, a))
            s, p, done, score, s_key = wrapper_step(env, a)

        while not done:
            _, _, done, score, _ = wrapper_step(env,
                                                np.random.choice(ACTIONSPACE))

        traces.append([score, trace])
        if time.time() - start > time_limit:
            break

    return traces


def UCT(values, iters=500000, cores=4, time_limit=10):
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
    parser.add_argument('--time_limit', type=int, default=10)
    parser.add_argument('--iters', type=int, default=500)

    args = parser.parse_args()

    ray.init(num_cpus=args.cores)

    values = defaultdict(Node)
    values = UCT(
        values, args.iters, cores=args.cores, time_limit=args.time_limit)

    v = {}
    for s in values:
        d = values[s].tried
        v[s] = max(ACTIONSPACE, key=lambda a: d[a]['wins'] / d[a]['visits'])

    import pickle
    with open('save/mcts.pkl', 'wb') as f:
        pickle.dump(v, f)
